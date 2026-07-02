from __future__ import annotations

import copy
import csv
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from data_utils import build_dataloaders
from models.capra import CAPRAModel
from utils import ensure_dir, rowwise_masked_pearson, save_json


DEG_POLICY_TAG = "trainval_deg_mask_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPRA_SEED = 24
DEFAULT_CONTROL_CONTEXT_MODE = "stratified_multik_globalstd"
DEFAULT_CONTROL_CONTEXT_K_CHOICES = "2,4,8"
DEFAULT_CONTROL_CONTEXT_MEAN_LOCAL_WEIGHT = 0.70


def _build_target_gene_embedding_matrix(assets) -> np.ndarray:
    """Return GenePT vectors aligned to expression genes; missing genes stay zero."""
    table = assets.embedding_table
    table_index = set(table.index.astype(str))
    matrix = np.zeros((len(assets.var_names), int(assets.embedding_dim)), dtype=np.float32)
    for idx, gene in enumerate(assets.var_names.astype(str)):
        if gene in table_index:
            matrix[idx] = table.loc[gene].to_numpy(dtype=np.float32)
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _resolve_unified_seed(
    *,
    seed: int | None = None,
    model_init_seed: int | None = None,
    control_context_seed: int | None = None,
) -> int:
    """Resolve CAPRA's single run seed from current and compatibility API names."""
    supplied: list[tuple[str, int]] = []
    for name, value in (
        ("seed", seed),
        ("model_init_seed", model_init_seed),
        ("control_context_seed", control_context_seed),
    ):
        if value is not None:
            supplied.append((name, int(value)))
    if not supplied:
        return DEFAULT_CAPRA_SEED
    values = {value for _, value in supplied}
    if len(values) != 1:
        details = ", ".join(f"{name}={value}" for name, value in supplied)
        raise ValueError(
            "CAPRA uses one unified run seed for model initialization, "
            f"control-context sampling, and dataloader shuffling; got {details}"
        )
    return supplied[0][1]


def project_seed(cfg: Dict[str, Any]) -> int:
    """Return the unified CAPRA run seed from a config dictionary."""
    project = cfg.get("project", {})
    if "seed" in project:
        seed_value = int(project["seed"])
        mismatched = [
            f"{name}={int(project[name])}"
            for name in ("model_init_seed", "control_context_seed")
            if name in project and int(project[name]) != seed_value
        ]
        if mismatched:
            raise ValueError(
                "CAPRA config must use one unified project.seed; "
                f"got seed={seed_value} and " + ", ".join(mismatched)
            )
        return seed_value
    return _resolve_unified_seed(
        model_init_seed=project.get("model_init_seed"),
        control_context_seed=project.get("control_context_seed"),
    )


def _to_float(value: Any) -> float | None:
    """Convert a scalar-like object to a finite Python float."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().float().cpu().item()
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if np.isfinite(scalar) else None


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move tensor values in a collated batch to the selected device."""
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def resolve_capra_device(cfg: Dict[str, Any]) -> torch.device:
    """Resolve the single native PyTorch device used for a CAPRA run."""
    train_cfg = cfg["train"]
    accelerator = str(train_cfg.get("accelerator", "auto")).lower()
    devices = train_cfg.get("devices", 1)

    if accelerator == "cpu":
        return torch.device("cpu")
    if accelerator not in {"auto", "gpu", "cuda"}:
        raise ValueError("native PyTorch CAPRA backend supports accelerator='auto', 'cpu', 'gpu', or 'cuda'")

    if not torch.cuda.is_available():
        if accelerator in {"gpu", "cuda"}:
            raise RuntimeError("CUDA accelerator was requested but torch.cuda.is_available() is false")
        return torch.device("cpu")

    if isinstance(devices, int) and devices != 1:
        raise ValueError("native PyTorch CAPRA backend supports exactly one CUDA device per process")
    if isinstance(devices, str) and devices not in {"1", "auto"}:
        raise ValueError("native PyTorch CAPRA backend supports devices=1 or devices='auto'")
    if isinstance(devices, (list, tuple)) and len(devices) != 1:
        raise ValueError("native PyTorch CAPRA backend supports one CUDA device per process")
    if isinstance(devices, (list, tuple)):
        return torch.device(f"cuda:{int(devices[0])}")
    return torch.device("cuda")


def _precision_dtype(precision: Any) -> torch.dtype | None:
    """Return the autocast dtype implied by a precision config value."""
    if precision in {16, "16", "16-mixed"}:
        return torch.float16
    if precision in {"bf16", "bf16-mixed"}:
        return torch.bfloat16
    return None


def _autocast_context(device: torch.device, precision: Any):
    """Build an autocast context for CUDA mixed precision, or a no-op context."""
    dtype = _precision_dtype(precision)
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _build_grad_scaler(device: torch.device, precision: Any):
    """Create a gradient scaler only for CUDA float16 mixed precision."""
    enabled = device.type == "cuda" and _precision_dtype(precision) is torch.float16
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _limit_to_batch_count(limit: Any, loader_len: int) -> int:
    """Translate fractional or integer batch limits into a batch count."""
    if loader_len <= 0:
        return 0
    if isinstance(limit, bool):
        return loader_len
    if isinstance(limit, int):
        return max(1, min(int(limit), loader_len))
    limit_float = float(limit)
    if math.isclose(limit_float, 1.0):
        return loader_len
    if 0.0 < limit_float < 1.0:
        return max(1, min(int(math.ceil(loader_len * limit_float)), loader_len))
    if limit_float > 1.0:
        return max(1, min(int(limit_float), loader_len))
    raise ValueError("limit_*_batches must be positive")


def _limited_batches(loader: Iterable, limit: Any):
    """Yield at most the configured number of batches from a dataloader."""
    max_batches = _limit_to_batch_count(limit, len(loader))
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        yield batch_idx, batch


def _accumulate_metrics(
    sums: Dict[str, float],
    weights: Dict[str, float],
    metrics: Dict[str, Any],
    *,
    weight: float,
) -> None:
    """Accumulate finite scalar metrics using explicit sample weights."""
    for key, value in metrics.items():
        if key == "loss":
            continue
        scalar = _to_float(value)
        if scalar is None:
            continue
        sums[key] = sums.get(key, 0.0) + scalar * float(weight)
        weights[key] = weights.get(key, 0.0) + float(weight)


def _finalize_metrics(sums: Dict[str, float], weights: Dict[str, float]) -> Dict[str, float]:
    """Convert accumulated metric sums into weighted means."""
    return {
        key: sums[key] / weights[key]
        for key in sorted(sums)
        if weights.get(key, 0.0) > 0.0
    }


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return the first optimizer group learning rate for metrics output."""
    if not optimizer.param_groups:
        return float("nan")
    return float(optimizer.param_groups[0].get("lr", float("nan")))


def _format_log_scalar(value: Any) -> str:
    """Format a scalar metric for compact stdout progress logs."""
    scalar = _to_float(value)
    if scalar is None:
        return "nan"
    return f"{scalar:.4f}"


def _emit_epoch_log(
    cfg: Dict[str, Any],
    *,
    epoch: int,
    max_epochs: int,
    epoch_metrics: Dict[str, float],
    monitor: str,
    monitor_value: float,
    improved: bool,
    min_epochs: int,
    wait_count: int,
    patience: int,
) -> None:
    """Write one compact training-progress line to stdout when enabled."""
    train_cfg = cfg["train"]
    if not bool(train_cfg.get("enable_stdout_logging", True)):
        return
    log_every = max(1, int(train_cfg.get("log_every_n_epochs", 1)))
    if epoch != 1 and epoch != max_epochs and epoch % log_every != 0 and not improved:
        return
    early_stop = (
        f"early_stop=warmup({epoch}/{min_epochs})"
        if epoch < min_epochs
        else f"patience={wait_count}/{patience}"
    )
    print(
        " | ".join([
            "CAPRA training",
            f"epoch {epoch}/{max_epochs}",
            f"train_loss={_format_log_scalar(epoch_metrics.get('train_loss'))}",
            f"val_unseen_loss={_format_log_scalar(epoch_metrics.get('val_unseen_loss'))}",
            f"monitor={_format_log_scalar(monitor_value)}",
            early_stop,
        ]),
        flush=True,
    )


def _monitor_improved(value: float, best: float | None, mode: str, min_delta: float = 0.0) -> bool:
    """Return whether a monitor value improves over the current best value."""
    if best is None:
        return True
    min_delta = max(0.0, float(min_delta))
    if mode == "min":
        return value < best - min_delta
    if mode == "max":
        return value > best + min_delta
    raise ValueError("monitor_mode must be 'min' or 'max'")


def _write_metrics_csv(path: Path, rows: list[Dict[str, float]]) -> None:
    """Write compact epoch-level metrics for plotting and run auditing."""
    ensure_dir(path.parent)
    if not rows:
        return
    preferred = [
        "epoch",
        "lr",
        "train_loss",
        "train_deg_loss",
        "train_pearson_loss",
        "train_operator_reg",
        "train_top100deg_dual_objective",
        "val_unseen_loss",
        "val_unseen_deg_loss",
        "val_unseen_pearson_loss",
        "val_unseen_operator_reg",
        "val_unseen_top100deg_pearson_distance",
        "val_unseen_top100deg_dual_objective",
    ]
    discovered = sorted({key for row in rows for key in row})
    fieldnames = [key for key in preferred if key in discovered] + [
        key for key in discovered if key not in preferred
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _torch_load(path: Path, *, map_location: torch.device | str | None = None) -> Any:
    """Load a PyTorch checkpoint while supporting old and new torch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_native_checkpoint(path: Path, *, map_location: torch.device | str | None = None) -> Dict[str, Any]:
    """Load and validate a native PyTorch CAPRA checkpoint."""
    payload = _torch_load(path, map_location=map_location)
    if not isinstance(payload, dict) or payload.get("format") != "capra_native_pytorch_v1":
        raise ValueError("checkpoint is not a native PyTorch CAPRA checkpoint")
    if not isinstance(payload.get("cfg"), dict):
        raise ValueError("checkpoint is missing a CAPRA config")
    if not isinstance(payload.get("state_dict"), dict):
        raise ValueError("checkpoint is missing a model state_dict")
    return payload


def load_capra_checkpoint_config(
    checkpoint_path: str | Path,
    *,
    map_location: torch.device | str | None = "cpu",
) -> Dict[str, Any]:
    """Load only the stored CAPRA config from a native PyTorch checkpoint."""
    payload = _load_native_checkpoint(Path(checkpoint_path).expanduser().resolve(), map_location=map_location)
    return copy.deepcopy(payload["cfg"])


def _save_checkpoint(
    path: Path,
    *,
    model: "CAPRATrainingModule",
    cfg: Dict[str, Any],
    epoch: int,
    metrics: Dict[str, float],
    monitor: str,
    monitor_mode: str,
    best_score: float | None,
) -> None:
    """Persist a native PyTorch CAPRA checkpoint."""
    ensure_dir(path.parent)
    torch.save(
        {
            "format": "capra_native_pytorch_v1",
            "state_dict": model.state_dict(),
            "cfg": copy.deepcopy(cfg),
            "epoch": int(epoch),
            "metrics": copy.deepcopy(metrics),
            "monitor": str(monitor),
            "monitor_mode": str(monitor_mode),
            "best_score": best_score,
        },
        path,
    )


class CAPRATrainingModule(nn.Module):
    """Native PyTorch wrapper around `CAPRAModel` and CAPRA's three-term objective."""

    def __init__(self, cfg: Dict[str, Any], assets) -> None:
        """Instantiate model modules and bind dataset assets for training."""
        super().__init__()
        self.cfg = cfg
        self.assets = assets
        self.model = CAPRAModel(
            num_genes=assets.adata.n_vars,
            embedding_dim=assets.embedding_dim,
            latent_dim=int(cfg["model"]["latent_dim"]),
            gene_hidden_dim=int(cfg["model"]["gene_hidden_dim"]),
            prior_hidden_dim=int(cfg["model"]["prior_hidden_dim"]),
            operator_hidden_dim=int(cfg["model"]["operator_hidden_dim"]),
            operator_rank=int(cfg["model"]["operator_rank"]),
            feature_dim=int(next(iter(assets.condition_priors.values()))["features"].shape[0]),
            single_stat_dim=int(next(iter(assets.condition_priors.values()))["single_prior_stats"].shape[1]),
            dropout=float(cfg["model"]["dropout"]),
            target_gene_embeddings=torch.from_numpy(_build_target_gene_embedding_matrix(assets)),
            initialization_seed=project_seed(cfg),
        )
        self.deg_weight = float(cfg["loss"]["deg_weight"])
        self.pearson_weight = float(cfg["loss"]["pearson_weight"])
        self.operator_reg_weight = float(cfg["loss"]["operator_reg_weight"])
        self.register_buffer("global_control_mean", torch.from_numpy(assets.global_control_mean.astype(np.float32)))

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        assets: Any,
        map_location: torch.device | str | None = "cpu",
        strict: bool = True,
    ) -> "CAPRATrainingModule":
        """Load a native PyTorch CAPRA checkpoint into the model wrapper."""
        if assets is None:
            raise ValueError("assets are required when loading a CAPRA checkpoint")
        payload = _load_native_checkpoint(Path(checkpoint_path).expanduser().resolve(), map_location=map_location)
        module = cls(cfg=payload["cfg"], assets=assets)
        module.load_state_dict(payload["state_dict"], strict=strict)
        return module

    def forward(self, batch: Dict[str, torch.Tensor]):
        """Run the CAPRA model on a collated training or prediction batch."""
        return self.model(
            control_mean=batch["control_mean"],
            control_std=batch["control_std"],
            single_prior_deltas=batch["single_prior_deltas"],
            single_prior_stats=batch["single_prior_stats"],
            gene_embeddings=batch["gene_embeddings"],
            gene_mask=batch["gene_mask"],
            features=batch["features"],
            prediction_noise_seed=batch.get("prediction_noise_seed"),
        )

    def _compute_losses(
        self,
        batch: Dict[str, torch.Tensor],
        prediction: torch.Tensor,
        aux: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute CAPRA's DEG MSE, Pearson distance, and operator regularizer."""
        target = batch["target"]
        deg_mask = batch["deg_mask"]
        deg_denom = deg_mask.sum(dim=1).clamp_min(1.0)
        pred_delta = prediction - self.global_control_mean.unsqueeze(0)
        target_delta = target - self.global_control_mean.unsqueeze(0)
        deg_loss = ((((prediction - target) ** 2) * deg_mask).sum(dim=1) / deg_denom).mean()
        pearson_loss = (1.0 - rowwise_masked_pearson(pred_delta, target_delta, deg_mask)).mean()
        operator_reg = aux["operator_energy"].mean()
        total_loss = self.deg_weight * deg_loss + self.pearson_weight * pearson_loss + self.operator_reg_weight * operator_reg
        return {
            "loss": total_loss,
            "deg_loss": deg_loss.detach(),
            "pearson_loss": pearson_loss.detach(),
            "operator_reg": operator_reg.detach(),
            "top100deg_pearson_distance": pearson_loss.detach(),
            "top100deg_dual_objective": (deg_loss + pearson_loss).detach(),
        }

    def _compute_unseen_condition_validation(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Evaluate a held-out condition using test-like condition-mean prediction."""
        model_batch = {
            "control_mean": batch["control_mean"].squeeze(0),
            "control_std": batch["control_std"].squeeze(0),
            "single_prior_deltas": batch["single_prior_deltas"].squeeze(0),
            "single_prior_stats": batch["single_prior_stats"].squeeze(0),
            "gene_embeddings": batch["gene_embeddings"].squeeze(0),
            "gene_mask": batch["gene_mask"].squeeze(0),
            "features": batch["features"].squeeze(0),
        }
        if "prediction_noise_seed" in batch:
            model_batch["prediction_noise_seed"] = batch["prediction_noise_seed"].squeeze(0)
        row_count = int(model_batch["control_mean"].shape[0])
        for key, value in list(model_batch.items()):
            if key in {"control_mean", "control_std", "prediction_noise_seed"}:
                continue
            if value.dim() > 0 and value.shape[0] == 1 and row_count > 1:
                model_batch[key] = value.expand((row_count,) + tuple(value.shape[1:]))
        prediction, aux = self(model_batch)
        prediction_mean = prediction.mean(dim=0, keepdim=True)
        target_mean = batch["target_mean"]
        if target_mean.dim() == 1:
            target_mean = target_mean.unsqueeze(0)
        deg_mask = batch["deg_mask"]
        if deg_mask.dim() == 1:
            deg_mask = deg_mask.unsqueeze(0)
        loss_batch = {
            "target": target_mean.to(prediction_mean.device, non_blocking=True),
            "deg_mask": deg_mask.to(prediction_mean.device, non_blocking=True),
        }
        return self._compute_losses(loss_batch, prediction_mean, aux)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Run one cell-level supervised training step."""
        prediction, aux = self(batch)
        losses = self._compute_losses(batch, prediction, aux)
        return {
            "loss": losses["loss"],
            "train_loss": losses["loss"].detach(),
            "train_deg_loss": losses["deg_loss"],
            "train_pearson_loss": losses["pearson_loss"],
            "train_operator_reg": losses["operator_reg"],
            "train_top100deg_pearson_distance": losses["top100deg_pearson_distance"],
            "train_top100deg_dual_objective": losses["top100deg_dual_objective"],
        }

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Run one unseen-condition validation step."""
        losses = self._compute_unseen_condition_validation(batch)
        return {
            "val_unseen_loss": losses["loss"].detach(),
            "val_unseen_deg_loss": losses["deg_loss"],
            "val_unseen_pearson_loss": losses["pearson_loss"],
            "val_unseen_operator_reg": losses["operator_reg"],
            "val_unseen_top100deg_pearson_distance": losses["top100deg_pearson_distance"],
            "val_unseen_top100deg_dual_objective": losses["top100deg_dual_objective"],
        }

    def configure_optimizers(self):
        """Build the AdamW optimizer with CUDA fast paths when available."""
        optimizer_kwargs = {
            "lr": float(self.cfg["train"]["learning_rate"]),
            "weight_decay": float(self.cfg["train"]["weight_decay"]),
        }
        first_parameter = next(self.parameters(), None)
        parameters_on_cuda = first_parameter is not None and first_parameter.device.type == "cuda"
        if parameters_on_cuda:
            gradient_clip_val = float(self.cfg["train"].get("gradient_clip_val", 0.0))
            if gradient_clip_val <= 0.0:
                try:
                    return AdamW(self.parameters(), fused=True, **optimizer_kwargs)
                except (TypeError, RuntimeError):
                    pass
            try:
                return AdamW(self.parameters(), foreach=True, **optimizer_kwargs)
            except (TypeError, RuntimeError):
                pass
        return AdamW(self.parameters(), **optimizer_kwargs)


def train_one_epoch(
    model: CAPRATrainingModule,
    dataloader,
    optimizer: torch.optim.Optimizer,
    *,
    cfg: Dict[str, Any],
    device: torch.device,
    scaler,
    epoch_index: int,
) -> Dict[str, float]:
    """Train the native PyTorch CAPRA module for one epoch."""
    model.train()
    dataset = getattr(dataloader, "dataset", None)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(int(epoch_index))
    sums: Dict[str, float] = {}
    weights: Dict[str, float] = {}
    precision = cfg["train"]["precision"]
    gradient_clip_val = float(cfg["train"].get("gradient_clip_val", 0.0))
    for batch_idx, batch in _limited_batches(dataloader, cfg["train"]["limit_train_batches"]):
        batch = _move_batch_to_device(batch, device)
        batch_size = int(batch["target"].shape[0])
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, precision):
            outputs = model.training_step(batch, batch_idx)
            loss = outputs["loss"]
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if gradient_clip_val > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip_val > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            optimizer.step()
        _accumulate_metrics(sums, weights, outputs, weight=batch_size)
    return _finalize_metrics(sums, weights)


def validate_capra(
    model: CAPRATrainingModule,
    dataloader,
    *,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    """Validate CAPRA with the unseen-condition mean protocol."""
    model.eval()
    sums: Dict[str, float] = {}
    weights: Dict[str, float] = {}
    precision = cfg["train"]["precision"]
    with torch.no_grad():
        for batch_idx, batch in _limited_batches(dataloader, cfg["train"]["limit_val_batches"]):
            batch = _move_batch_to_device(batch, device)
            with _autocast_context(device, precision):
                outputs = model.validation_step(batch, batch_idx)
            _accumulate_metrics(sums, weights, outputs, weight=1.0)
    return _finalize_metrics(sums, weights)


def build_capra_config(
    *,
    study_name: str = "capra_study",
    seed: int | None = None,
    model_init_seed: int | None = None,
    control_context_seed: int | None = None,
    results_root: str | Path | None = None,
    output_name: str | None = None,
    fast_dev_run: bool = False,
    topk_deg: int = 100,
    train_batch_size: int = 192,
    val_n_pred: int | None = None,
    eval_batch_size: int = 2048,
    num_workers: int = 8,
    pin_memory: bool = True,
    knn_topk: int = 5,
    knn_temperature: float = 12.0,
    latent_dim: int = 288,
    gene_hidden_dim: int = 288,
    prior_hidden_dim: int = 128,
    operator_hidden_dim: int = 288,
    operator_rank: int = 10,
    dropout: float = 0.12,
    deg_weight: float = 1.0,
    pearson_weight: float = 0.60,
    operator_reg_weight: float = 0.025,
    accelerator: str = "auto",
    devices: int | str = 1,
    precision: str | int = "16-mixed",
    allow_tf32: bool = True,
    min_epochs: int = 10,
    max_epochs: int = 80,
    patience: int = 10,
    monitor: str = "val_unseen_top100deg_pearson_distance",
    monitor_mode: str = "min",
    monitor_min_delta: float = 1e-4,
    learning_rate: float = 6e-4,
    weight_decay: float = 1e-4,
    gradient_clip_val: float = 1.0,
    limit_train_batches: float = 1.0,
    limit_val_batches: float = 1.0,
    control_context_mode: str = DEFAULT_CONTROL_CONTEXT_MODE,
    control_context_k_choices: Any = DEFAULT_CONTROL_CONTEXT_K_CHOICES,
    control_context_mean_local_weight: float = DEFAULT_CONTROL_CONTEXT_MEAN_LOCAL_WEIGHT,
    enable_stdout_logging: bool = True,
    log_every_n_epochs: int = 1,
) -> Dict[str, Any]:
    """Build a complete CAPRA configuration from explicit Python arguments."""
    resolved_seed = _resolve_unified_seed(
        seed=seed,
        model_init_seed=model_init_seed,
        control_context_seed=control_context_seed,
    )
    val_n_pred = 16 if fast_dev_run else (128 if val_n_pred is None else int(val_n_pred))
    min_epochs = 1 if fast_dev_run else max(0, int(min_epochs))
    max_epochs = max(min_epochs, 1 if fast_dev_run else int(max_epochs))
    return {
        "project": {
            "study_name": str(study_name),
            "seed": int(resolved_seed),
            "output_name": output_name or "capra_response_operator",
        },
        "paths": {
            "results_root": str(Path(results_root or (PROJECT_ROOT / "tmp" / "capra_results")).resolve()),
        },
        "data": {
            "topk_deg": int(topk_deg),
            "train_batch_size": int(train_batch_size),
            "val_n_pred": int(val_n_pred),
            "eval_batch_size": int(eval_batch_size),
            "num_workers": 0 if fast_dev_run else int(num_workers),
            "pin_memory": bool(pin_memory),
            "knn_topk": int(knn_topk),
            "knn_temperature": float(knn_temperature),
            "control_context_mode": str(control_context_mode),
            "control_context_k_choices": control_context_k_choices,
            "control_context_mean_local_weight": float(control_context_mean_local_weight),
        },
        "model": {
            "latent_dim": int(latent_dim),
            "gene_hidden_dim": int(gene_hidden_dim),
            "prior_hidden_dim": int(prior_hidden_dim),
            "operator_hidden_dim": int(operator_hidden_dim),
            "operator_rank": int(operator_rank),
            "dropout": float(dropout),
        },
        "loss": {
            "deg_weight": float(deg_weight),
            "pearson_weight": float(pearson_weight),
            "operator_reg_weight": float(operator_reg_weight),
        },
        "train": {
            "accelerator": accelerator,
            "devices": devices,
            "precision": precision,
            "allow_tf32": bool(allow_tf32),
            "min_epochs": int(min_epochs),
            "max_epochs": int(max_epochs),
            "patience": int(patience),
            "monitor": monitor,
            "monitor_mode": monitor_mode,
            "monitor_min_delta": float(monitor_min_delta),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "gradient_clip_val": float(gradient_clip_val),
            "limit_train_batches": 2 if fast_dev_run else float(limit_train_batches),
            "limit_val_batches": 1 if fast_dev_run else float(limit_val_batches),
            "fast_dev_run": bool(fast_dev_run),
            "enable_stdout_logging": bool(enable_stdout_logging),
            "log_every_n_epochs": int(log_every_n_epochs),
        },
        "run": {
            "method_name": "capra",
        },
    }


def fit_capra(cfg: Dict[str, Any], assets: Any) -> Dict[str, Any]:
    """Train CAPRA with native PyTorch and return the best checkpoint-loaded module."""
    if assets is None:
        raise ValueError("fit_capra requires prebuilt DatasetAssets")
    if str(cfg["train"]["monitor_mode"]) not in {"min", "max"}:
        raise ValueError("monitor_mode must be 'min' or 'max'")

    dataloaders = build_dataloaders(assets, cfg)
    run_dir = Path(cfg["paths"]["results_root"]).resolve() / cfg["project"]["output_name"]
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    metrics_csv = run_dir / "logs" / "metrics.csv"

    device = resolve_capra_device(cfg)
    model = CAPRATrainingModule(cfg=cfg, assets=assets).to(device)
    optimizer = model.configure_optimizers()
    scaler = _build_grad_scaler(device, cfg["train"]["precision"])

    monitor = str(cfg["train"]["monitor"])
    monitor_mode = str(cfg["train"]["monitor_mode"])
    monitor_min_delta = float(cfg["train"].get("monitor_min_delta", 1e-4))
    min_epochs = int(cfg["train"]["min_epochs"])
    max_epochs = int(cfg["train"]["max_epochs"])
    patience = int(cfg["train"]["patience"])
    best_score: float | None = None
    best_epoch: int | None = None
    early_stop_score: float | None = None
    wait_count = 0
    epoch_rows: list[Dict[str, float]] = []
    best_path = checkpoint_dir / "best.ckpt"
    last_path = checkpoint_dir / "last.ckpt"
    epochs_completed = 0
    stopped_epoch: int | None = None

    for epoch_index in range(max_epochs):
        epoch = epoch_index + 1
        train_metrics = train_one_epoch(
            model,
            dataloaders["train"],
            optimizer,
            cfg=cfg,
            device=device,
            scaler=scaler,
            epoch_index=epoch_index,
        )
        val_metrics = validate_capra(
            model,
            dataloaders["val"],
            cfg=cfg,
            device=device,
        )
        epoch_metrics: Dict[str, float] = {
            "epoch": float(epoch),
            "lr": _current_lr(optimizer),
            **train_metrics,
            **val_metrics,
        }
        monitor_value = _to_float(epoch_metrics.get(monitor))
        if monitor_value is None:
            raise KeyError(f"monitor metric {monitor!r} was not produced by native CAPRA validation")
        if not np.isfinite(monitor_value):
            raise ValueError(f"monitor metric {monitor!r} is non-finite at epoch {epoch}: {monitor_value}")

        improved = _monitor_improved(monitor_value, best_score, monitor_mode, monitor_min_delta)
        if improved:
            best_score = monitor_value
            best_epoch = epoch
            _save_checkpoint(
                best_path,
                model=model,
                cfg=cfg,
                epoch=epoch,
                metrics=epoch_metrics,
                monitor=monitor,
                monitor_mode=monitor_mode,
                best_score=best_score,
            )

        if epoch >= min_epochs:
            if _monitor_improved(monitor_value, early_stop_score, monitor_mode, monitor_min_delta):
                early_stop_score = monitor_value
                wait_count = 0
            else:
                wait_count += 1

        _save_checkpoint(
            last_path,
            model=model,
            cfg=cfg,
            epoch=epoch,
            metrics=epoch_metrics,
            monitor=monitor,
            monitor_mode=monitor_mode,
            best_score=best_score,
        )
        epoch_rows.append(epoch_metrics)
        _write_metrics_csv(metrics_csv, epoch_rows)
        _emit_epoch_log(
            cfg,
            epoch=epoch,
            max_epochs=max_epochs,
            epoch_metrics=epoch_metrics,
            monitor=monitor,
            monitor_value=monitor_value,
            improved=improved,
            min_epochs=min_epochs,
            wait_count=wait_count,
            patience=patience,
        )

        epochs_completed = epoch
        if epoch >= min_epochs and wait_count >= patience:
            stopped_epoch = epoch
            break

    best_path_str = str(best_path) if best_path.exists() else ""
    if best_path_str:
        model = CAPRATrainingModule.load_from_checkpoint(
            best_path,
            assets=assets,
            map_location=device,
        ).to(device)

    metrics = validate_capra(
        model,
        dataloaders["val"],
        cfg=cfg,
        device=device,
    )

    metrics_payload = {
        "study_name": cfg["project"].get("study_name", "capra_study"),
        "seed": project_seed(cfg),
        "seed_policy": "unified_run_seed_v1",
        "method_name": cfg["run"]["method_name"],
        "deg_policy": DEG_POLICY_TAG,
        "fast_dev_run": bool(cfg["train"]["fast_dev_run"]),
        "fit_skipped": False,
        "training_backend": "native_pytorch",
        "best_checkpoint": best_path_str,
        "last_checkpoint": str(last_path) if last_path.exists() else "",
        "best_epoch": best_epoch,
        "best_monitor": best_score,
        "monitor_min_delta": monitor_min_delta,
        "epochs_completed": epochs_completed,
        "stopped_epoch": stopped_epoch,
        "validation_protocol": "unseen_condition_test_like_mean",
        "val_n_pred": int(cfg["data"]["val_n_pred"]),
        "dropped_embedding_conditions": getattr(assets, "dropped_embedding_conditions", {}),
        "control_context": {
            "mode": cfg["data"].get("control_context_mode", "sampled"),
            "k_choices": cfg["data"].get("control_context_k_choices", ""),
            "mean_local_weight": cfg["data"].get("control_context_mean_local_weight", 1.0),
        },
        "runtime": {
            "device": str(device),
            "precision": cfg["train"].get("precision"),
            "allow_tf32": bool(cfg["train"].get("allow_tf32", True)),
        },
        "loss_config": {
            "deg_weight": cfg["loss"].get("deg_weight", 1.0),
            "pearson_weight": cfg["loss"].get("pearson_weight", 0.60),
            "operator_reg_weight": cfg["loss"].get("operator_reg_weight", 0.025),
        },
        "val_metrics": metrics,
    }
    save_json(run_dir / "metrics.json", metrics_payload)
    return {
        "model": model,
        "assets": assets,
        "metrics": metrics_payload,
        "run_dir": run_dir,
    }
