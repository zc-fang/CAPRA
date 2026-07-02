from __future__ import annotations
import math

import hashlib
import json
from torch import nn
import random
from pathlib import Path
from typing import Any, Callable, Dict, List

import anndata as ad
import numpy as np
import torch


def ensure_dir(path: Path) -> Path:
    """Create a directory tree if it does not already exist.

    Parameters
    ----------
    path:
        Directory path to materialize.

    Returns
    -------
    Path
        The same path object, enabling concise call sites such as
        `run_dir = ensure_dir(path)`.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON payload with stable UTF-8 formatting.

    The parent directory is created automatically. This helper is used for run
    metadata and metric files, so it deliberately keeps output human-readable.
    """
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds for one CAPRA run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_condition_name(condition: str) -> str:
    """Canonicalize a perturbation condition into CAPRA's invariant format.

    Control tokens (`ctrl` and `control`) are removed. Remaining perturbation
    genes are sorted lexicographically so that pair conditions such as
    `A+B` and `B+A` map to the same key.
    """
    tokens = [item.strip() for item in str(condition).split("+") if item.strip()]
    genes = [item for item in tokens if item not in {"ctrl", "control"}]
    genes = sorted(genes)
    return "control" if not genes else "+".join(genes)


def split_condition_genes(condition: str) -> List[str]:
    """Return the perturbation genes represented by a canonical condition."""
    canonical = canonical_condition_name(condition)
    return [] if canonical == "control" else canonical.split("+")


def stable_int_from_text(text: str) -> int:
    """Map text to a deterministic 32-bit seed component with BLAKE2b."""
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _unique_canonical_conditions(conditions: List[str]) -> List[str]:
    """Return sorted unique non-control canonical condition names."""
    canonical = {
        canonical_condition_name(condition)
        for condition in conditions
        if canonical_condition_name(condition) != "control"
    }
    too_many_genes = sorted(condition for condition in canonical if len(split_condition_genes(condition)) > 2)
    if too_many_genes:
        preview = ", ".join(too_many_genes[:5])
        suffix = "" if len(too_many_genes) <= 5 else f", ... (+{len(too_many_genes) - 5} more)"
        raise ValueError(f"CAPRA simulation split supports at most two perturbation genes per condition: {preview}{suffix}")
    return sorted(canonical)


def _normalize_split_ratio(split_ratio: List[float] | tuple[float, float, float] | np.ndarray) -> np.ndarray:
    if len(split_ratio) != 3:
        raise ValueError("split_ratio must contain exactly three values")
    ratios = np.asarray(split_ratio, dtype=np.float64)
    if np.any(ratios < 0) or float(ratios.sum()) <= 0:
        raise ValueError("split ratios must be non-negative and sum to a positive value")
    return ratios / ratios.sum()


def _ratio_split_conditions(
    conditions: List[str],
    *,
    split_ratio: List[float] | tuple[float, float, float] | np.ndarray,
    seed: int,
) -> Dict[str, List[str]]:
    """Split conditions by a normalized train/validation/test ratio."""
    ratios = _normalize_split_ratio(split_ratio)
    shuffled = list(conditions)
    np.random.default_rng(int(seed)).shuffle(shuffled)
    n_total = len(shuffled)
    n_train = int(np.floor(n_total * ratios[0]))
    n_val = int(np.floor(n_total * ratios[1]))
    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }


def _condition_genes_set(conditions: List[str]) -> np.ndarray:
    """Return sorted unique perturbation genes from canonical conditions."""
    genes = sorted({gene for condition in conditions for gene in split_condition_genes(condition)})
    return np.asarray(genes, dtype=object)


def _condition_subgroups_by_seen_genes(conditions: List[str], seen_genes: set[str]) -> Dict[str, List[str]]:
    """Classify held-out conditions by how many perturbed genes are seen."""
    subgroup: Dict[str, List[str]] = {
        "combo_seen0": [],
        "combo_seen1": [],
        "combo_seen2": [],
        "unseen_single": [],
    }
    for condition in conditions:
        condition_genes = split_condition_genes(condition)
        seen_count = sum(gene in seen_genes for gene in condition_genes)
        if len(condition_genes) == 1 and seen_count == 0:
            subgroup["unseen_single"].append(condition)
        elif len(condition_genes) == 2:
            subgroup[f"combo_seen{seen_count}"].append(condition)
    return {key: sorted(set(value)) for key, value in subgroup.items()}


def _split_by_holdout_fraction(items: List[str], *, holdout_fraction: float, seed: int) -> tuple[List[str], List[str]]:
    """Return kept and held-out items after a deterministic shuffled split."""
    shuffled = list(items)
    np.random.default_rng(int(seed)).shuffle(shuffled)
    n_holdout = int(np.floor(len(shuffled) * float(holdout_fraction)))
    held_out = sorted(shuffled[:n_holdout])
    kept = sorted(shuffled[n_holdout:])
    return kept, held_out


def _build_seen_gene_simulation_split(
    conditions: List[str],
    *,
    seed: int = 1,
    split_ratio: List[float] | tuple[float, float, float] | np.ndarray = (0.8, 0.1, 0.1),
) -> Dict[str, Any]:
    """Build CAPRA's automatic single/double-gene condition split.

    Single-gene-only condition lists use the requested ratio directly. When
    double-gene conditions are present, the split holds out perturbation genes
    first, then reports held-out combinations as combo_seen0/1/2.
    """
    canonical = _unique_canonical_conditions(list(conditions))
    ratios = _normalize_split_ratio(split_ratio)
    has_combo = any(len(split_condition_genes(condition)) == 2 for condition in canonical)
    if not has_combo:
        splits = _ratio_split_conditions(canonical, split_ratio=ratios, seed=seed)
        subgroup = {
            "test_subgroup": {"unseen_single": list(splits["test"])},
            "val_subgroup": {"unseen_single": list(splits["val"])},
            "mode": "single_ratio",
        }
        return {"splits": splits, "subgroup": subgroup}

    unique_genes = _condition_genes_set(canonical)
    rng = np.random.default_rng(int(seed))
    n_seen_genes = int(np.floor(len(unique_genes) * float(ratios[0])))
    if len(unique_genes) > 0 and ratios[0] > 0:
        n_seen_genes = max(1, n_seen_genes)
    seen_genes = set(str(gene) for gene in rng.choice(unique_genes, n_seen_genes, replace=False))

    train_val_singles: List[str] = []
    seen2_candidates: List[str] = []
    test_conditions: List[str] = []
    for condition in canonical:
        genes = split_condition_genes(condition)
        seen_count = sum(gene in seen_genes for gene in genes)
        if len(genes) == 1:
            if seen_count == 1:
                train_val_singles.append(condition)
            else:
                test_conditions.append(condition)
        elif seen_count == 2:
            seen2_candidates.append(condition)
        else:
            test_conditions.append(condition)

    seen2_train_val, combo_seen2_test = _split_by_holdout_fraction(
        seen2_candidates,
        holdout_fraction=float(ratios[2]),
        seed=int(seed) + 1009,
    )
    test_conditions = sorted(set(test_conditions) | set(combo_seen2_test))
    test_subgroup = _condition_subgroups_by_seen_genes(test_conditions, seen_genes)

    val_fraction = float(ratios[1] / max(ratios[0] + ratios[1], 1e-12))
    single_train, single_val = _split_by_holdout_fraction(
        train_val_singles,
        holdout_fraction=val_fraction,
        seed=int(seed) + 2003,
    )
    combo_train, combo_val = _split_by_holdout_fraction(
        seen2_train_val,
        holdout_fraction=val_fraction,
        seed=int(seed) + 3001,
    )
    train = sorted(set(single_train) | set(combo_train))
    val = sorted(set(single_val) | set(combo_val))
    train_genes = {gene for condition in train for gene in split_condition_genes(condition)}
    val_subgroup = _condition_subgroups_by_seen_genes(val, train_genes)

    covered = set(train) | set(val) | set(test_conditions)
    missing = sorted(set(canonical) - covered)
    if missing:
        raise RuntimeError("CAPRA automatic split failed to cover conditions: " + ", ".join(missing[:10]))
    if canonical and not train:
        raise ValueError("CAPRA automatic split produced an empty train partition; provide explicit splits instead")

    splits = {
        "train": train,
        "val": val,
        "test": sorted(set(test_conditions)),
    }
    subgroup = {
        "test_subgroup": test_subgroup,
        "val_subgroup": val_subgroup,
        "mode": "seen_gene_auto",
    }
    return {"splits": splits, "subgroup": subgroup}


def split_anndata_train_val_test(
    adata: ad.AnnData,
    *,
    split_key: str,
    split_dict: Dict[str, List[str]] | None = None,
    train_conds: List[str] | None = None,
    val_conds: List[str] | None = None,
    test_conds: List[str] | None = None,
    split_ratio: List[float] | tuple[float, float, float] | None = None,
    split_strategy: str = "standard",
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    test_ratio: float | None = None,
    transform: Callable[[str], str] | None = None,
    control_mask: Any | None = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Create condition-level train, validation, and test AnnData partitions.

    CAPRA splits perturbation conditions, not individual cells. Control cells
    are included in every returned AnnData view because both training and
    prediction require a control background. Splits can be supplied explicitly
    or sampled from observed non-control conditions using a ratio triplet.

    Parameters
    ----------
    adata:
        Source single-cell matrix.
    split_key:
        Observation column containing condition labels.
    split_dict / train_conds / val_conds / test_conds:
        Explicit condition partitions. `train` may be omitted when validation
        and test conditions are supplied; remaining observed perturbations are
        used for training.
    split_ratio / train_ratio / val_ratio / test_ratio:
        Ratio-based condition split specification used only when explicit
        partitions are absent.
    split_strategy:
        `standard` samples condition labels directly. `auto` detects whether
        the condition list contains double-gene perturbations; single-gene-only
        lists use the ratio split, while combination lists additionally report
        combo_seen0/1/2 subgroups.
    transform:
        Optional label canonicalization function.
    control_mask:
        Optional Boolean mask identifying control cells.
    seed:
        Random seed used only for ratio-based splits.

    Returns
    -------
    dict
        Normalized condition names plus AnnData slices for train, validation,
        test, and the train+validation reference set.
    """
    transform_fn = transform or (lambda x: str(x))
    split_values = adata.obs[split_key].astype(str).map(transform_fn)
    split_values_np = split_values.to_numpy(dtype=object, copy=False)

    if control_mask is None:
        control_mask_np = split_values.isin(["control", "ctrl"]).to_numpy()
    else:
        control_mask_np = np.asarray(control_mask, dtype=bool)
        if control_mask_np.shape[0] != adata.n_obs:
            raise ValueError("control_mask must have the same length as adata.n_obs")

    strategy = str(split_strategy).lower().replace("-", "_")
    allowed_strategies = {"standard", "ratio", "auto", "simulation", "seen_gene", "seen_gene_auto"}
    if strategy not in allowed_strategies:
        raise ValueError(f"unsupported split_strategy={split_strategy!r}")
    use_auto_strategy = strategy in {"auto", "simulation", "seen_gene", "seen_gene_auto"}
    explicit_split_requested = (
        split_dict is not None or train_conds is not None or val_conds is not None or test_conds is not None
    )
    if use_auto_strategy and not explicit_split_requested and split_ratio is None and all(
        value is None for value in (train_ratio, val_ratio, test_ratio)
    ):
        split_ratio = (0.8, 0.1, 0.1)

    use_ratio_mode = split_dict is None and split_ratio is not None
    use_ratio_triplet_mode = split_dict is None and split_ratio is None and any(
        value is not None for value in (train_ratio, val_ratio, test_ratio)
    )

    subgroup = None
    if use_ratio_mode or use_ratio_triplet_mode:
        if split_dict is not None or train_conds is not None or val_conds is not None or test_conds is not None:
            raise ValueError("ratio mode cannot be combined with explicit split lists")
        if split_ratio is not None:
            ratios = _normalize_split_ratio(split_ratio)
        else:
            if train_ratio is None or val_ratio is None or test_ratio is None:
                raise ValueError("train_ratio/val_ratio/test_ratio must all be provided in ratio mode")
            ratios = _normalize_split_ratio([train_ratio, val_ratio, test_ratio])

        observed = sorted(set(split_values[~control_mask_np].tolist()))
        if use_auto_strategy:
            split_payload = _build_seen_gene_simulation_split(observed, seed=int(seed), split_ratio=ratios)
            split_dict = split_payload["splits"]
            subgroup = split_payload["subgroup"]
        else:
            split_dict = _ratio_split_conditions(observed, split_ratio=ratios, seed=int(seed))
    else:
        if split_dict is None:
            if val_conds is None or test_conds is None:
                raise ValueError("either explicit split lists or split ratios must be provided")
            split_dict = {"train": train_conds, "val": val_conds, "test": test_conds}

    val_set = {transform_fn(item) for item in split_dict.get("val", [])}
    test_set = {transform_fn(item) for item in split_dict.get("test", [])}
    train_set = split_dict.get("train")
    if train_set is None:
        observed = set(split_values[~control_mask_np].tolist())
        train_set = sorted(observed - val_set - test_set)
    else:
        train_set = [transform_fn(item) for item in train_set]

    normalized_splits = {
        "train": sorted(set(train_set)),
        "val": sorted(val_set),
        "test": sorted(test_set),
    }
    non_control_splits = {
        key: {value for value in values if value not in {"control", "ctrl"}}
        for key, values in normalized_splits.items()
    }
    overlaps = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(non_control_splits[left] & non_control_splits[right])
        if overlap:
            overlaps.append(f"{left}/{right}: {overlap}")
    if overlaps:
        raise ValueError("split partitions overlap after canonicalization; " + "; ".join(overlaps))

    observed_non_control = {
        value
        for value in split_values_np.tolist()
        if value not in {"control", "ctrl"}
    }
    missing_requested = []
    for split_name, values in normalized_splits.items():
        for value in values:
            if value in {"control", "ctrl"}:
                continue
            if value not in observed_non_control:
                missing_requested.append(f"{split_name}:{value}")
    if missing_requested:
        raise ValueError(
            "split partitions contain conditions not observed in adata after canonicalization; "
            + "; ".join(missing_requested)
        )

    train_mask = control_mask_np | np.isin(split_values_np, normalized_splits["train"])
    val_mask = control_mask_np | np.isin(split_values_np, normalized_splits["val"])
    test_mask = control_mask_np | np.isin(split_values_np, normalized_splits["test"])
    train_val_mask = control_mask_np | np.isin(
        split_values_np,
        normalized_splits["train"] + normalized_splits["val"],
    )

    payload = {
        "splits": normalized_splits,
        "train_adata": adata[train_mask],
        "val_adata": adata[val_mask],
        "test_adata": adata[test_mask],
        "train_val_adata": adata[train_val_mask],
    }
    if subgroup is not None:
        payload["subgroup"] = subgroup
    return payload


def rowwise_masked_pearson(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute a per-row Pearson correlation over masked gene dimensions.

    Rows with degenerate variance are converted to zero correlation instead of
    propagating NaNs. The caller usually converts this correlation to Pearson
    distance via `1 - corr`.
    """
    pred = pred.float()
    target = target.float()
    mask = mask.float()
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    pred_mean = (pred * mask).sum(dim=1, keepdim=True) / denom
    target_mean = (target * mask).sum(dim=1, keepdim=True) / denom
    pred_centered = (pred - pred_mean) * mask
    target_centered = (target - target_mean) * mask
    numerator = (pred_centered * target_centered).sum(dim=1)
    pred_norm = torch.sqrt((pred_centered**2).sum(dim=1).clamp_min(eps))
    target_norm = torch.sqrt((target_centered**2).sum(dim=1).clamp_min(eps))
    corr = numerator / (pred_norm * target_norm + eps)
    return torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

# ── Variance calibration (internal) ─────────────────────────────────────────

def _rng_fork_devices() -> list[int]:
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


# Internal sparse-output guard (fixed strategy, not a public parameter).
# Internal sparse-output guard — fixed parameters, not exposed as public config.
# Attenuates residual magnitude for genes with low control expression,
# reducing noise in pseudobulk means without affecting DEG-signal genes.
_SPARSE_RESIDUAL_MODE = "anchor_rescue_softmask"
_SPARSE_RESIDUAL_EPS = 0.0005          # softshrink threshold on residual
_SPARSE_CONTROL_THRESHOLD = 0.05       # detectability sigmoid midpoint
_SPARSE_CONTROL_TEMPERATURE = 0.02     # detectability sigmoid steepness
_SPARSE_CONTROL_USE_POSITIVE_QUANTILE = True
_SPARSE_CONTROL_POSITIVE_QUANTILE = 0.25  # use q25 of positive ctrl genes as floor
_SPARSE_CONTROL_MASK_FLOOR = 0.25      # min mask value (25% residual preserved)
_SPARSE_ANCHOR_RESCUE_THRESHOLD = 0.05  # |anchor_delta| above which rescue activates
_SPARSE_ANCHOR_RESCUE_TEMPERATURE = 0.05


def _micro_softshrink(residual):
    """Soft-threshold tiny residuals toward zero: sign(x) * relu(|x| - eps)."""
    eps = float(_SPARSE_RESIDUAL_EPS)
    if eps <= 0.0:
        return residual
    return residual.sign() * torch.relu(residual.abs() - eps)


def _control_detectability_mask(control_mean, anchor_delta):
    threshold = torch.full(
        (control_mean.shape[0], 1), float(_SPARSE_CONTROL_THRESHOLD),
        device=control_mean.device, dtype=control_mean.dtype,
    )
    if _SPARSE_CONTROL_USE_POSITIVE_QUANTILE:
        positive = control_mean > 0.0
        positive_count = positive.sum(dim=1)
        sorted_values = torch.sort(
            torch.where(positive, control_mean, torch.full_like(control_mean, float("inf"))), dim=1
        ).values
        q = float(_SPARSE_CONTROL_POSITIVE_QUANTILE)
        q_index = torch.floor((positive_count.float() - 1.0).clamp_min(0.0) * q).long().unsqueeze(1)
        qt = sorted_values.gather(1, q_index)
        threshold = torch.where(positive_count.unsqueeze(1) > 0, torch.minimum(qt, threshold), threshold)
    detectability = torch.sigmoid((control_mean - threshold) / float(_SPARSE_CONTROL_TEMPERATURE))
    if float(_SPARSE_ANCHOR_RESCUE_THRESHOLD) > 0:
        rescue = torch.sigmoid(
            (anchor_delta.abs() - float(_SPARSE_ANCHOR_RESCUE_THRESHOLD))
            / float(_SPARSE_ANCHOR_RESCUE_TEMPERATURE)
        )
        detectability = torch.maximum(detectability, rescue)
    return float(_SPARSE_CONTROL_MASK_FLOOR) + (1.0 - float(_SPARSE_CONTROL_MASK_FLOOR)) * detectability


def apply_sparse_residual_guard(*, additive_residual, pair_residual, control_mean, anchor_delta, is_training):
    """Apply control-detectability soft mask to residuals (eval only)."""
    mask = _control_detectability_mask(control_mean, anchor_delta)
    additive_residual = additive_residual * mask
    pair_residual = pair_residual * mask
    if not is_training:
        additive_residual = _micro_softshrink(additive_residual)
        pair_residual = _micro_softshrink(pair_residual)
    return additive_residual, pair_residual


class TorchInitializationContext:
    """Local, RNG-isolated initializer for CAPRA module construction."""

    def __init__(self, seed: int) -> None:
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))

    def linear(self, in_features: int, out_features: int) -> nn.Linear:
        layer = nn.Linear(in_features, out_features, device=torch.device("cpu"))
        with torch.no_grad():
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), generator=self.generator)
            bound = 1 / math.sqrt(in_features) if in_features > 0 else 0
            nn.init.uniform_(layer.bias, -bound, bound, generator=self.generator)
        return layer

    @staticmethod
    def linear_without_global_rng(in_features: int, out_features: int) -> nn.Linear:
        with torch.random.fork_rng(devices=_rng_fork_devices()):
            return nn.Linear(in_features, out_features, device=torch.device("cpu"))

    @staticmethod
    def zero_init_final_linear(module: nn.Sequential) -> None:
        for layer in reversed(module):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                return


def _masked_row_median(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Median of each row considering only masked elements."""
    x_masked = x.float().clone()
    x_masked[~mask] = float("inf")
    sorted_x = torch.sort(x_masked, dim=1).values
    valid_counts = mask.sum(dim=1).clamp_min(1)
    mid_idx = ((valid_counts - 1) // 2).long().unsqueeze(1)
    return sorted_x.gather(1, mid_idx).squeeze(1)


def variance_calibration_noise_scale(
    control_mean: torch.Tensor,
    control_std: torch.Tensor,
    *,
    expressed_threshold: float,
    se_floor_mult: float,
    noise_scale: float,
) -> torch.Tensor:
    """Infer extra predictive standard deviation from control-only statistics."""
    control_mean_f32 = control_mean.float()
    control_std_f32 = control_std.float().clamp_min(0.0)
    expressed = control_mean_f32 > float(expressed_threshold)
    floor = _masked_row_median(control_std_f32, expressed).unsqueeze(1)
    floor = floor * float(se_floor_mult)
    target_std = torch.maximum(control_std_f32, floor)
    extra_var = (target_std.square() - control_std_f32.square()).clamp_min(0.0)
    extra_std = torch.sqrt(extra_var) * float(noise_scale)
    return extra_std.to(dtype=control_std.dtype)


def deterministic_standard_normal_like(
    template: torch.Tensor,
    per_row_seeds: torch.Tensor,
    salt: int,
) -> torch.Tensor:
    """Produce deterministic standard-normal noise keyed by (row_seed, col, salt)."""
    gen = torch.Generator(device=template.device)
    noise = torch.empty_like(template)
    n_rows = template.shape[0]
    for row in range(n_rows):
        gen.manual_seed(int(per_row_seeds[row].item()) * 31337 + salt)
        noise[row].normal_(generator=gen)
    return noise
