"""High-level CAPRA public API.

This module owns user-facing data preparation, training orchestration, model
loading, prediction, and plotting helpers. Low-level neural-network components
live under `models/`, and native PyTorch training internals live in `training.py`.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse

from data_utils import build_prediction_batch, load_condition_dataset_from_adata
from training import (
    CAPRATrainingModule,
    DEFAULT_CONTROL_CONTEXT_MODE,
    build_capra_config,
    fit_capra,
    load_capra_checkpoint_config,
    project_seed,
    resolve_capra_device,
)
from utils import canonical_condition_name, set_seed, split_anndata_train_val_test


DEFAULT_CONTROL_LABEL = "control"


def load_gene_embedding_table(
    path: str | Path,
    *,
    genes: Iterable[str] | None = None,
    add_control: bool = True,
    control_label: str = "ctrl",
) -> pd.DataFrame:
    """Load a gene embedding table from pickle with genes as the index."""
    with Path(path).expanduser().resolve().open("rb") as handle:
        raw = pickle.load(handle)

    table = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw).T
    table.index = table.index.astype(str)
    table = table.astype(np.float32)

    if genes is not None:
        requested = [str(gene) for gene in genes]
        table = table.loc[[gene for gene in requested if gene in table.index]].copy()

    if add_control and control_label not in table.index:
        control = pd.DataFrame(
            [np.zeros(table.shape[1], dtype=np.float32)],
            columns=table.columns,
            index=[control_label],
        )
        table = pd.concat([control, table])
    return table


def load_single_cell_adata(path: str | Path, *, backed: str | None = None) -> ad.AnnData:
    """Load an AnnData object without applying any external file-layout convention."""
    return ad.read_h5ad(Path(path).expanduser().resolve(), backed=backed)


def _matrix_mean(matrix: Any) -> np.ndarray:
    """Compute a float32 column mean for dense or sparse expression matrices."""
    if sparse.issparse(matrix):
        return np.asarray(matrix.mean(axis=0)).reshape(-1).astype(np.float32, copy=False)
    return np.asarray(matrix, dtype=np.float32).mean(axis=0).astype(np.float32, copy=False)


def _as_condition_set(conditions: Iterable[str]) -> set[str]:
    """Canonicalize non-control condition labels into a set."""
    return {canonical_condition_name(condition) for condition in conditions if canonical_condition_name(condition) != "control"}


def load_differential_expression_table(
    path: str | Path,
    *,
    allowed_conditions: Iterable[str] | None = None,
    control_label: str = DEFAULT_CONTROL_LABEL,
) -> Dict[str, pd.DataFrame]:
    """Load a pickled DEG table and optionally keep only selected conditions."""
    with Path(path).expanduser().resolve().open("rb") as handle:
        raw = pickle.load(handle)

    control = canonical_condition_name(control_label)
    allowed = None
    if allowed_conditions is not None:
        allowed = {
            canonical_condition_name(condition)
            for condition in allowed_conditions
            if canonical_condition_name(condition) != control
        }

    deg_tables: Dict[str, pd.DataFrame] = {}
    for condition, frame in raw.items():
        canonical = canonical_condition_name(condition)
        if canonical == control:
            continue
        if allowed is not None and canonical not in allowed:
            continue
        deg_tables[canonical] = frame.copy() if hasattr(frame, "copy") else frame

    if allowed is not None:
        missing = sorted(allowed - set(deg_tables))
        if missing:
            preview = ", ".join(missing[:5])
            suffix = "" if len(missing) <= 5 else f", ... (+{len(missing) - 5} more)"
            warnings.warn(
                f"Missing DEG entries for {len(missing)} conditions: {preview}{suffix}",
                RuntimeWarning,
            )
    return deg_tables


def _mean_shift_differential_expression_table(
    adata: ad.AnnData,
    *,
    conditions: Iterable[str],
    condition_key: str = "condition",
    perturbation_key: str = "perturbation",
    control_label: str = DEFAULT_CONTROL_LABEL,
) -> Dict[str, pd.DataFrame]:
    """Build DEG-like rankings from absolute mean shifts relative to control.

    This fallback is deterministic and does not invoke Scanpy tests. It is
    useful for fast checks or cases where a statistical test is unavailable,
    but full evaluation runs should normally use the configured rank-genes method.
    """
    condition_values = adata.obs[condition_key].astype(str).map(canonical_condition_name).to_numpy()
    if perturbation_key in adata.obs:
        perturbation_values = adata.obs[perturbation_key].astype(str).map(canonical_condition_name).to_numpy()
        control_mask = perturbation_values == canonical_condition_name(control_label)
    else:
        control_mask = condition_values == canonical_condition_name(control_label)

    if not np.any(control_mask):
        raise ValueError("cannot build DEG table: no control cells were found")

    control_mean = _matrix_mean(adata.X[control_mask])
    var_names = np.asarray(adata.var_names).astype(str)
    deg_tables: Dict[str, pd.DataFrame] = {}

    for condition in sorted(_as_condition_set(conditions)):
        condition_mask = condition_values == condition
        if not np.any(condition_mask):
            continue
        delta = _matrix_mean(adata.X[condition_mask]) - control_mean
        deg_tables[condition] = pd.DataFrame(
            {
                "scores": delta.astype(np.float32, copy=False),
                "abs_scores": np.abs(delta).astype(np.float32, copy=False),
            },
            index=var_names,
        ).sort_values("abs_scores", ascending=False)
    return deg_tables


def compute_differential_expression_table(
    adata: ad.AnnData,
    *,
    conditions: Iterable[str],
    condition_key: str = "condition",
    perturbation_key: str = "perturbation",
    control_label: str = DEFAULT_CONTROL_LABEL,
    method: str = "t-test",
) -> Dict[str, pd.DataFrame]:
    """Compute per-condition DEG tables from AnnData.

    The Scanpy branch follows the standard `rank_genes_groups` workflow:
    `sc.tl.rank_genes_groups(..., reference=control, method='t-test')`, then
    stores `names`, adjusted p-values, fold changes, scores, and `abs_scores`.
    """
    if method == "mean_shift":
        return _mean_shift_differential_expression_table(
            adata,
            conditions=conditions,
            condition_key=condition_key,
            perturbation_key=perturbation_key,
            control_label=control_label,
        )

    if method not in {"t-test", "wilcoxon", "logreg", "t-test_overestim_var"}:
        raise ValueError("method must be one of: 't-test', 't-test_overestim_var', 'wilcoxon', 'logreg', 'mean_shift'")

    work = adata.copy()
    condition_values = work.obs[condition_key].astype(str).map(canonical_condition_name)
    work.obs[condition_key] = condition_values.astype("category")
    control = canonical_condition_name(control_label)
    observed = set(condition_values.astype(str).tolist())
    groups = sorted(condition for condition in _as_condition_set(conditions) if condition in observed)
    groups = [condition for condition in groups if condition != control]
    if not groups:
        return {}
    if control not in observed:
        raise ValueError(f"control_label={control_label!r} was not found in {condition_key!r}")

    work.uns["log1p"] = {}
    work.uns["log1p"]["base"] = None
    sc.tl.rank_genes_groups(work, condition_key, groups=groups, reference=control, method=method)
    result = work.uns["rank_genes_groups"]

    deg_tables: Dict[str, pd.DataFrame] = {}
    for condition in groups:
        frame_payload = {}
        for key in ("names", "pvals_adj", "logfoldchanges", "scores"):
            if key in result:
                frame_payload[key] = result[key][condition]
        frame = pd.DataFrame(frame_payload)
        if "logfoldchanges" in frame:
            frame["foldchanges"] = 2 ** frame["logfoldchanges"]
            frame.drop(labels=["logfoldchanges"], inplace=True, axis=1)
        frame.set_index("names", inplace=True)
        if "scores" in frame:
            frame["abs_scores"] = np.abs(frame["scores"])
            frame.sort_values("abs_scores", ascending=False, inplace=True)
        deg_tables[condition] = frame
    return deg_tables


def _compute_ranked_differential_expression_reference(
    adata: ad.AnnData,
    *,
    conditions: Iterable[str] | None = None,
    condition_column: str = "perturbation",
    control_tag: str = DEFAULT_CONTROL_LABEL,
    method: str = "t-test",
) -> Dict[str, pd.DataFrame]:
    """Compute a full ranked DEG reference, then optionally filter conditions."""
    if condition_column not in adata.obs:
        raise KeyError(f"condition_column={condition_column!r} is not present in adata.obs")

    condition_values = adata.obs[condition_column].astype(str).map(canonical_condition_name)
    control = canonical_condition_name(control_tag)
    observed = condition_values.astype(str).tolist()
    observed_set = set(observed)
    if control not in observed_set:
        raise ValueError(f"control_tag={control_tag!r} was not found in {condition_column!r}")

    allowed_conditions = _as_condition_set(conditions) if conditions is not None else None
    observed_groups: List[str] = []
    seen: set[str] = set()
    for condition in observed:
        if condition == control or condition in seen:
            continue
        if allowed_conditions is not None and condition not in allowed_conditions:
            continue
        seen.add(condition)
        observed_groups.append(condition)
    if not observed_groups:
        return {}

    if method == "mean_shift":
        deg_tables = _mean_shift_differential_expression_table(
            adata,
            conditions=observed_groups,
            condition_key=condition_column,
            perturbation_key=condition_column,
            control_label=control_tag,
        )
    else:
        if method not in {"t-test", "wilcoxon", "logreg", "t-test_overestim_var"}:
            raise ValueError("method must be one of: 't-test', 't-test_overestim_var', 'wilcoxon', 'logreg', 'mean_shift'")

        work = adata.copy()
        work.obs[condition_column] = condition_values.astype("category")
        work.uns["log1p"] = {}
        work.uns["log1p"]["base"] = None
        sc.tl.rank_genes_groups(work, condition_column, groups=observed_groups, reference=control, method=method)
        result = work.uns["rank_genes_groups"]

        deg_tables = {}
        for condition in observed_groups:
            frame_payload = {}
            for key in ("names", "pvals_adj", "logfoldchanges", "scores"):
                if key in result:
                    frame_payload[key] = result[key][condition]
            frame = pd.DataFrame(frame_payload)
            if "logfoldchanges" in frame:
                frame["foldchanges"] = 2 ** frame["logfoldchanges"]
                frame.drop(labels=["logfoldchanges"], inplace=True, axis=1)
            frame.set_index("names", inplace=True)
            if "scores" in frame:
                frame["abs_scores"] = np.abs(frame["scores"])
                frame.sort_values("abs_scores", ascending=False, inplace=True)
            deg_tables[condition] = frame

    if conditions is None:
        return deg_tables

    allowed_conditions = _as_condition_set(conditions)
    return {
        condition: frame
        for condition, frame in deg_tables.items()
        if condition in allowed_conditions
    }


class CAPRAData:
    """Data container for CAPRA independent of any external file layout."""

    def __init__(
        self,
        adata: ad.AnnData | None = None,
        embedding_table: pd.DataFrame | None = None,
        *,
        study_name: str = "capra_study",
        condition_key: str = "condition",
        perturbation_key: str = "perturbation",
        var_gene_key: str | None = None,
        control_label: str = DEFAULT_CONTROL_LABEL,
    ) -> None:
        """Initialize a data-preparation state object.

        Parameters
        ----------
        adata:
            Optional AnnData object containing expression and observation
            metadata.
        embedding_table:
            Optional gene embedding table indexed by perturbation gene.
        condition_key / perturbation_key:
            Observation columns used to derive canonical CAPRA condition labels.
        var_gene_key:
            Optional variable column used to replace `adata.var_names`.
        control_label:
            Label identifying control cells before canonicalization.
        """
        self.study_name = str(study_name)
        self.condition_key = str(condition_key)
        self.perturbation_key = str(perturbation_key)
        self.var_gene_key = var_gene_key
        self.control_label = str(control_label)
        self.adata = adata
        self.embedding_table = embedding_table
        self.splits: Dict[str, List[str]] | None = None
        self.split_payload: Dict[str, Any] | None = None
        self.subgroup: Dict[str, Any] | None = None
        self.deg_frames_by_condition: Dict[str, pd.DataFrame] | None = None
        self.assets = None

    def load_single_cell_adata(self, path: str | Path, *, backed: str | None = None) -> "CAPRAData":
        """Load AnnData into this object and return `self` for chaining."""
        self.adata = load_single_cell_adata(path, backed=backed)
        return self

    def load_gene_embedding_table(
        self,
        path: str | Path,
        *,
        genes: Iterable[str] | None = None,
        add_control: bool = True,
        control_label: str = "ctrl",
    ) -> "CAPRAData":
        """Load and attach a gene embedding table for this CAPRAData object."""
        self.embedding_table = load_gene_embedding_table(
            path,
            genes=genes,
            add_control=add_control,
            control_label=control_label,
        )
        return self

    def harmonize_perturbation_metadata(self, *, copy: bool = True) -> "CAPRAData":
        """Create CAPRA's canonical condition and perturbation metadata."""
        if self.adata is None:
            raise ValueError("adata is not set")

        adata = self.adata.copy() if copy else self.adata
        if self.var_gene_key is not None:
            if self.var_gene_key not in adata.var:
                raise KeyError(f"var_gene_key={self.var_gene_key!r} is not present in adata.var")
            adata.var_names = adata.var[self.var_gene_key].astype(str).to_numpy()

        if self.condition_key not in adata.obs:
            raise KeyError(f"condition_key={self.condition_key!r} is not present in adata.obs")
        adata.obs["condition"] = adata.obs[self.condition_key].astype(str).map(canonical_condition_name)

        if self.perturbation_key in adata.obs:
            adata.obs["perturbation"] = adata.obs[self.perturbation_key].astype(str).map(canonical_condition_name)
        else:
            adata.obs["perturbation"] = adata.obs["condition"]

        self.adata = adata
        return self

    def register_evaluation_partitions(
        self,
        *,
        split_dict: Dict[str, List[str]] | None = None,
        train_conds: List[str] | None = None,
        val_conds: List[str] | None = None,
        test_conds: List[str] | None = None,
        split_ratio: tuple[float, float, float] | List[float] | None = None,
        split_strategy: str = "standard",
        random_state: int = 0,
    ) -> "CAPRAData":
        """Register condition-level partitions used by CAPRA.

        Explicit evaluation partitions are deterministic inputs to CAPRA. The
        random_state is only used when CAPRA itself creates a generated split.
        Set split_strategy="auto" to detect single-gene-only versus
        double-gene condition lists and expose combo_seen0/1/2 subgroups.
        """
        if self.adata is None:
            raise ValueError("adata is not set")
        if "condition" not in self.adata.obs or "perturbation" not in self.adata.obs:
            self.harmonize_perturbation_metadata(copy=False)

        control_mask = self.adata.obs["perturbation"].astype(str).map(canonical_condition_name) == "control"
        self.split_payload = split_anndata_train_val_test(
            self.adata,
            split_key="condition",
            split_dict=split_dict,
            train_conds=train_conds,
            val_conds=val_conds,
            test_conds=test_conds,
            split_ratio=split_ratio,
            split_strategy=split_strategy,
            transform=canonical_condition_name,
            control_mask=control_mask,
            seed=int(random_state),
        )
        self.splits = self.split_payload["splits"]
        self.subgroup = self.split_payload.get("subgroup")
        return self

    def set_differential_expression_table(self, deg_frames_by_condition: Dict[str, pd.DataFrame]) -> "CAPRAData":
        """Attach precomputed DEG tables keyed by canonical condition name."""
        self.deg_frames_by_condition = {
            canonical_condition_name(condition): frame
            for condition, frame in deg_frames_by_condition.items()
            if canonical_condition_name(condition) != "control"
        }
        return self

    def load_differential_expression_table(
        self,
        path: str | Path,
        *,
        allowed_conditions: Iterable[str] | None = None,
    ) -> "CAPRAData":
        """Load precomputed DEG tables and store them on this data object."""
        if allowed_conditions is None and self.splits is not None:
            allowed_conditions = self.splits["train"] + self.splits["val"]
        self.deg_frames_by_condition = load_differential_expression_table(
            path,
            allowed_conditions=allowed_conditions,
            control_label=self.control_label,
        )
        return self

    def compute_differential_expression_table(
        self,
        *,
        conditions: Iterable[str] | None = None,
        method: str = "t-test",
    ) -> "CAPRAData":
        """Compute train/validation DEG tables from the current split payload."""
        if self.adata is None:
            raise ValueError("adata is not set")
        if self.split_payload is None or self.splits is None:
            self.register_evaluation_partitions(split_ratio=(0.8, 0.1, 0.1), random_state=0)
        if conditions is None:
            conditions = self.splits["train"] + self.splits["val"]
        self.deg_frames_by_condition = compute_differential_expression_table(
            self.split_payload["train_val_adata"],
            conditions=conditions,
            condition_key="condition",
            perturbation_key="perturbation",
            control_label=self.control_label,
            method=method,
        )
        return self

    def estimate_trainval_deg_reference(
        self,
        *,
        source_adata: ad.AnnData | None = None,
        conditions: Iterable[str] | None = None,
        condition_column: str = "perturbation",
        method: str = "t-test",
        allow_source_test_conditions: bool = False,
    ) -> "CAPRAData":
        """Estimate CAPRA's train/validation DEG reference table."""
        if self.adata is None and source_adata is None:
            raise ValueError("adata is not set")
        if self.split_payload is None or self.splits is None:
            self.register_evaluation_partitions(split_ratio=(0.8, 0.1, 0.1), random_state=0)

        target_adata = source_adata if source_adata is not None else self.split_payload["train_val_adata"]
        if source_adata is None:
            if condition_column not in target_adata.obs and (
                "condition" not in target_adata.obs or "perturbation" not in target_adata.obs
            ):
                self.harmonize_perturbation_metadata(copy=False)
                target_adata = self.split_payload["train_val_adata"]
        elif condition_column not in target_adata.obs and (
            "condition" not in target_adata.obs or "perturbation" not in target_adata.obs
        ):
            target_adata = target_adata.copy()
            target_adata.obs["condition"] = target_adata.obs[self.condition_key].astype(str).map(canonical_condition_name)
            if self.perturbation_key in target_adata.obs:
                target_adata.obs["perturbation"] = (
                    target_adata.obs[self.perturbation_key].astype(str).map(canonical_condition_name)
                )
            else:
                target_adata.obs["perturbation"] = target_adata.obs["condition"]
        if source_adata is not None and not allow_source_test_conditions and self.splits is not None:
            if condition_column not in target_adata.obs:
                raise KeyError(f"condition_column={condition_column!r} is not present in source_adata.obs")
            observed_conditions = set(target_adata.obs[condition_column].astype(str).map(canonical_condition_name))
            test_conditions = {
                canonical_condition_name(condition)
                for condition in self.splits.get("test", [])
                if canonical_condition_name(condition) != "control"
            }
            leaked_conditions = sorted(observed_conditions & test_conditions)
            if leaked_conditions:
                preview = ", ".join(leaked_conditions[:5])
                suffix = "" if len(leaked_conditions) <= 5 else f", ... (+{len(leaked_conditions) - 5} more)"
                raise ValueError(
                    "source_adata contains test split conditions while estimating train/validation DEG reference: "
                    f"{preview}{suffix}. Pass allow_source_test_conditions=True only for explicit non-benchmark audits."
                )
        if conditions is None and self.splits is not None:
            conditions = self.splits["train"] + self.splits["val"]
        self.deg_frames_by_condition = _compute_ranked_differential_expression_reference(
            target_adata,
            conditions=conditions,
            condition_column=condition_column,
            control_tag=self.control_label,
            method=method,
        )
        return self

    def build_control_relative_training_state(
        self,
        *,
        study_name: str | None = None,
        topk_deg: int = 100,
        knn_topk: int = 5,
        knn_temperature: float = 12.0,
        deg_frames_by_condition: Dict[str, pd.DataFrame] | None = None,
        deg_pickle_path: str | Path | None = None,
        deg_method: str = "t-test",
    ) -> "CAPRAData":
        """Assemble CAPRA's control-relative training state."""
        if self.adata is None:
            raise ValueError("adata is not set")
        if self.embedding_table is None:
            raise ValueError("embedding_table is not set")
        if self.split_payload is None or self.splits is None:
            self.register_evaluation_partitions(split_ratio=(0.8, 0.1, 0.1), random_state=0)

        if study_name is not None:
            self.study_name = str(study_name)

        train_val_conditions = self.splits["train"] + self.splits["val"]
        if deg_pickle_path is not None:
            self.load_differential_expression_table(deg_pickle_path, allowed_conditions=train_val_conditions)
        elif deg_frames_by_condition is not None:
            self.set_differential_expression_table(deg_frames_by_condition)

        deg_frames = self.deg_frames_by_condition
        if deg_frames is None:
            self.estimate_trainval_deg_reference(method=deg_method)
            deg_frames = self.deg_frames_by_condition

        self.assets = load_condition_dataset_from_adata(
            study_name=self.study_name,
            adata=self.split_payload["train_val_adata"],
            splits=self.splits,
            embedding_table=self.embedding_table,
            topk_deg=int(topk_deg),
            knn_topk=int(knn_topk),
            knn_temperature=float(knn_temperature),
            deg_frames_by_condition=deg_frames,
        )
        return self


def build_capra_data(*args, **kwargs) -> CAPRAData:
    """Construct a `CAPRAData` object and immediately harmonize metadata."""
    data = CAPRAData(*args, **kwargs)
    data.harmonize_perturbation_metadata()
    return data


def _apply_overrides(base: Dict[str, Any], overrides: Dict[str, Any] | None) -> Dict[str, Any]:
    """Recursively merge user overrides into a nested CAPRA config."""
    if not overrides:
        return base
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _apply_overrides(base[key], value)
        else:
            base[key] = value
    return base


def _enable_torch_runtime(*, allow_tf32: bool) -> None:
    """Enable PyTorch runtime options used by CAPRA training and inference."""
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)


def _materialize_prediction_slice(
    batch: Dict[str, torch.Tensor],
    *,
    start: int,
    end: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Move one slice of a prediction batch to the target device.

    Condition-level tensors with a leading singleton dimension are expanded to
    match the requested row count, while sampled control tensors are sliced.
    """
    row_count = int(end - start)
    materialized: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if key in {"control_mean", "control_std", "prediction_noise_seed"}:
            materialized[key] = value[start:end].to(device, non_blocking=True)
        elif value.shape[0] == 1:
            materialized[key] = value.to(device, non_blocking=True).expand((row_count,) + tuple(value.shape[1:]))
        elif start == 0 and value.shape[0] == row_count:
            materialized[key] = value.to(device, non_blocking=True)
        else:
            materialized[key] = value[start:end].to(device, non_blocking=True)
    return materialized


def _asset_build_kwargs_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract asset-construction knobs from a CAPRA config."""
    data_cfg = cfg.get("data", {})
    return {
        "topk_deg": int(data_cfg.get("topk_deg", 100)),
        "knn_topk": int(data_cfg.get("knn_topk", 5)),
        "knn_temperature": float(data_cfg.get("knn_temperature", 12.0)),
    }


def _validate_assets_match_cfg(assets: Any, cfg: Dict[str, Any], *, context: str) -> None:
    """Fail fast when prebuilt assets do not match a run's asset-building config."""
    expected = _asset_build_kwargs_from_cfg(cfg)
    actual = getattr(assets, "asset_config", None)
    if not isinstance(actual, dict):
        raise ValueError(
            f"{context}: prebuilt CAPRA assets do not record asset_config; "
            "rebuild the assets with build_control_relative_training_state()"
        )

    mismatches = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, float):
            matched = actual_value is not None and np.isclose(
                float(actual_value),
                expected_value,
                rtol=0.0,
                atol=1e-12,
            )
        else:
            matched = actual_value is not None and int(actual_value) == int(expected_value)
        if not matched:
            mismatches.append(f"{key}: assets={actual_value!r}, config={expected_value!r}")
    if mismatches:
        raise ValueError(
            f"{context}: prebuilt CAPRA assets do not match the requested config; "
            + "; ".join(mismatches)
            + ". Rebuild assets or clear CAPRAData.assets before fitting/loading."
        )


def _current_model_device(module: torch.nn.Module) -> torch.device:
    """Return the device that currently holds model parameters."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


class CAPRA:
    """Train and apply the CAPRA perturbation model."""

    def __init__(self, data: CAPRAData | None = None) -> None:
        """Create a CAPRA model wrapper around optional prepared data."""
        self.data = data
        self.fitted: Dict[str, Any] | None = None
        self.predictions: Dict[str, np.ndarray] | None = None

    def fit_capra_response_operator(
        self,
        data: CAPRAData | None = None,
        *,
        n_epochs: int = 80,
        min_epochs: int = 10,
        patience: int = 10,
        batch_size: int = 192,
        learning_rate: float = 6e-4,
        seed: int | None = None,
        model_init_seed: int | None = None,
        control_context_seed: int | None = None,
        output_dir: str | Path | None = None,
        run_name: str | None = None,
        fast_dev_run: bool = False,
        config_overrides: Dict[str, Any] | None = None,
        **kwargs,
    ) -> "CAPRA":
        """Fit CAPRA's interpretable response operator."""
        if data is not None:
            self.data = data
        if self.data is None:
            raise ValueError("fit_capra_response_operator requires a CAPRAData object")

        results_root = Path(output_dir).expanduser().resolve() if output_dir is not None else None
        cfg = build_capra_config(
            study_name=self.data.study_name,
            seed=seed,
            model_init_seed=model_init_seed,
            control_context_seed=control_context_seed,
            results_root=results_root,
            output_name=run_name or "capra_response_operator",
            fast_dev_run=bool(fast_dev_run),
            train_batch_size=int(batch_size),
            min_epochs=int(min_epochs),
            max_epochs=int(n_epochs),
            patience=int(patience),
            learning_rate=float(learning_rate),
            **kwargs,
        )
        _apply_overrides(cfg, config_overrides)
        if self.data.assets is None:
            self.data.build_control_relative_training_state(**_asset_build_kwargs_from_cfg(cfg))
        else:
            _validate_assets_match_cfg(self.data.assets, cfg, context="fit_capra_response_operator")
        set_seed(project_seed(cfg))
        _enable_torch_runtime(allow_tf32=bool(cfg["train"].get("allow_tf32", True)))
        self.fitted = fit_capra(cfg, assets=self.data.assets)
        return self

    def load_capra_response_operator(
        self,
        checkpoint_path: str | Path,
        data: CAPRAData | None = None,
        *,
        n_epochs: int = 80,
        min_epochs: int = 10,
        patience: int = 10,
        batch_size: int = 192,
        learning_rate: float = 6e-4,
        seed: int | None = None,
        model_init_seed: int | None = None,
        control_context_seed: int | None = None,
        output_dir: str | Path | None = None,
        fast_dev_run: bool = False,
        config_overrides: Dict[str, Any] | None = None,
        **kwargs,
    ) -> "CAPRA":
        """Load a trained CAPRA checkpoint for inference.

        The same dataset assets used during training must be available so that
        the model can reconstruct shape-dependent modules and prediction
        batches. This method does not fit or modify model weights.
        """
        if data is not None:
            self.data = data
        if self.data is None:
            raise ValueError("load_capra_response_operator requires a CAPRAData object")
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        checkpoint_cfg = load_capra_checkpoint_config(checkpoint)

        results_root = Path(output_dir).expanduser().resolve() if output_dir is not None else None
        cfg = build_capra_config(
            study_name=self.data.study_name,
            seed=seed,
            model_init_seed=model_init_seed,
            control_context_seed=control_context_seed,
            results_root=results_root,
            output_name="capra_response_operator",
            fast_dev_run=bool(fast_dev_run),
            train_batch_size=int(batch_size),
            min_epochs=int(min_epochs),
            max_epochs=int(n_epochs),
            patience=int(patience),
            learning_rate=float(learning_rate),
            **kwargs,
        )
        _apply_overrides(cfg, config_overrides)
        if self.data.assets is None:
            self.data.build_control_relative_training_state(**_asset_build_kwargs_from_cfg(checkpoint_cfg))
        else:
            _validate_assets_match_cfg(
                self.data.assets,
                checkpoint_cfg,
                context="load_capra_response_operator",
            )
        _enable_torch_runtime(allow_tf32=bool(cfg["train"].get("allow_tf32", True)))
        model = CAPRATrainingModule.load_from_checkpoint(str(checkpoint), assets=self.data.assets)
        model = model.to(resolve_capra_device(cfg))
        run_dir = Path(cfg["paths"]["results_root"]).resolve() / cfg["project"]["output_name"]
        self.fitted = {
            "model": model,
            "assets": self.data.assets,
            "metrics": {
                "fit_skipped": True,
                "training_backend": "native_pytorch",
                "best_checkpoint": str(checkpoint),
            },
            "run_dir": run_dir,
        }
        return self

    def generate_counterfactual_profiles(
        self,
        pert_list: List[str] | None = None,
        *,
        n_pred: int = 500,
        batch_size: int | None = None,
        sampling_state: int | None = None,
        device: torch.device | str | None = None,
        control_prediction_mode: str | None = None,
        control_context_k_choices: Any | None = None,
        control_context_mean_local_weight: float | None = None,
    ) -> Dict[str, np.ndarray]:
        """Generate CAPRA counterfactual expression profiles."""
        if self.fitted is None or self.data is None or self.data.assets is None:
            raise ValueError("generate_counterfactual_profiles requires a trained model")

        cfg = self.fitted["model"].cfg
        if sampling_state is None:
            sampling_state = project_seed(cfg)
        batch_size = int(cfg["data"]["eval_batch_size"] if batch_size is None else batch_size)
        n_pred = int(n_pred)
        if n_pred <= 0:
            raise ValueError("n_pred must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if control_prediction_mode is None:
            control_prediction_mode = str(cfg["data"].get("control_context_mode", DEFAULT_CONTROL_CONTEXT_MODE))
        if control_context_k_choices is None:
            control_context_k_choices = cfg["data"].get("control_context_k_choices", "")
        if control_context_mean_local_weight is None:
            control_context_mean_local_weight = float(cfg["data"].get("control_context_mean_local_weight", 0.70))
        if pert_list is None:
            conditions = [condition for condition in self.data.assets.splits.get("test", []) if condition != "control"]
        else:
            conditions = list(pert_list)
        conditions = [canonical_condition_name(condition) for condition in conditions]

        device = torch.device(device) if device is not None else _current_model_device(self.fitted["model"].model)
        predictor = self.fitted["model"].model.to(device)
        predictor.eval()
        predictions: Dict[str, np.ndarray] = {}
        chunk_entries: list[tuple[str, int, int, Dict[str, torch.Tensor]]] = []
        chunk_rows = 0

        def flush_chunk() -> None:
            """Run one accumulated prediction chunk and append outputs by condition."""
            nonlocal chunk_entries, chunk_rows
            if not chunk_entries:
                return
            mega_inputs: Dict[str, list[torch.Tensor]] = {}
            chunk_meta: list[tuple[str, int]] = []
            for condition, start, end, batch in chunk_entries:
                materialized = _materialize_prediction_slice(batch, start=start, end=end, device=device)
                for key, value in materialized.items():
                    mega_inputs.setdefault(key, []).append(value)
                chunk_meta.append((condition, int(end - start)))

            mega_batch = {key: torch.cat(value, dim=0) for key, value in mega_inputs.items()}
            pred, _ = predictor(
                control_mean=mega_batch["control_mean"],
                control_std=mega_batch["control_std"],
                single_prior_deltas=mega_batch["single_prior_deltas"],
                single_prior_stats=mega_batch["single_prior_stats"],
                gene_embeddings=mega_batch["gene_embeddings"],
                gene_mask=mega_batch["gene_mask"],
                features=mega_batch["features"],
                prediction_noise_seed=mega_batch.get("prediction_noise_seed"),
            )
            pred_cpu = pred.detach().cpu().numpy()
            offset = 0
            for condition, size in chunk_meta:
                chunk_prediction = pred_cpu[offset : offset + size]
                if condition in predictions:
                    predictions[condition] = np.concatenate([predictions[condition], chunk_prediction], axis=0)
                else:
                    predictions[condition] = chunk_prediction
                offset += size
            chunk_entries = []
            chunk_rows = 0

        with torch.inference_mode():
            for condition in conditions:
                batch = build_prediction_batch(
                    assets=self.data.assets,
                    condition=condition,
                    n_pred=int(n_pred),
                    seed=int(sampling_state),
                    control_prediction_mode=control_prediction_mode,
                    control_context_k_choices=control_context_k_choices,
                    control_context_mean_local_weight=control_context_mean_local_weight,
                )
                condition_rows = int(batch["control_mean"].shape[0])
                if condition_rows > batch_size:
                    flush_chunk()
                    for start in range(0, condition_rows, batch_size):
                        end = min(condition_rows, start + batch_size)
                        chunk_entries = [(condition, start, end, batch)]
                        chunk_rows = end - start
                        flush_chunk()
                    continue
                if chunk_entries and chunk_rows + condition_rows > batch_size:
                    flush_chunk()
                chunk_entries.append((condition, 0, condition_rows, batch))
                chunk_rows += condition_rows
            flush_chunk()

        self.predictions = predictions
        return predictions

    def plot_training_metric(self, metric: str = "val_unseen_loss", *, ax=None):
        """Plot one metric from the native PyTorch training CSV for this run."""
        if self.fitted is None:
            raise ValueError("CAPRA.plot requires a trained model")
        import matplotlib.pyplot as plt

        run_dir = Path(self.fitted["run_dir"])
        metric_file = run_dir / "logs" / "metrics.csv"
        if not metric_file.exists():
            raise FileNotFoundError(f"no CAPRA metrics.csv found under {run_dir}")

        frame = pd.read_csv(metric_file)
        if metric not in frame.columns:
            available = ", ".join(sorted(column for column in frame.columns if column not in {"step"}))
            raise KeyError(f"metric={metric!r} not found. Available metrics: {available}")

        values = frame[["epoch", metric]].dropna()
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 3))
        ax.plot(values["epoch"], values[metric], marker="o", linewidth=1.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.set_title(f"CAPRA {metric}")
        return ax


__all__ = [
    "CAPRA",
    "CAPRAData",
    "build_capra_data",
    "compute_differential_expression_table",
    "load_differential_expression_table",
    "load_gene_embedding_table",
    "load_single_cell_adata",
]
