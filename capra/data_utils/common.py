from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from threadpoolctl import threadpool_limits
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from utils import (
    canonical_condition_name,
    split_condition_genes,
    stable_int_from_text,
)


MAX_PERTURBATION_GENES = 2
DEFAULT_CAPRA_SEED = 24
DEFAULT_CONTROL_CONTEXT_K_CHOICES = "2,4,8"
CAPRA_KNN_THREAD_LIMIT = 4


@dataclass
class DatasetAssets:
    """Immutable-style bundle of arrays and metadata used by CAPRA training.

    The public API constructs this object once from AnnData, split definitions,
    DEG masks, and GenePT embeddings. Training and prediction then operate on
    these precomputed assets without repeatedly touching AnnData internals.
    """

    study_name: str
    adata: ad.AnnData
    x_matrix: Any
    var_names: np.ndarray
    splits: Dict[str, List[str]]
    subgroup_map: Dict[str, str]
    condition_to_indices: Dict[str, np.ndarray]
    split_cell_indices: Dict[str, np.ndarray]
    control_indices: np.ndarray
    control_dense: np.ndarray
    control_strata: List[np.ndarray]
    dropped_embedding_conditions: Dict[str, List[str]]
    global_control_mean: np.ndarray
    global_control_std: np.ndarray
    train_single_delta: Dict[str, np.ndarray]
    asset_config: Dict[str, Any]
    deg_mask_by_condition: Dict[str, np.ndarray]
    condition_priors: Dict[str, Dict[str, np.ndarray]]
    embedding_table: pd.DataFrame
    embedding_dim: int


def _mean_over_indices(matrix: Any, indices: np.ndarray) -> np.ndarray:
    """Compute the gene-wise mean over selected rows of dense or sparse data."""
    sub = matrix[indices]
    if sparse.issparse(sub):
        mean = np.asarray(sub.mean(axis=0)).reshape(-1)
    else:
        mean = np.asarray(sub).mean(axis=0)
    return mean.astype(np.float32, copy=False)


def _dense_rows(matrix: Any, indices: np.ndarray) -> np.ndarray:
    """Materialize selected rows as a dense float32 array."""
    sub = matrix[indices]
    if sparse.issparse(sub):
        sub = sub.toarray()
    return np.asarray(sub, dtype=np.float32)


def _validate_control_sampling_inputs(
    control_dense: np.ndarray,
    *,
    n_samples: int,
    control_sample_size: int,
) -> None:
    """Validate control-context sampling dimensions before indexing."""
    if int(n_samples) < 0:
        raise ValueError("n_samples must be non-negative")
    if int(control_sample_size) <= 0:
        raise ValueError("control_sample_size must be positive")
    if int(control_dense.shape[0]) <= 0:
        raise ValueError("control sampling requires at least one control cell")


def _validate_mean_local_weight(mean_local_weight: float, *, mode: str) -> float:
    """Return a finite local/global mixing weight for control contexts."""
    local_weight = float(mean_local_weight)
    if not np.isfinite(local_weight):
        raise ValueError("control_context_mean_local_weight must be finite")
    if mode in {"mix_global_globalstd", "stratified_mix_global_globalstd"} and not 0.0 <= local_weight <= 1.0:
        raise ValueError("control_context_mean_local_weight must be in [0, 1] for mixed control modes")
    return local_weight


def _sample_control_means_dense(
    control_dense: np.ndarray,
    *,
    n_samples: int,
    control_sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample control-cell groups and return their mean expression profiles."""
    n_samples = int(n_samples)
    control_sample_size = int(control_sample_size)
    _validate_control_sampling_inputs(
        control_dense,
        n_samples=n_samples,
        control_sample_size=control_sample_size,
    )
    picked = rng.integers(0, control_dense.shape[0], size=(n_samples, control_sample_size))
    means = np.zeros((n_samples, control_dense.shape[1]), dtype=np.float32)
    for offset in range(control_sample_size):
        means += control_dense[picked[:, offset]]
    means /= np.float32(control_sample_size)
    return means.astype(np.float32, copy=False)


def _sample_control_stats_dense(
    control_dense: np.ndarray,
    *,
    n_samples: int,
    control_sample_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized sampling of control-group means and standard deviations."""
    n_samples = int(n_samples)
    control_sample_size = int(control_sample_size)
    _validate_control_sampling_inputs(
        control_dense,
        n_samples=n_samples,
        control_sample_size=control_sample_size,
    )
    picked = rng.integers(
        0,
        control_dense.shape[0],
        size=(n_samples, control_sample_size),
    )
    means = np.zeros((n_samples, control_dense.shape[1]), dtype=np.float32)
    for offset in range(control_sample_size):
        means += control_dense[picked[:, offset]]
    means /= np.float32(control_sample_size)

    variances = np.zeros_like(means)
    for offset in range(control_sample_size):
        centered = control_dense[picked[:, offset]] - means
        np.square(centered, out=centered)
        variances += centered
    variances /= np.float32(control_sample_size)
    np.maximum(variances, 0.0, out=variances)
    stds = np.sqrt(variances, out=variances)
    return means, stds


def _build_control_strata(control_dense: np.ndarray, *, n_strata: int = 8) -> List[np.ndarray]:
    """Partition control cells into expression strata for balanced sampling.

    Strata are built along the first principal direction of high-variance genes.
    The goal is to keep sampled control contexts diverse while remaining fully
    dataset-agnostic.
    """
    n_cells = int(control_dense.shape[0])
    if n_cells == 0:
        return [np.asarray([], dtype=np.int64)]
    n_strata = max(1, min(int(n_strata), n_cells))
    if n_strata == 1 or n_cells < 4:
        return [np.arange(n_cells, dtype=np.int64)]

    gene_var = np.var(control_dense, axis=0)
    top_gene_count = min(512, int(control_dense.shape[1]))
    top_genes = np.argsort(gene_var)[-top_gene_count:]
    sample_count = min(4096, n_cells)
    if sample_count == n_cells:
        sample_rows = np.arange(n_cells, dtype=np.int64)
    else:
        sample_rows = np.linspace(0, n_cells - 1, sample_count, dtype=np.int64)
    sampled = control_dense[sample_rows][:, top_genes].astype(np.float32, copy=False)
    center = sampled.mean(axis=0, dtype=np.float32)
    centered = sampled - center
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        pc1 = vh[0].astype(np.float32, copy=False)
        scores = (control_dense[:, top_genes] - center) @ pc1
    except np.linalg.LinAlgError:
        scores = control_dense[:, top_genes].mean(axis=1)

    edges = np.quantile(scores, np.linspace(0.0, 1.0, n_strata + 1))
    edges = np.unique(edges)
    if len(edges) <= 2:
        return [np.arange(n_cells, dtype=np.int64)]
    bin_ids = np.searchsorted(edges[1:-1], scores, side="right")
    strata = [np.flatnonzero(bin_ids == idx).astype(np.int64, copy=False) for idx in range(len(edges) - 1)]
    strata = [idx for idx in strata if len(idx) > 0]
    return strata or [np.arange(n_cells, dtype=np.int64)]


def _sample_from_strata(
    control_dense: np.ndarray,
    control_strata: List[np.ndarray] | None,
    *,
    n_samples: int,
    control_sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample control means while spreading each group across control strata."""
    if not control_strata:
        return _sample_control_means_dense(
            control_dense,
            n_samples=n_samples,
            control_sample_size=control_sample_size,
            rng=rng,
        )
    strata = [np.asarray(indices, dtype=np.int64) for indices in control_strata if len(indices) > 0]
    if not strata:
        strata = [np.arange(control_dense.shape[0], dtype=np.int64)]
    n_samples = int(n_samples)
    control_sample_size = int(control_sample_size)
    _validate_control_sampling_inputs(
        control_dense,
        n_samples=n_samples,
        control_sample_size=control_sample_size,
    )
    n_strata = len(strata)
    random_scores = rng.random((n_samples, n_strata))
    stratum_order = np.argsort(random_scores, axis=1)
    if control_sample_size > n_strata:
        repeats = int(np.ceil(control_sample_size / n_strata))
        stratum_order = np.tile(stratum_order, (1, repeats))
    sampled_strata = stratum_order[:, :control_sample_size]

    picked = np.empty((n_samples, control_sample_size), dtype=np.int64)
    for stratum_id, stratum in enumerate(strata):
        rows, offsets = np.nonzero(sampled_strata == stratum_id)
        if len(rows) == 0:
            continue
        picked[rows, offsets] = stratum[rng.integers(0, len(stratum), size=len(rows))]
    means = np.zeros((n_samples, control_dense.shape[1]), dtype=np.float32)
    for offset in range(control_sample_size):
        means += control_dense[picked[:, offset]]
    means /= np.float32(control_sample_size)
    return means.astype(np.float32, copy=False)


def _parse_control_k_choices(k_choices: Any, default_k: int | None = None) -> list[int]:
    """Parse allowed control sample sizes for multi-k context sampling."""
    if k_choices is None:
        if default_k is None:
            raise ValueError("control_context_k_choices must contain at least one positive integer")
        return [int(default_k)]
    if isinstance(k_choices, str):
        if not k_choices.strip():
            if default_k is None:
                raise ValueError("control_context_k_choices must contain at least one positive integer")
            return [int(default_k)]
        values = [item.strip() for item in k_choices.split(",") if item.strip()]
    elif np.isscalar(k_choices):
        values = [k_choices]
    else:
        values = list(k_choices)
    parsed_values: set[int] = set()
    for value in values:
        parsed_value = int(value)
        if parsed_value <= 0:
            raise ValueError("control_context_k_choices must contain only positive integers")
        parsed_values.add(parsed_value)
    parsed = sorted(parsed_values)
    if parsed:
        return parsed
    if default_k is None:
        raise ValueError("control_context_k_choices must contain at least one positive integer")
    return [int(default_k)]


def _primary_control_context_k(k_choices: Any) -> int:
    """Return the single-k control context size implied by the configured choices."""
    choices = _parse_control_k_choices(k_choices)
    return int(choices[len(choices) // 2])


def _sample_control_context_stats_dense(
    control_dense: np.ndarray,
    *,
    control_strata: List[np.ndarray] | None = None,
    global_control_mean: np.ndarray,
    global_control_std: np.ndarray | None = None,
    n_samples: int,
    control_sample_size: int,
    rng: np.random.Generator,
    mode: str = "sampled",
    k_choices: Any = None,
    mean_local_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample control context statistics under CAPRA's supported modes.

    Modes control whether the prediction context uses sampled local means,
    global control statistics, stratified sampling, or a mixture of local and
    global means. The returned arrays always have shape `(n_samples, n_genes)`.
    """
    mode = str(mode)
    n_samples = int(n_samples)
    control_sample_size = int(control_sample_size)
    _validate_control_sampling_inputs(
        control_dense,
        n_samples=n_samples,
        control_sample_size=control_sample_size,
    )
    local_weight = _validate_mean_local_weight(mean_local_weight, mode=mode)
    global_mean = np.repeat(
        global_control_mean[None, :],
        n_samples,
        axis=0,
    ).astype(np.float32, copy=False)
    if global_control_std is None:
        global_std_row = control_dense.std(axis=0, dtype=np.float32).astype(np.float32, copy=False)
    else:
        global_std_row = np.asarray(global_control_std, dtype=np.float32)
    global_std = np.repeat(global_std_row[None, :], n_samples, axis=0).astype(np.float32, copy=False)

    if mode == "sampled":
        return _sample_control_stats_dense(
            control_dense,
            n_samples=n_samples,
            control_sample_size=control_sample_size,
            rng=rng,
        )
    if mode == "global_mean":
        return global_mean, global_std

    if mode in {"sampled_globalstd", "mix_global_globalstd"}:
        sampled_mean = _sample_control_means_dense(
            control_dense,
            n_samples=n_samples,
            control_sample_size=control_sample_size,
            rng=rng,
        )
        if mode == "mix_global_globalstd":
            local_weight = float(mean_local_weight)
            sampled_mean = (
                local_weight * sampled_mean + (1.0 - local_weight) * global_mean
            ).astype(np.float32, copy=False)
        return sampled_mean.astype(np.float32, copy=False), global_std

    if mode == "multik_globalstd":
        choices = _parse_control_k_choices(k_choices, default_k=control_sample_size)
        means = np.empty((n_samples, control_dense.shape[1]), dtype=np.float32)
        selected_k = rng.choice(np.asarray(choices, dtype=np.int64), size=n_samples, replace=True)
        for k in choices:
            rows = np.flatnonzero(selected_k == int(k))
            if len(rows) == 0:
                continue
            means[rows] = _sample_control_means_dense(
                control_dense,
                n_samples=len(rows),
                control_sample_size=int(k),
                rng=rng,
            )
        return means, global_std

    if mode in {"stratified_globalstd", "stratified_mix_global_globalstd"}:
        sampled_mean = _sample_from_strata(
            control_dense,
            control_strata,
            n_samples=n_samples,
            control_sample_size=control_sample_size,
            rng=rng,
        )
        if mode == "stratified_mix_global_globalstd":
            local_weight = float(mean_local_weight)
            sampled_mean = (
                local_weight * sampled_mean + (1.0 - local_weight) * global_mean
            ).astype(np.float32, copy=False)
        return sampled_mean.astype(np.float32, copy=False), global_std

    if mode == "stratified_multik_globalstd":
        choices = _parse_control_k_choices(k_choices, default_k=control_sample_size)
        means = np.empty((n_samples, control_dense.shape[1]), dtype=np.float32)
        selected_k = rng.choice(np.asarray(choices, dtype=np.int64), size=n_samples, replace=True)
        for k in choices:
            rows = np.flatnonzero(selected_k == int(k))
            if len(rows) == 0:
                continue
            means[rows] = _sample_from_strata(
                control_dense,
                control_strata,
                n_samples=len(rows),
                control_sample_size=int(k),
                rng=rng,
            )
        return means, global_std

    raise ValueError(f"unknown control context mode={mode!r}")


def _condition_has_complete_embeddings(condition: str, embedding_index: set[str]) -> bool:
    """Check whether every perturbation gene in a condition has an embedding."""
    genes = split_condition_genes(condition)
    return all(gene in embedding_index for gene in genes)


def _unique_preserve_order(items: List[str]) -> List[str]:
    """Remove duplicates while preserving the first occurrence of each item."""
    seen: set[str] = set()
    ordered: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _project_seed(cfg: Dict[str, Any]) -> int:
    """Return CAPRA's unified run seed, accepting old keys only for config compatibility."""
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
    aliases = [
        int(project[name])
        for name in ("model_init_seed", "control_context_seed")
        if name in project
    ]
    if aliases:
        if len(set(aliases)) != 1:
            raise ValueError("CAPRA seed aliases disagree; use project.seed")
        return aliases[0]
    return DEFAULT_CAPRA_SEED


def _sample_seed(*parts: Any) -> int:
    """Derive a deterministic non-negative seed from integer/text parts."""
    text = "|".join(str(part) for part in parts)
    return stable_int_from_text(text)


def _filter_splits_by_embeddings(
    *,
    splits: Dict[str, List[str]],
    embedding_index: set[str],
    study_name: str,
) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Drop split conditions containing perturbation genes without embeddings."""
    filtered_splits: Dict[str, List[str]] = {}
    dropped_splits: Dict[str, List[str]] = {}

    for split_name, conditions in splits.items():
        kept_conditions: List[str] = []
        dropped_conditions: List[str] = []
        for condition in conditions:
            canonical_condition = canonical_condition_name(condition)
            if canonical_condition == "control" or _condition_has_complete_embeddings(canonical_condition, embedding_index):
                kept_conditions.append(canonical_condition)
            else:
                dropped_conditions.append(canonical_condition)
        filtered_splits[split_name] = _unique_preserve_order(kept_conditions)
        dropped_splits[split_name] = sorted(set(dropped_conditions))

    dropped_total = sorted({condition for values in dropped_splits.values() for condition in values})
    if dropped_total:
        split_counts = ", ".join(
            f"{split_name}={len(dropped_splits.get(split_name, []))}"
            for split_name in ("train", "val", "test")
        )
        warnings.warn(
            f"{study_name}: skipped {len(dropped_total)} conditions without complete gene embeddings "
            f"({split_counts})",
            RuntimeWarning,
        )
    return filtered_splits, dropped_splits


def _filter_adata_to_valid_conditions(adata: ad.AnnData, valid_conditions: set[str]) -> ad.AnnData:
    """Keep control cells plus perturbation cells whose conditions remain valid."""
    control_mask = adata.obs["perturbation_canonical"].astype(str).to_numpy() == "control"
    condition_values = adata.obs["condition_canonical"].astype(str).to_numpy(dtype=object, copy=False)
    condition_mask = np.isin(condition_values, sorted(valid_conditions))
    keep_mask = control_mask | condition_mask
    return adata[keep_mask]


def _validate_filtered_splits(splits: Dict[str, List[str]], study_name: str) -> None:
    """Ensure the filtered condition split still contains train perturbations."""
    train_conditions = [condition for condition in splits.get("train", []) if condition != "control"]
    if not train_conditions:
        raise ValueError(
            f"{study_name}: no train perturbation conditions remain after filtering missing gene embeddings"
        )


def _build_deg_masks_from_frames(
    *,
    adata_var_names: np.ndarray,
    top_k: int,
    deg_frames_by_condition: Dict[str, pd.DataFrame],
    allowed_conditions: List[str] | None = None,
) -> Dict[str, np.ndarray]:
    """Convert ranked DEG tables into binary Top-k supervision masks."""
    allowed_set = None
    if allowed_conditions is not None:
        allowed_set = {
            canonical_condition_name(condition)
            for condition in allowed_conditions
            if canonical_condition_name(condition) != "control"
        }
    gene_to_idx = {gene: idx for idx, gene in enumerate(adata_var_names.tolist())}
    masks: Dict[str, np.ndarray] = {}

    for condition, frame in deg_frames_by_condition.items():
        canonical_condition = canonical_condition_name(condition)
        if canonical_condition == "control" or frame is None:
            continue
        if allowed_set is not None and canonical_condition not in allowed_set:
            continue

        if "abs_scores" in frame.columns:
            ranked = frame.sort_values("abs_scores", ascending=False)
        elif "scores" in frame.columns:
            ranked = frame.assign(abs_scores=frame["scores"].abs()).sort_values("abs_scores", ascending=False)
        else:
            ranked = frame

        mask = np.zeros(len(adata_var_names), dtype=np.float32)
        for gene in ranked.index[:top_k]:
            idx = gene_to_idx.get(str(gene))
            if idx is not None:
                mask[idx] = 1.0
        masks[canonical_condition] = mask
    return masks


def _validate_supervised_deg_masks(
    *,
    splits: Dict[str, List[str]],
    deg_mask_by_condition: Dict[str, np.ndarray],
    study_name: str,
) -> None:
    """Fail fast if supervised train/validation conditions lack DEG masks."""
    missing: list[str] = []
    empty: list[str] = []
    for split_name in ("train", "val"):
        for condition in splits.get(split_name, []):
            canonical_condition = canonical_condition_name(condition)
            if canonical_condition == "control":
                continue
            mask = deg_mask_by_condition.get(canonical_condition)
            if mask is None:
                missing.append(f"{split_name}:{canonical_condition}")
            elif float(mask.sum()) <= 0.0:
                empty.append(f"{split_name}:{canonical_condition}")
    if missing or empty:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if empty:
            details.append(f"empty={empty}")
        raise ValueError(
            f"{study_name}: train/validation DEG masks are incomplete; "
            + "; ".join(details)
        )


def _get_gene_vector(embedding_table: pd.DataFrame, gene: str) -> np.ndarray:
    """Fetch a perturbation gene embedding as float32, failing loudly if absent."""
    if gene in embedding_table.index:
        return embedding_table.loc[gene].to_numpy(dtype=np.float32)
    raise KeyError(f"missing gene embedding for perturbation gene: {gene}")


def _weighted_average(vectors: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Return a normalized weighted average of same-shaped vectors."""
    stacked = np.stack(vectors, axis=0)
    weights = weights / np.clip(weights.sum(), 1e-8, None)
    return (stacked * weights[:, None]).sum(axis=0).astype(np.float32)


def _genept_knn_similarity(query: np.ndarray, bank_matrix: np.ndarray) -> np.ndarray:
    """Compute GenePT KNN similarities under CAPRA's local replay thread limit."""
    with threadpool_limits(limits=CAPRA_KNN_THREAD_LIMIT):
        return cosine_similarity(query, bank_matrix)[0].astype(np.float32)


def _build_condition_priors(
    *,
    conditions: List[str],
    embedding_table: pd.DataFrame,
    train_single_delta: Dict[str, np.ndarray],
    global_control_mean: np.ndarray,
    knn_topk: int,
    knn_temperature: float,
    embedding_dim: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Build per-condition priors used by CAPRA's response operator.

    For each single or double perturbation, this function prepares two-slot
    gene embeddings, slot masks, empirical or KNN-retrieved single-gene deltas,
    and reliability features consumed by CAPRA's response operator.
    """
    if train_single_delta:
        bank_genes = sorted(train_single_delta.keys())
        bank_matrix = np.stack([_get_gene_vector(embedding_table, gene) for gene in bank_genes], axis=0)
    else:
        bank_genes = []
        bank_matrix = np.zeros((1, embedding_dim), dtype=np.float32)

    condition_priors: Dict[str, Dict[str, np.ndarray]] = {}
    for condition in conditions:
        genes = split_condition_genes(condition)
        if len(genes) > MAX_PERTURBATION_GENES:
            raise ValueError(
                f"CAPRA supports at most {MAX_PERTURBATION_GENES} perturbation genes per condition; "
                f"got {condition!r}"
            )
        gene_embeddings = np.zeros((2, embedding_dim), dtype=np.float32)
        gene_mask = np.zeros(2, dtype=np.float32)
        single_prior_deltas = np.zeros((2, global_control_mean.shape[0]), dtype=np.float32)
        single_prior_stats = np.zeros((2, 4), dtype=np.float32)
        exact_seen = 0
        nn_scores: List[float] = []
        topk_scores: List[float] = []

        for gene_idx, gene in enumerate(genes):
            gene_mask[gene_idx] = 1.0
            gene_embeddings[gene_idx] = _get_gene_vector(embedding_table, gene)
            if bank_genes:
                query = gene_embeddings[gene_idx].reshape(1, -1)
                sims = _genept_knn_similarity(query, bank_matrix)
                top_idx = np.argsort(sims)[::-1][: max(1, min(knn_topk, len(bank_genes)))]
                top_sims = sims[top_idx]
                top_genes = [bank_genes[idx] for idx in top_idx]
                nn_score = float(top_sims[0])
                topk_mean = float(np.mean(top_sims))
                topk_std = float(np.std(top_sims))
                nn_scores.append(nn_score)
                topk_scores.extend(top_sims.tolist())
                weights = np.exp(top_sims * knn_temperature)
                retrieved_knn = _weighted_average([train_single_delta[item] for item in top_genes], weights)
            else:
                nn_score = 0.0
                topk_mean = 0.0
                topk_std = 0.0
                nn_scores.append(nn_score)
                topk_scores.append(0.0)
                retrieved_knn = np.zeros_like(global_control_mean)

            if gene in train_single_delta:
                exact_seen += 1
                exact_delta = train_single_delta[gene]
                single_prior_deltas[gene_idx] = exact_delta.astype(np.float32)
                single_prior_stats[gene_idx] = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
            else:
                single_prior_deltas[gene_idx] = retrieved_knn.astype(np.float32)
                single_prior_stats[gene_idx] = np.array([0.0, nn_score, topk_mean, topk_std], dtype=np.float32)

        n_genes = max(1, len(genes))
        prior_features = np.array(
            [
                float(len(genes) == 1),
                float(len(genes) == 2),
                float(exact_seen),
                float(exact_seen) / float(n_genes),
                float(np.mean(nn_scores) if nn_scores else 0.0),
                float(np.std(nn_scores) if nn_scores else 0.0),
                float(np.mean(topk_scores) if topk_scores else 0.0),
                float(np.std(topk_scores) if topk_scores else 0.0),
            ],
            dtype=np.float32,
        )
        condition_priors[condition] = {
            "gene_embeddings": gene_embeddings,
            "gene_mask": gene_mask,
            "single_prior_deltas": single_prior_deltas.astype(np.float32),
            "single_prior_stats": single_prior_stats.astype(np.float32),
            "features": prior_features,
        }
    return condition_priors


def load_condition_dataset_from_adata(
    *,
    study_name: str = "capra_study",
    adata: ad.AnnData,
    splits: Dict[str, List[str]],
    embedding_table: pd.DataFrame,
    topk_deg: int = 100,
    knn_topk: int = 5,
    knn_temperature: float = 12.0,
    deg_frames_by_condition: Dict[str, pd.DataFrame] | None = None,
) -> DatasetAssets:
    """Construct all CAPRA training assets from an AnnData object.

    This is the main data-construction entry point. It canonicalizes condition
    labels, filters conditions without complete embeddings, computes control
    summaries, builds Top-k DEG masks, and prepares condition priors shared by
    training, validation, and prediction.
    """
    if deg_frames_by_condition is None:
        raise ValueError("deg_frames_by_condition is required")

    adata = adata.copy()
    adata.obs["condition_canonical"] = adata.obs["condition"].astype(str).map(canonical_condition_name)
    adata.obs["perturbation_canonical"] = adata.obs["perturbation"].astype(str).map(canonical_condition_name)

    canonical_splits: Dict[str, List[str]] = {}
    for key, values in splits.items():
        canonical_splits[key] = [canonical_condition_name(item) for item in values]

    embedding_table = embedding_table.copy()
    embedding_table.index = embedding_table.index.astype(str)
    embedding_table = embedding_table.astype(np.float32)
    embedding_dim = int(embedding_table.shape[1])
    canonical_splits, dropped_embedding_conditions = _filter_splits_by_embeddings(
        splits=canonical_splits,
        embedding_index=set(embedding_table.index.astype(str)),
        study_name=study_name,
    )
    _validate_filtered_splits(canonical_splits, study_name)

    valid_conditions = {
        condition
        for split_name in ("train", "val", "test")
        for condition in canonical_splits.get(split_name, [])
        if condition != "control"
    }
    adata = _filter_adata_to_valid_conditions(adata, valid_conditions)
    x_matrix = adata.X.tocsr() if sparse.issparse(adata.X) else np.asarray(adata.X, dtype=np.float32)
    var_names = np.asarray(adata.var_names).astype(str)

    conditions = adata.obs["condition_canonical"].astype(str).to_numpy()
    condition_to_indices: Dict[str, np.ndarray] = {}
    for condition in sorted(np.unique(conditions).tolist()):
        condition_to_indices[condition] = np.where(conditions == condition)[0]

    missing_observed = []
    for split_name in ("train", "val"):
        for condition in canonical_splits.get(split_name, []):
            if condition != "control" and condition not in condition_to_indices:
                missing_observed.append(f"{split_name}:{condition}")
    if missing_observed:
        raise ValueError(
            f"{study_name}: split conditions are absent from the training AnnData; "
            + "; ".join(missing_observed)
        )

    control_indices = np.where(adata.obs["perturbation_canonical"].to_numpy() == "control")[0]
    if len(control_indices) == 0:
        raise ValueError(f"{study_name}: no control cells remain after condition filtering")
    control_dense = _dense_rows(x_matrix, control_indices)
    control_strata = _build_control_strata(control_dense, n_strata=8)
    split_cell_indices: Dict[str, np.ndarray] = {}
    for split_name in ("train", "val", "test"):
        allowed = set(item for item in canonical_splits[split_name] if item != "control")
        indices = np.where(np.isin(conditions, list(allowed)))[0]
        split_cell_indices[split_name] = indices
    empty_supervised_splits = [name for name in ("train", "val") if len(split_cell_indices[name]) == 0]
    if empty_supervised_splits:
        raise ValueError(
            f"{study_name}: supervised splits have no perturbation cells after filtering: "
            + ", ".join(empty_supervised_splits)
        )

    global_control_mean = _mean_over_indices(x_matrix, control_indices)
    global_control_std = control_dense.std(axis=0, dtype=np.float32).astype(np.float32, copy=False)
    train_single_delta: Dict[str, np.ndarray] = {}
    train_conditions = [item for item in canonical_splits["train"] if item != "control"]
    for condition in train_conditions:
        indices = condition_to_indices.get(condition)
        if indices is None or len(indices) == 0:
            continue
        mean_expr = _mean_over_indices(x_matrix, indices)
        delta = (mean_expr - global_control_mean).astype(np.float32)
        genes = split_condition_genes(condition)
        if len(genes) == 1:
            train_single_delta[genes[0]] = delta

    deg_mask_by_condition = _build_deg_masks_from_frames(
        adata_var_names=var_names,
        top_k=int(topk_deg),
        deg_frames_by_condition=deg_frames_by_condition,
        allowed_conditions=canonical_splits["train"] + canonical_splits["val"],
    )
    _validate_supervised_deg_masks(
        splits=canonical_splits,
        deg_mask_by_condition=deg_mask_by_condition,
        study_name=study_name,
    )
    union_conditions = sorted(valid_conditions)
    condition_priors = _build_condition_priors(
        conditions=union_conditions,
        embedding_table=embedding_table,
        train_single_delta=train_single_delta,
        global_control_mean=global_control_mean,
        knn_topk=int(knn_topk),
        knn_temperature=float(knn_temperature),
        embedding_dim=embedding_dim,
    )

    return DatasetAssets(
        study_name=study_name,
        adata=adata,
        x_matrix=x_matrix,
        var_names=var_names,
        splits=canonical_splits,
        subgroup_map={},
        condition_to_indices=condition_to_indices,
        split_cell_indices=split_cell_indices,
        control_indices=control_indices,
        control_dense=control_dense,
        control_strata=control_strata,
        dropped_embedding_conditions=dropped_embedding_conditions,
        global_control_mean=global_control_mean,
        global_control_std=global_control_std,
        train_single_delta=train_single_delta,
        asset_config={
            "topk_deg": int(topk_deg),
            "knn_topk": int(knn_topk),
            "knn_temperature": float(knn_temperature),
        },
        deg_mask_by_condition=deg_mask_by_condition,
        condition_priors=condition_priors,
        embedding_table=embedding_table,
        embedding_dim=embedding_dim,
    )


class PerturbationCellDataset(Dataset):
    """Cell-level dataset used for supervised CAPRA training."""

    def __init__(
        self,
        assets: DatasetAssets,
        split_name: str,
        base_seed: int,
        train: bool,
        control_context_mode: str = "sampled",
        control_context_k_choices: Any = DEFAULT_CONTROL_CONTEXT_K_CHOICES,
        control_context_mean_local_weight: float = 1.0,
    ) -> None:
        """Bind assets and configure per-item control-context sampling.

        Training mode samples deterministic item-local control contexts keyed by
        epoch and item index. Evaluation mode precomputes fixed contexts.
        """
        self.assets = assets
        self.split_name = split_name
        self.base_seed = int(base_seed)
        self.train = bool(train)
        self.control_context_mode = str(control_context_mode)
        self.control_context_k_choices = control_context_k_choices
        self.control_context_size = _primary_control_context_k(self.control_context_k_choices)
        self.control_context_mean_local_weight = float(control_context_mean_local_weight)
        self.indices = assets.split_cell_indices[split_name]
        self.conditions = assets.adata.obs["condition_canonical"].astype(str).to_numpy()[self.indices]
        self.targets = _dense_rows(assets.x_matrix, self.indices)
        self.control_dense = assets.control_dense
        self.empty_mask = np.zeros(assets.adata.n_vars, dtype=np.float32)
        self._worker_rng: np.random.Generator | None = None
        self.epoch_index = 0
        self.fixed_control_means = None
        self.fixed_control_stds = None
        if not self.train:
            self.fixed_control_means, self.fixed_control_stds = _sample_control_context_stats_dense(
                self.control_dense,
                control_strata=self.assets.control_strata,
                global_control_mean=self.assets.global_control_mean,
                global_control_std=self.assets.global_control_std,
                n_samples=len(self.indices),
                control_sample_size=self.control_context_size,
                rng=np.random.default_rng(self.base_seed),
                mode=self.control_context_mode,
                k_choices=self.control_context_k_choices,
                mean_local_weight=self.control_context_mean_local_weight,
            )

    def __len__(self) -> int:
        """Return the number of perturbed cells in the selected split."""
        return int(len(self.indices))

    def _get_worker_rng(self) -> np.random.Generator:
        """Return a worker-local RNG for legacy non-training fallback paths."""
        if self._worker_rng is None:
            worker = get_worker_info()
            worker_id = worker.id if worker is not None else 0
            self._worker_rng = np.random.default_rng(self.base_seed + 100_003 * worker_id)
        return self._worker_rng

    def set_epoch(self, epoch_index: int) -> None:
        """Set the epoch used for item-local deterministic control-context sampling."""
        self.epoch_index = int(epoch_index)

    def _sample_control_stats(self, item_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return control context statistics for one dataset item."""
        if self.fixed_control_means is not None:
            return self.fixed_control_means[item_index], self.fixed_control_stds[item_index]
        if self.train:
            rng = np.random.default_rng(_sample_seed(self.base_seed, self.epoch_index, item_index))
        else:
            rng = self._get_worker_rng()
        means, stds = _sample_control_context_stats_dense(
            self.control_dense,
            control_strata=self.assets.control_strata,
            global_control_mean=self.assets.global_control_mean,
            global_control_std=self.assets.global_control_std,
            n_samples=1,
            control_sample_size=self.control_context_size,
            rng=rng,
            mode=self.control_context_mode,
            k_choices=self.control_context_k_choices,
            mean_local_weight=self.control_context_mean_local_weight,
        )
        return means[0], stds[0]

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """Return one cell-level training example with target and prior tensors."""
        condition = self.conditions[item]
        target = self.targets[item]
        priors = self.assets.condition_priors[condition]
        control_mean, control_std = self._sample_control_stats(item)
        deg_mask = self.assets.deg_mask_by_condition.get(condition, self.empty_mask)

        return {
            "condition": condition,
            "target": target,
            "control_mean": control_mean,
            "control_std": control_std,
            "deg_mask": deg_mask,
            "gene_embeddings": priors["gene_embeddings"],
            "gene_mask": priors["gene_mask"],
            "features": priors["features"],
            "single_prior_deltas": priors["single_prior_deltas"],
            "single_prior_stats": priors["single_prior_stats"],
        }


class PerturbationConditionDataset(Dataset):
    """Condition-level validation dataset with prebuilt prediction batches."""

    def __init__(
        self,
        assets: DatasetAssets,
        split_name: str,
        base_seed: int,
        n_pred: int,
        control_context_mode: str = "sampled",
        control_context_k_choices: Any = DEFAULT_CONTROL_CONTEXT_K_CHOICES,
        control_context_mean_local_weight: float = 1.0,
    ) -> None:
        """Precompute test-like validation inputs for every condition in a split."""
        self.assets = assets
        self.split_name = split_name
        self.base_seed = int(base_seed)
        self.n_pred = int(n_pred)
        self.control_context_mode = str(control_context_mode)
        self.control_context_k_choices = control_context_k_choices
        self.control_context_size = _primary_control_context_k(self.control_context_k_choices)
        self.control_context_mean_local_weight = float(control_context_mean_local_weight)
        self.conditions = [item for item in assets.splits[split_name] if item != "control"]
        self.empty_mask = np.zeros(assets.adata.n_vars, dtype=np.float32)
        self.target_mean_by_condition: Dict[str, np.ndarray] = {}
        self.prediction_batch_by_condition: Dict[str, Dict[str, torch.Tensor]] = {}

        for condition in self.conditions:
            indices = assets.condition_to_indices.get(condition)
            if indices is None or len(indices) == 0:
                continue
            # Validation uses unseen perturbation conditions in a test-like way:
            # we compare condition-level means instead of forcing a fake cell pairing.
            self.target_mean_by_condition[condition] = _mean_over_indices(assets.x_matrix, indices)
            self.prediction_batch_by_condition[condition] = build_prediction_batch(
                assets=self.assets,
                condition=condition,
                n_pred=self.n_pred,
                seed=self.base_seed,
                control_prediction_mode=self.control_context_mode,
                control_context_k_choices=self.control_context_k_choices,
                control_context_mean_local_weight=self.control_context_mean_local_weight,
            )

    def __len__(self) -> int:
        """Return the number of non-control perturbation conditions."""
        return int(len(self.conditions))

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """Return one condition-level validation batch and target mean."""
        condition = self.conditions[item]
        prediction_batch = {
            key: value.clone()
            for key, value in self.prediction_batch_by_condition[condition].items()
        }
        target_mean = self.target_mean_by_condition.get(condition)
        if target_mean is None:
            target_mean = np.zeros(self.assets.adata.n_vars, dtype=np.float32)
        deg_mask = self.assets.deg_mask_by_condition.get(condition, self.empty_mask)
        prediction_batch.update(
            {
                "condition": condition,
                "target_mean": target_mean.astype(np.float32, copy=False),
                "deg_mask": deg_mask.astype(np.float32, copy=False),
            }
        )
        return prediction_batch


def build_dataloaders(assets: DatasetAssets, cfg: Dict[str, Any]) -> Dict[str, DataLoader]:
    """Create CAPRA train and validation dataloaders from assets and config."""
    run_seed = _project_seed(cfg)
    num_workers = int(cfg["data"]["num_workers"])
    train_batch_size = int(cfg["data"]["train_batch_size"])
    train_loader_kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(cfg["data"]["pin_memory"]),
        "drop_last": False,
    }
    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = 4
    val_loader_kwargs: Dict[str, Any] = {
        "num_workers": 0,
        "pin_memory": bool(cfg["data"]["pin_memory"]),
        "drop_last": False,
    }
    train_dataset = PerturbationCellDataset(
        assets=assets,
        split_name="train",
        base_seed=run_seed,
        train=True,
        control_context_mode=str(cfg["data"].get("control_context_mode", "sampled")),
        control_context_k_choices=cfg["data"].get("control_context_k_choices", ""),
        control_context_mean_local_weight=float(cfg["data"].get("control_context_mean_local_weight", 1.0)),
    )
    val_dataset = PerturbationConditionDataset(
        assets=assets,
        split_name="val",
        base_seed=run_seed + 10_000,
        n_pred=int(cfg["data"]["val_n_pred"]),
        control_context_mode=str(cfg["data"].get("control_context_mode", "sampled")),
        control_context_k_choices=cfg["data"].get("control_context_k_choices", ""),
        control_context_mean_local_weight=float(cfg["data"].get("control_context_mean_local_weight", 1.0)),
    )
    if len(train_dataset) == 0:
        raise ValueError(f"{assets.study_name}: training split has no perturbation cells")
    if len(val_dataset) == 0:
        raise ValueError(f"{assets.study_name}: validation split has no perturbation conditions")
    train_generator = torch.Generator()
    train_generator.manual_seed(run_seed + 20_000)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        generator=train_generator,
        **train_loader_kwargs,
    )
    return {
        "train": train_loader,
        "val": DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            **val_loader_kwargs,
        ),
    }


def build_prediction_batch(
    assets: DatasetAssets,
    condition: str,
    n_pred: int,
    seed: int,
    control_prediction_mode: str = "sampled",
    control_context_k_choices: Any = DEFAULT_CONTROL_CONTEXT_K_CHOICES,
    control_context_mean_local_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Build a tensor batch for generating one perturbation condition.

    The batch repeats condition-level gene/prior features and samples `n_pred`
    control contexts. `generate_counterfactual_profiles` may concatenate these
    batches across conditions for efficient forward passes.
    """
    priors = assets.condition_priors[condition]
    rng = np.random.default_rng(seed + stable_int_from_text(condition))
    n_pred = int(n_pred)
    control_sample_size = _primary_control_context_k(control_context_k_choices)
    control_mean, control_std = _sample_control_context_stats_dense(
        assets.control_dense,
        control_strata=assets.control_strata,
        global_control_mean=assets.global_control_mean,
        global_control_std=assets.global_control_std,
        n_samples=n_pred,
        control_sample_size=control_sample_size,
        rng=rng,
        mode=str(control_prediction_mode),
        k_choices=control_context_k_choices,
        mean_local_weight=float(control_context_mean_local_weight),
    )
    prediction_noise_seed = np.asarray(
        [stable_int_from_text(f"{condition}|{int(seed)}|{sample_index}") for sample_index in range(n_pred)],
        dtype=np.int64,
    )
    feature_batch = np.ascontiguousarray(priors["features"][None, :], dtype=np.float32)
    gene_embedding_batch = np.ascontiguousarray(priors["gene_embeddings"][None, :, :], dtype=np.float32)
    gene_mask_batch = np.ascontiguousarray(priors["gene_mask"][None, :], dtype=np.float32)
    single_prior_delta_batch = np.ascontiguousarray(priors["single_prior_deltas"][None, :, :], dtype=np.float32)
    single_prior_stat_batch = np.ascontiguousarray(priors["single_prior_stats"][None, :, :], dtype=np.float32)
    return {
        "control_mean": torch.from_numpy(control_mean),
        "control_std": torch.from_numpy(control_std),
        "prediction_noise_seed": torch.from_numpy(prediction_noise_seed),
        "features": torch.from_numpy(feature_batch),
        "gene_embeddings": torch.from_numpy(gene_embedding_batch),
        "gene_mask": torch.from_numpy(gene_mask_batch),
        "single_prior_deltas": torch.from_numpy(single_prior_delta_batch),
        "single_prior_stats": torch.from_numpy(single_prior_stat_batch),
    }
