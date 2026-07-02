from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from utils import (
    apply_sparse_residual_guard,
    TorchInitializationContext,
    deterministic_standard_normal_like,
    variance_calibration_noise_scale,
)


# CAPRA models single- and double-gene perturbations with a fixed two-slot
# representation.  Single perturbations use slot 0; the second slot is zeroed.
NUM_PERTURBATION_SLOTS = 2
class GeneAwareResidualDecoder(nn.Module):
    """Decode residuals with expression-gene GenePT features as fixed targets."""

    def __init__(
        self,
        *,
        latent_dim: int,
        embedding_dim: int,
        num_genes: int,
        dropout: float,
        target_gene_embeddings: torch.Tensor,
        init: TorchInitializationContext,
    ) -> None:
        super().__init__()
        min_decoder_dim = 32
        max_decoder_dim = 128
        decoder_dim = max(min_decoder_dim, min(max_decoder_dim, int(latent_dim)))
        target_gene_embeddings = torch.as_tensor(target_gene_embeddings, dtype=torch.float32)
        expected_shape = (int(num_genes), int(embedding_dim))
        if target_gene_embeddings.shape != expected_shape:
            raise ValueError(f"target_gene_embeddings must have shape {expected_shape}")

        target_gene_mask = target_gene_embeddings.abs().sum(dim=1, keepdim=True) > 0.0
        self.target_scale = 1.0 / math.sqrt(float(decoder_dim))
        self.normalize_eps = 1e-8
        self.register_buffer("target_gene_embeddings", target_gene_embeddings.contiguous())
        self.register_buffer("target_gene_mask", target_gene_mask.contiguous())
        self.latent_context = nn.Sequential(
            nn.LayerNorm(latent_dim),
            init.linear(
                latent_dim,
                latent_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            init.linear_without_global_rng(
                latent_dim,
                decoder_dim,
            ),
        )
        self.target_encoder = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            init.linear(
                embedding_dim,
                decoder_dim,
            ),
            nn.GELU(),
            init.linear(
                decoder_dim,
                decoder_dim,
            ),
        )
        init.zero_init_final_linear(self.latent_context)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        context = self.latent_context(latent)
        target = self.target_encoder(self.target_gene_embeddings)
        normalize_eps = max(self.normalize_eps, torch.finfo(target.dtype).tiny)
        target = F.normalize(target, p=2, dim=1, eps=normalize_eps) * self.target_gene_mask
        return torch.matmul(context, target.transpose(0, 1)) * self.target_scale


class CAPRAModel(nn.Module):
    """Control-anchored perturbation response operator.

    The model receives a sampled control context, per-slot perturbation gene
    embeddings, single-gene prior deltas, and condition-level reliability
    features. It predicts expression as a control-relative anchor plus learned
    additive and pair residuals. The architecture is intentionally shared
    across datasets and contains no dataset-specific routing.
    """

    def __init__(
        self,
        *,
        num_genes: int,
        embedding_dim: int,
        latent_dim: int,
        gene_hidden_dim: int,
        prior_hidden_dim: int,
        operator_hidden_dim: int,
        operator_rank: int,
        feature_dim: int,
        single_stat_dim: int,
        dropout: float,
        target_gene_embeddings: torch.Tensor,
        initialization_seed: int,
    ) -> None:
        """Initialize CAPRA's encoders, low-rank operator head, and decoders.

        Parameters define dimensionality only; all biological or dataset state
        enters through tensors supplied to `forward`. `initialization_seed` is
        only a local parameter-initialization seed and is never used for data
        splitting, sampling, or external evaluation selection.
        """
        super().__init__()
        initial_composition_gain = 1.0
        initial_residual_gate_logit = -2.0
        additive_residual_output_scale = 1.0
        pair_residual_output_scale = 0.25
        initialization_seed = int(initialization_seed)
        init = TorchInitializationContext(initialization_seed)

        self.latent_dim = int(latent_dim)
        self.operator_rank = int(operator_rank)
        self.additive_residual_output_scale = additive_residual_output_scale
        self.pair_residual_output_scale = pair_residual_output_scale
        # Fixed inference-time variance calibration (SE floor from control stats).
        self._se_floor_mult = 2.0
        self._se_floor_expressed_threshold = 0.01
        self._se_floor_noise_scale = 0.5
        # Global composition gain is shared across all datasets and perturbations.
        # It only rescales the universal interaction term; it does not introduce
        # any dataset- or condition-specific behavior.
        self.composition_gain_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(initial_composition_gain)), dtype=torch.float32)
        )
        # Shared residual gains keep the model anchored to the empirical single-prior
        # baseline at initialization and across all datasets.
        self.single_residual_gain_raw = nn.Parameter(torch.tensor(initial_residual_gate_logit, dtype=torch.float32))
        self.pair_residual_gain_raw = nn.Parameter(torch.tensor(initial_residual_gate_logit, dtype=torch.float32))

        self.control_encoder = nn.Sequential(
            nn.LayerNorm(num_genes * 2),
            init.linear(
                num_genes * 2,
                latent_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            init.linear(
                latent_dim,
                latent_dim,
            ),
        )
        self.prior_delta_encoder = nn.Sequential(
            nn.LayerNorm(num_genes),
            init.linear(
                num_genes,
                prior_hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            init.linear(
                prior_hidden_dim,
                prior_hidden_dim,
            ),
            nn.GELU(),
        )
        self.gene_encoder = nn.Sequential(
            init.linear(
                embedding_dim + prior_hidden_dim + single_stat_dim + feature_dim + 1,
                gene_hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            init.linear(
                gene_hidden_dim,
                gene_hidden_dim,
            ),
            nn.GELU(),
        )

        operator_param_dim = 2 * latent_dim * operator_rank + operator_rank + latent_dim
        self.operator_head = nn.Sequential(
            init.linear(
                gene_hidden_dim,
                operator_hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            init.linear(
                operator_hidden_dim,
                operator_param_dim,
            ),
        )
        self.additive_residual_decoder = GeneAwareResidualDecoder(
            latent_dim=latent_dim,
            embedding_dim=embedding_dim,
            num_genes=num_genes,
            dropout=dropout,
            target_gene_embeddings=target_gene_embeddings,
            init=init,
        )
        self.pair_residual_decoder = GeneAwareResidualDecoder(
            latent_dim=latent_dim,
            embedding_dim=embedding_dim,
            num_genes=num_genes,
            dropout=dropout,
            target_gene_embeddings=target_gene_embeddings,
            init=init,
        )


    def _split_operator_params(
        self,
        params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reshape flat operator-head output into low-rank operator factors.

        Returns slot-specific `(u, v, singular, bias)` tensors used to transform
        the encoded control state for each perturbation gene slot.
        """
        batch_size, n_slots, _ = params.shape
        offset = 0
        total_uv = self.latent_dim * self.operator_rank

        u = params[:, :, offset : offset + total_uv].reshape(
            batch_size, n_slots, self.latent_dim, self.operator_rank
        )
        offset += total_uv
        v = params[:, :, offset : offset + total_uv].reshape(
            batch_size, n_slots, self.latent_dim, self.operator_rank
        )
        offset += total_uv
        singular = params[:, :, offset : offset + self.operator_rank]
        offset += self.operator_rank
        bias = params[:, :, offset : offset + self.latent_dim]
        return u, v, singular, bias

    @staticmethod
    def _apply_single_operator(
        vector: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        singular: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one low-rank operator to one latent vector per batch item."""
        projected = torch.einsum("bl,blr->br", vector, v)
        projected = projected * singular
        return torch.einsum("br,blr->bl", projected, u)

    def _apply_slot_operators(
        self,
        base: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        singular: torch.Tensor,
    ) -> torch.Tensor:
        """Apply both perturbation-slot operators to the control latent state."""
        projected = torch.einsum("bl,bslr->bsr", base, v)
        projected = projected * singular
        return torch.einsum("bsr,bslr->bsl", projected, u)

    @staticmethod
    def _validate_forward_inputs(
        *,
        control_mean: torch.Tensor,
        control_std: torch.Tensor,
        single_prior_deltas: torch.Tensor,
        single_prior_stats: torch.Tensor,
        gene_embeddings: torch.Tensor,
        gene_mask: torch.Tensor,
        features: torch.Tensor,
        prediction_noise_seed: torch.Tensor | None = None,
    ) -> None:
        """Fail fast when a caller violates CAPRA's fixed two-slot tensor contract."""
        if control_mean.dim() != 2:
            raise ValueError("control_mean must have shape [batch, genes]")
        if control_std.shape != control_mean.shape:
            raise ValueError("control_std must have the same shape as control_mean")

        batch_size, num_genes = control_mean.shape
        expected_slot_shape = (batch_size, NUM_PERTURBATION_SLOTS)
        if gene_mask.shape != expected_slot_shape:
            raise ValueError(f"gene_mask must have shape {expected_slot_shape}")
        if gene_embeddings.dim() != 3 or gene_embeddings.shape[:2] != expected_slot_shape:
            raise ValueError("gene_embeddings must have shape [batch, 2, embedding_dim]")
        if single_prior_deltas.dim() != 3 or single_prior_deltas.shape != (
            batch_size,
            NUM_PERTURBATION_SLOTS,
            num_genes,
        ):
            raise ValueError("single_prior_deltas must have shape [batch, 2, genes]")
        if single_prior_stats.dim() != 3 or single_prior_stats.shape[:2] != expected_slot_shape:
            raise ValueError("single_prior_stats must have shape [batch, 2, stat_dim]")
        if features.dim() != 2 or features.shape[0] != batch_size:
            raise ValueError("features must have shape [batch, feature_dim]")
        if prediction_noise_seed is not None and prediction_noise_seed.shape != (batch_size,):
            raise ValueError(f"prediction_noise_seed must have shape {(batch_size,)}")

    def forward(
        self,
        *,
        control_mean: torch.Tensor,
        control_std: torch.Tensor,
        single_prior_deltas: torch.Tensor,
        single_prior_stats: torch.Tensor,
        gene_embeddings: torch.Tensor,
        gene_mask: torch.Tensor,
        features: torch.Tensor,
        prediction_noise_seed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Predict perturbed expression profiles and regularization tensors.

        Parameters
        ----------
        control_mean / control_std:
            Control context statistics for each predicted cell.
        single_prior_deltas / single_prior_stats:
            Two perturbation slots containing empirical or KNN-retrieved
            single-gene priors and reliability summaries.
        gene_embeddings / gene_mask:
            GenePT embeddings and binary slot mask. Single-gene perturbations
            use only slot 0; double perturbations use both slots.
        features:
            Condition-level summary features produced during data construction.
        prediction_noise_seed:
            Optional per-row seeds used only for deterministic inference-time
            predictive variance calibration.

        Returns
        -------
        tuple
            Predicted expression matrix and auxiliary tensors required by
            CAPRA's operator regularization term.
        """
        self._validate_forward_inputs(
            control_mean=control_mean,
            control_std=control_std,
            single_prior_deltas=single_prior_deltas,
            single_prior_stats=single_prior_stats,
            gene_embeddings=gene_embeddings,
            gene_mask=gene_mask,
            features=features,
            prediction_noise_seed=prediction_noise_seed,
        )
        batch_size = control_mean.shape[0]
        gene_mask = gene_mask.to(dtype=gene_embeddings.dtype)
        control_input = torch.cat([control_mean, control_std], dim=1)
        gene_mask_3d = gene_mask.unsqueeze(-1)
        expanded_features = features.unsqueeze(1).expand(-1, NUM_PERTURBATION_SLOTS, -1)
        prior_latents = self.prior_delta_encoder(
            single_prior_deltas.reshape(batch_size * NUM_PERTURBATION_SLOTS, -1)
        ).reshape(batch_size, NUM_PERTURBATION_SLOTS, -1)
        gene_inputs = torch.cat(
            [gene_embeddings, prior_latents, single_prior_stats, expanded_features, gene_mask_3d],
            dim=-1,
        )
        gene_latents = self.gene_encoder(gene_inputs)
        z0 = self.control_encoder(control_input)

        operator_params = self.operator_head(gene_latents.reshape(batch_size * NUM_PERTURBATION_SLOTS, -1))
        operator_params = operator_params.reshape(batch_size, NUM_PERTURBATION_SLOTS, -1)
        u, v, singular, bias = self._split_operator_params(operator_params)

        slot_linear = self._apply_slot_operators(z0, u, v, singular) * gene_mask_3d
        slot_bias = bias * gene_mask_3d
        slot_shift = slot_linear + slot_bias
        additive_shift = slot_shift.sum(dim=1)
        anchor_delta = (single_prior_deltas * gene_mask_3d).sum(dim=1)

        double_mask = (gene_mask[:, :1] * gene_mask[:, 1:2]).to(dtype=slot_linear.dtype)
        slot_interaction_ab = self._apply_single_operator(slot_linear[:, 1], u[:, 0], v[:, 0], singular[:, 0]) * double_mask
        slot_interaction_ba = self._apply_single_operator(slot_linear[:, 0], u[:, 1], v[:, 1], singular[:, 1]) * double_mask
        composition_shift = 0.5 * (slot_interaction_ab + slot_interaction_ba)
        composition_scale = F.softplus(self.composition_gain_raw)
        pair_shift = composition_scale * composition_shift

        single_residual_gain = torch.sigmoid(self.single_residual_gain_raw)
        pair_residual_gain = torch.sigmoid(self.pair_residual_gain_raw)
        additive_residual = self.additive_residual_decoder(additive_shift)
        pair_residual = double_mask * self.pair_residual_decoder(pair_shift)
        additive_residual = additive_residual * self.additive_residual_output_scale
        pair_residual = pair_residual * self.pair_residual_output_scale
        additive_residual, pair_residual = apply_sparse_residual_guard(
            additive_residual=additive_residual,
            pair_residual=pair_residual,
            control_mean=control_mean,
            anchor_delta=anchor_delta,
            is_training=self.training,
        )

        prediction = (
            control_mean
            + anchor_delta
            + single_residual_gain * additive_residual
            + pair_residual_gain * pair_residual
        )

        # Inference-time SE-floor variance calibration.  Adds per-gene Gaussian
        # noise so that per-cell std ≥ _se_floor_mult × median(ctrl_std of
        # expressed genes).  This prevents standard-error collapse for genes with
        # near-zero imputed variance, making Welch t-test rankings more accurate.
        calibration_noise_scale = torch.zeros_like(control_mean)
        if not self.training:
            if prediction_noise_seed is None:
                raise ValueError(
                    "prediction_noise_seed is required for deterministic variance calibration during evaluation"
                )
            calibration_noise_scale = variance_calibration_noise_scale(
                control_mean,
                control_std,
                expressed_threshold=self._se_floor_expressed_threshold,
                se_floor_mult=self._se_floor_mult,
                noise_scale=self._se_floor_noise_scale,
            )
            eps_add = deterministic_standard_normal_like(
                control_mean, prediction_noise_seed, salt=0,
            )
            eps_pair = deterministic_standard_normal_like(
                control_mean, prediction_noise_seed, salt=1,
            )
            noise_components = eps_add + double_mask * eps_pair
            noise_norm = torch.sqrt(1.0 + double_mask).to(dtype=control_mean.dtype)
            prediction = prediction + calibration_noise_scale * noise_components / noise_norm

        operator_energy = (
            (u.pow(2).mean(dim=(2, 3)) + v.pow(2).mean(dim=(2, 3)) + singular.pow(2).mean(dim=2))
            * gene_mask
        )
        operator_energy = operator_energy.sum(dim=1) / gene_mask.sum(dim=1).clamp_min(1.0)

        aux = {
            "operator_energy": operator_energy,
            "variance_calibration_noise_scale": calibration_noise_scale.detach(),
        }
        return prediction, aux
