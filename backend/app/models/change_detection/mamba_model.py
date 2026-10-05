"""Compact selective state-space model for multi-temporal change segmentation.

This implements the input-dependent selective recurrence from Mamba in plain
PyTorch so it runs on CPU-only installations. The official fused CUDA scan can
replace the reference scan later without changing the temporal model contract.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SelectiveTemporalSSM(nn.Module):
    """Mamba-style selective scan over each spatial location's time sequence."""

    def __init__(self, model_dim: int, state_dim: int = 8, conv_width: int = 3):
        super().__init__()
        self.model_dim = model_dim
        self.state_dim = state_dim
        self.inner_dim = model_dim * 2
        self.dt_rank = max(1, math.ceil(model_dim / 16))

        self.in_proj = nn.Linear(model_dim, self.inner_dim * 2)
        self.temporal_conv = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=conv_width,
            groups=self.inner_dim,
            padding=0,
        )
        self.parameter_proj = nn.Linear(self.inner_dim, self.dt_rank + state_dim * 2)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, model_dim)

        state_rates = torch.arange(1, state_dim + 1, dtype=torch.float32)
        self.log_state_rates = nn.Parameter(state_rates.log().repeat(self.inner_dim, 1))
        self.skip = nn.Parameter(torch.ones(self.inner_dim))

        # Keep initial step sizes in the stable range used by selective SSMs.
        nn.init.uniform_(self.dt_proj.weight, -self.dt_rank ** -0.5, self.dt_rank ** -0.5)
        initial_dt = torch.exp(
            torch.rand(self.inner_dim) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inverse_softplus = initial_dt + torch.log(-torch.expm1(-initial_dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inverse_softplus)

    def forward(
        self,
        features: Tensor,
        initial_state: Optional[Tensor] = None,
        initial_conv_state: Optional[Tensor] = None,
    ) -> tuple[Tensor, list[Tensor], list[Tensor], Tensor, Tensor]:
        """Return temporal features, per-date states, and the final state.

        Args:
            features: (batch, time, channels, height, width).
            initial_state: optional (batch, inner, state, height, width).
        """
        batch, timesteps, channels, height, width = features.shape
        positions = height * width
        sequence = features.permute(0, 3, 4, 1, 2).reshape(batch * positions, timesteps, channels)
        projected = self.in_proj(sequence)
        value, gate = projected.chunk(2, dim=-1)

        conv_context_length = self.temporal_conv.kernel_size[0] - 1
        value_for_conv = value.transpose(1, 2)
        if initial_conv_state is None:
            conv_context = value_for_conv.new_zeros((batch * positions, self.inner_dim, conv_context_length))
        else:
            expected_conv_shape = (batch, self.inner_dim, conv_context_length, height, width)
            if initial_conv_state.shape != expected_conv_shape:
                raise ValueError(
                    "Mamba convolution state shape does not match input tile: "
                    f"{tuple(initial_conv_state.shape)} vs {expected_conv_shape}"
                )
            conv_context = initial_conv_state.permute(0, 3, 4, 1, 2).reshape(
                batch * positions, self.inner_dim, conv_context_length
            ).to(dtype=features.dtype, device=features.device)
        joined_conv_input = torch.cat((conv_context, value_for_conv), dim=-1)
        convolved = self.temporal_conv(joined_conv_input).transpose(1, 2)
        value = F.silu(convolved)

        dynamic = self.parameter_proj(value)
        dt_low, input_state, output_state = torch.split(
            dynamic, [self.dt_rank, self.state_dim, self.state_dim], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt_low))
        state_rates = -torch.exp(self.log_state_rates).to(dtype=features.dtype)

        if initial_state is None:
            state = features.new_zeros((batch * positions, self.inner_dim, self.state_dim))
        else:
            if initial_state.shape != (batch, self.inner_dim, self.state_dim, height, width):
                raise ValueError(
                    "Mamba state shape does not match input tile: "
                    f"{tuple(initial_state.shape)} vs "
                    f"{(batch, self.inner_dim, self.state_dim, height, width)}"
                )
            state = initial_state.permute(0, 3, 4, 1, 2).reshape(
                batch * positions, self.inner_dim, self.state_dim
            ).to(dtype=features.dtype, device=features.device)

        outputs: list[Tensor] = []
        snapshots: list[Tensor] = []
        conv_snapshots: list[Tensor] = []
        for time_index in range(timesteps):
            step = delta[:, time_index]
            input_vector = value[:, time_index]
            select_in = input_state[:, time_index]
            select_out = output_state[:, time_index]

            decay = torch.exp(step.unsqueeze(-1) * state_rates.unsqueeze(0))
            state = decay * state + (
                step.unsqueeze(-1)
                * select_in.unsqueeze(1)
                * input_vector.unsqueeze(-1)
            )
            response = (state * select_out.unsqueeze(1)).sum(dim=-1)
            response = response + self.skip * input_vector
            response = self.out_proj(response * F.silu(gate[:, time_index]))
            outputs.append(response.reshape(batch, height, width, channels).permute(0, 3, 1, 2))
            snapshots.append(
                state.reshape(batch, height, width, self.inner_dim, self.state_dim)
                .permute(0, 3, 4, 1, 2)
                .contiguous()
            )
            conv_end = time_index + 1 + conv_context_length
            conv_snapshots.append(
                joined_conv_input[:, :, conv_end - conv_context_length:conv_end]
                .reshape(batch, height, width, self.inner_dim, conv_context_length)
                .permute(0, 3, 4, 1, 2)
                .contiguous()
            )

        temporal_features = torch.stack(outputs, dim=1)
        final_state = snapshots[-1]
        final_conv_state = conv_snapshots[-1]
        return temporal_features, snapshots, conv_snapshots, final_state, final_conv_state


class MambaTemporalChangeNet(nn.Module):
    """Spatial selective scans, temporal SSM, and a binary change decoder."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 32, state_dim: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, feature_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, feature_dim),
            nn.SiLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, feature_dim),
            nn.SiLU(),
        )
        self.spatial_ssm = SelectiveTemporalSSM(feature_dim, state_dim)
        self.temporal_ssm = SelectiveTemporalSSM(feature_dim, state_dim)
        pair_channels = feature_dim * 6
        self.change_decoder = nn.Sequential(
            nn.Conv2d(pair_channels, feature_dim * 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, feature_dim * 2),
            nn.SiLU(),
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, feature_dim),
            nn.SiLU(),
            nn.Conv2d(feature_dim, 1, kernel_size=1),
        )

    def _bidirectional_scan(self, sequence: Tensor) -> Tensor:
        """Scan a batch of 1D spatial rows/columns in both directions."""
        forward_input = sequence.unsqueeze(-1).unsqueeze(-1)
        forward = self.spatial_ssm(forward_input)[0][:, :, :, 0, 0]
        reverse = self.spatial_ssm(forward_input.flip(1))[0][:, :, :, 0, 0].flip(1)
        return 0.5 * (forward + reverse)

    def forward(
        self,
        images: Tensor,
        initial_state: Optional[Tensor] = None,
        initial_conv_state: Optional[Tensor] = None,
        previous_spatial: Optional[Tensor] = None,
        previous_temporal: Optional[Tensor] = None,
        return_context: bool = False,
    ) -> Tensor | tuple[
        Tensor,
        list[Tensor],
        list[Tensor],
        list[Tensor],
        list[Tensor],
    ]:
        """Predict a binary change mask from T co-registered dates.

        `previous_*` plus `initial_state` let incremental inference continue from
        a persisted temporal state and classify the newest date against the last
        processed acquisition.
        """
        if images.ndim != 5:
            raise ValueError("Expected images shaped (batch, time, channels, height, width)")
        batch, timesteps, channels, height, width = images.shape
        if timesteps < 1 or channels != self.in_channels:
            raise ValueError(f"Expected at least one date and {self.in_channels} channels")

        flattened = images.reshape(batch * timesteps, channels, height, width)
        spatial = self.spatial_encoder(flattened)
        feat_h, feat_w = spatial.shape[-2:]
        spatial = spatial.reshape(batch, timesteps, self.feature_dim, feat_h, feat_w)
        horizontal_rows = spatial.permute(0, 1, 3, 4, 2).reshape(
            batch * timesteps * feat_h, feat_w, self.feature_dim
        )
        vertical_columns = spatial.permute(0, 1, 4, 3, 2).reshape(
            batch * timesteps * feat_w, feat_h, self.feature_dim
        )
        horizontal = self._bidirectional_scan(horizontal_rows).reshape(
            batch, timesteps, feat_h, feat_w, self.feature_dim
        ).permute(0, 1, 4, 2, 3)
        vertical = self._bidirectional_scan(vertical_columns).reshape(
            batch, timesteps, feat_w, feat_h, self.feature_dim
        ).permute(0, 1, 4, 3, 2)
        spatial = spatial + 0.5 * (horizontal + vertical)
        temporal, states, conv_states, _, _ = self.temporal_ssm(
            spatial,
            initial_state=initial_state,
            initial_conv_state=initial_conv_state,
        )

        if previous_spatial is not None or previous_temporal is not None:
            if previous_spatial is None or previous_temporal is None:
                raise ValueError("Incremental inference requires both previous feature maps")
            expected_shape = (batch, self.feature_dim, feat_h, feat_w)
            if previous_spatial.shape != expected_shape or previous_temporal.shape != expected_shape:
                raise ValueError("Persisted Mamba feature maps do not match the current tile geometry")
            spatial_before = previous_spatial.to(device=images.device, dtype=spatial.dtype)
            temporal_before = previous_temporal.to(device=images.device, dtype=temporal.dtype)
            spatial_after = spatial[:, -1]
            temporal_after = temporal[:, -1]
        else:
            if timesteps < 2:
                raise ValueError("At least two dates are required without a persisted prior state")
            spatial_before = spatial[:, 0]
            temporal_before = temporal[:, 0]
            spatial_after = spatial[:, -1]
            temporal_after = temporal[:, -1]

        pair_features = torch.cat(
            (
                spatial_before,
                spatial_after,
                torch.abs(spatial_after - spatial_before),
                temporal_before,
                temporal_after,
                torch.abs(temporal_after - temporal_before),
            ),
            dim=1,
        )
        logits = self.change_decoder(pair_features)
        logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        if return_context:
            return (
                logits,
                states,
                conv_states,
                [item for item in spatial.unbind(dim=1)],
                [item for item in temporal.unbind(dim=1)],
            )
        return logits
