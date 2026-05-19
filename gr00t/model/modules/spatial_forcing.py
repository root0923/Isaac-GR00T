# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spatial Forcing module for aligning VLA visual embeddings with VGGT 3D features.

Based on: "Spatial Forcing: Implicit Spatial Representation Alignment for
Vision-Language-Action Model" (Li et al., ICLR 2026).
"""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pooling utilities (adapted from VGGT heads/utils.py)
# ---------------------------------------------------------------------------

# Cosmos-Reason2-2B (Qwen3-VL) image normalization, from preprocessor_config.json
# Not from dataset stats — dataset statistics are only used for state/action normalization.
_QWEN3VL_MEAN = [0.5, 0.5, 0.5]
_QWEN3VL_STD = [0.5, 0.5, 0.5]


def _make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    assert embed_dim % 2 == 0
    device = pos.device
    omega = torch.arange(
        embed_dim // 2,
        dtype=torch.float32 if device.type == "mps" else torch.double,
        device=device,
    )
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega

    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    return emb.float()


def _position_grid_to_embed(
    pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100
) -> torch.Tensor:
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)
    emb_x = _make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)
    emb_y = _make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)
    emb = torch.cat([emb_x, emb_y], dim=-1)
    return emb.view(H, W, embed_dim)


def _create_uv_grid(
    width: int,
    height: int,
    aspect_ratio: float | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)

    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    return torch.stack((uu, vv), dim=-1)


def _safe_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] | None = None,
    scale_factor: float | None = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736
    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners)
            for chunk in chunks
        ]
        return torch.cat(interpolated_chunks, dim=0).contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)


def _apply_pos_embed(x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
    patch_w = x.shape[-1]
    patch_h = x.shape[-2]
    pos_embed = _create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
    pos_embed = _position_grid_to_embed(pos_embed, x.shape[1])
    pos_embed = pos_embed * ratio
    pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
    return x + pos_embed


def _interpolate_pooling(hidden, patch_hw, img_hw, reference, pooling_func, use_vggt_pe):
    (patch_h, patch_w) = patch_hw
    (img_h, img_w) = img_hw
    bs, N, S, D = hidden.shape
    re_sample_ratio = 1 / np.sqrt(N * S / reference.shape[1])

    _hidden = hidden.permute(0, 1, 3, 2)
    _hidden = _hidden.reshape(bs * N, D, patch_h, patch_w)
    if use_vggt_pe:
        _hidden = _apply_pos_embed(_hidden, img_w, img_h)
    hidden_pooling = _safe_interpolate(
        _hidden, scale_factor=re_sample_ratio, mode=pooling_func, align_corners=True
    )
    hidden_pooling = hidden_pooling.reshape(bs, N, D, -1).permute(0, 1, 3, 2).reshape(bs, -1, D)
    return hidden_pooling


def custom_pooling(hidden, patch_hw, img_hw, reference, pooling_func, use_vggt_pe):
    if pooling_func in ["bilinear"]:
        return _interpolate_pooling(hidden, patch_hw, img_hw, reference, pooling_func, use_vggt_pe)
    else:
        raise NotImplementedError(f"Pooling function {pooling_func} is not implemented.")


# ---------------------------------------------------------------------------
# Image preprocessing for VGGT
# ---------------------------------------------------------------------------


def preprocess_images_for_vggt(
    pixel_values: torch.Tensor,
    target_size: int = 518,
) -> torch.Tensor:
    """Reverse Qwen3VL ImageNet normalization and resize for VGGT input.

    Args:
        pixel_values: [num_images, C, H, W] normalized by Qwen3VLProcessor.
        target_size: Target size for VGGT input (default 518).

    Returns:
        vggt_images: [num_images, 1, 3, target_size, target_size] ready for VGGT.
    """
    mean = torch.tensor(_QWEN3VL_MEAN, device=pixel_values.device).view(1, 3, 1, 1)
    std = torch.tensor(_QWEN3VL_STD, device=pixel_values.device).view(1, 3, 1, 1)

    # Reverse Qwen3VL normalization → [0, 1] range
    unnormed = pixel_values.float() * std + mean
    unnormed = unnormed.clamp(0.0, 1.0)

    # Resize so that the shorter edge = target_size, divisible by 14
    results = []
    for img in unnormed:
        _, H, W = img.shape
        if W >= H:
            new_width = target_size
            new_height = round(H * (new_width / W) / 14) * 14
        else:
            new_height = target_size
            new_width = round(W * (new_height / H) / 14) * 14

        resized = F.interpolate(
            img.unsqueeze(0), size=(new_height, new_width), mode="bicubic", align_corners=False
        )

        # Center crop if height exceeds target_size
        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            resized = resized[:, :, start_y : start_y + target_size, :]
        # Pad if smaller
        elif new_height < target_size or new_width < target_size:
            h_pad = target_size - resized.shape[-2]
            w_pad = target_size - resized.shape[-1]
            pad_top = h_pad // 2
            pad_bottom = h_pad - pad_top
            pad_left = w_pad // 2
            pad_right = w_pad - pad_left
            resized = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=1.0)

        results.append(resized)

    vggt_images = torch.cat(results, dim=0)  # [num_images, 3, target_size, target_size]
    # Add frame dimension: [num_images, 1, 3, H, W]
    vggt_images = vggt_images.unsqueeze(1)
    return vggt_images


# ---------------------------------------------------------------------------
# Align Projector
# ---------------------------------------------------------------------------


class AlignProjector(nn.Module):
    """Project VLA embeddings to VGGT feature dimension and compute alignment loss."""

    def __init__(
        self,
        llm_dim: int,
        vggt_dim: int,
        use_vlm_norm: bool = False,
    ):
        super().__init__()
        self.llm_dim = llm_dim
        self.vggt_dim = vggt_dim

        self.fc1 = nn.Linear(self.llm_dim, 2 * self.vggt_dim, bias=True)
        self.fc2 = nn.Linear(2 * self.vggt_dim, 2 * self.vggt_dim, bias=True)
        self.act_fn1 = nn.GELU()

        self.vlm_norm = nn.LayerNorm(llm_dim) if use_vlm_norm else None
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def align_dimension(self, llm_embedding: torch.Tensor) -> torch.Tensor:
        if self.vlm_norm is not None:
            llm_embedding = self.vlm_norm(llm_embedding)
        projected = self.fc1(llm_embedding)
        projected = self.act_fn1(projected)
        projected = self.fc2(projected)
        return projected

    def compute_align_loss_cosine(self, vision_hidden, vggt_hidden, align_mask=None):
        # vision_hidden: (bs, N, D), vggt_hidden: (bs, N, D)
        def mean_flat(x):
            return torch.mean(x, dim=list(range(1, len(x.size()))))

        align_loss = 0.0
        bsz = vision_hidden.shape[0]

        if align_mask is not None:
            for _vision, _vggt, _mask in zip(vision_hidden, vggt_hidden, align_mask):
                _vision = F.normalize(_vision, dim=-1)
                _vggt = F.normalize(_vggt, dim=-1)
                align_loss += 1 - mean_flat((_vision * _vggt)[_mask].sum(dim=-1))
        else:
            for _vision, _vggt in zip(vision_hidden, vggt_hidden):
                _vision = F.normalize(_vision, dim=-1)
                _vggt = F.normalize(_vggt, dim=-1)
                align_loss += 1 - mean_flat((_vision * _vggt).sum(dim=-1))

        align_loss /= bsz
        return align_loss

    def forward(self, llm_emb, target_emb, align_mask=None):
        llm_emb = self.align_dimension(llm_emb)
        align_loss = self.compute_align_loss_cosine(llm_emb, target_emb, align_mask).mean()
        return align_loss


# ---------------------------------------------------------------------------
# Spatial Forcing Module
# ---------------------------------------------------------------------------


class SpatialForcingModule(nn.Module):
    """Manages the frozen VGGT model and trainable AlignProjector for Spatial Forcing.

    Usage:
        sf = SpatialForcingModule(vggt_path, llm_dim=2048, vggt_dim=1024, ...)
        # During training forward:
        align_loss = sf(backbone_features, image_mask, pixel_values, image_grid_thw)
    """

    def __init__(
        self,
        vggt_path: str,
        llm_dim: int,
        vggt_dim: int = 1024,
        vggt_layers_align: int = -1,
        pooling_func: str = "bilinear",
        use_vggt_pe: bool = False,
        use_vlm_norm: bool = False,
    ):
        super().__init__()

        from gr00t.model.modules.vggt.models.vggt import VGGT

        self.vggt = VGGT(
            enable_camera=False,
            enable_point=False,
            enable_depth=False,
            enable_track=False,
        )
        self.vggt.load_state_dict(torch.load(vggt_path, map_location="cpu"), strict=False)
        self.vggt.eval()
        for p in self.vggt.parameters():
            p.requires_grad = False

        self.vggt_embed_dim = self.vggt.embed_dim
        self.vggt_patch_size = self.vggt.patch_size
        self.vggt_layers_align = vggt_layers_align
        self.pooling_func = pooling_func
        self.use_vggt_pe = use_vggt_pe

        self.align_projector = AlignProjector(
            llm_dim=llm_dim,
            vggt_dim=vggt_dim,
            use_vlm_norm=use_vlm_norm,
        )

    def _extract_vision_hidden(
        self,
        backbone_features: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract image-token features from backbone output.

        Args:
            backbone_features: [B, seq_len, D] full sequence from backbone.
            image_mask: [B, seq_len] bool, True for image tokens.

        Returns:
            vision_hidden: [B, N_img, D] image token features.
            align_mask: [B, N_img] bool, all True (valid image tokens).
        """
        B = backbone_features.shape[0]
        # Count image tokens per sample (should be uniform within a batch)
        n_img = image_mask[0].sum().item()
        vision_hidden = torch.zeros(
            B, n_img, backbone_features.shape[-1],
            device=backbone_features.device,
            dtype=backbone_features.dtype,
        )
        align_mask = torch.ones(B, n_img, dtype=torch.bool, device=backbone_features.device)

        for i in range(B):
            idx = image_mask[i]
            img_feats = backbone_features[i, idx]
            # Handle potential mismatch in token count
            actual_n = img_feats.shape[0]
            if actual_n == n_img:
                vision_hidden[i] = img_feats
            else:
                vision_hidden[i, :actual_n] = img_feats
                align_mask[i, actual_n:] = False

        return vision_hidden, align_mask

    def forward(
        self,
        backbone_features: torch.Tensor,
        image_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute spatial forcing alignment loss.

        Args:
            backbone_features: [B, seq_len, D] from Qwen3Backbone (raw, before vlln).
            image_mask: [B, seq_len] bool indicating image token positions.
            pixel_values: [num_images, C, H, W] normalized pixel values.
            image_grid_thw: [num_images, 3] grid info (unused currently, for future).

        Returns:
            align_loss: scalar tensor.
        """
        # 1. Extract vision hidden states from backbone
        vision_hidden, align_mask = self._extract_vision_hidden(backbone_features, image_mask)

        # 2. Prepare images for VGGT (unnormalize + resize)
        vggt_images = preprocess_images_for_vggt(pixel_values)  # [num_images, 1, 3, H, W]

        # Group images by sample: assume num_images_per_sample = total_images / batch_size
        B = backbone_features.shape[0]
        num_total_images = vggt_images.shape[0]
        num_images_per_sample = num_total_images // B

        if num_total_images % B != 0:
            logger.warning(
                f"Cannot evenly divide {num_total_images} images into {B} samples. "
                f"Falling back to processing all images as batch."
            )
            num_images_per_sample = 1

        # Reshape to [B, S, 1, 3, H, W] then squeeze to [B, S, 3, H, W]
        vggt_images = vggt_images.view(B, num_images_per_sample, 1, *vggt_images.shape[-3:])
        vggt_images = vggt_images.squeeze(2)  # [B, S, 3, H, W]
        vggt_images = vggt_images.to(backbone_features.device)

        # 3. VGGT forward (frozen, no grad)
        with torch.no_grad():
            vggt_output = self.vggt.aggregator(vggt_images)
            agg_vggt_hidden, patch_start_idx = vggt_output

        # Select alignment layer
        vggt_features = agg_vggt_hidden[self.vggt_layers_align]  # [B, S, P, 2C]

        # Remove special tokens (camera + register)
        vggt_hidden = vggt_features[:, :, patch_start_idx:, :]

        # 4. Resample VGGT features to match VLA token count
        # VGGT images shape for spatial info
        orig_img = vggt_images  # [B, S, 3, H, W]
        H, W = orig_img.shape[-2:]
        patch_h = H // self.vggt_patch_size
        patch_w = W // self.vggt_patch_size

        vggt_hidden = custom_pooling(
            vggt_hidden,
            (patch_h, patch_w),
            (H, W),
            vision_hidden,
            self.pooling_func,
            self.use_vggt_pe,
        )

        # 5. Compute alignment loss
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=backbone_features.is_cuda):
            align_loss = self.align_projector(vision_hidden, vggt_hidden, align_mask)

        return align_loss

    def train(self, mode: bool = True):
        super().train(mode)
        # VGGT is always in eval mode
        self.vggt.eval()
        return self
