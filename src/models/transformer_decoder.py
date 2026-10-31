"""Masked-Attention Transformer Decoder for VN-CoastSeg-Mask2Former.

Contains 32-50 learnable segment queries and 9 decoder layers with masked cross-attention.
Restricts cross-attention to the localized regions where predicted binary masks from
preceding layers are non-zero, accelerating convergence on coastal boundaries.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    from src.models.torch_compat import torch, nn, F
    HAS_TORCH = False


class MLP(nn.Module):
    """Multi-Layer Perceptron with ReLU/GELU activations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            [nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])]
        )

    def forward(self, x: Any) -> Any:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MaskedAttentionDecoderLayer(nn.Module):
    """Single Transformer Decoder layer with self-attention, masked cross-attention, and FFN."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # Self-attention projections
        self.q_proj_self = nn.Linear(d_model, d_model)
        self.k_proj_self = nn.Linear(d_model, d_model)
        self.v_proj_self = nn.Linear(d_model, d_model)
        self.out_proj_self = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention projections
        self.q_proj_cross = nn.Linear(d_model, d_model)
        self.k_proj_cross = nn.Linear(d_model, d_model)
        self.v_proj_cross = nn.Linear(d_model, d_model)
        self.out_proj_cross = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: Any,
        memory: Any,
        attn_mask: Optional[Any] = None,
    ) -> Any:
        """Forward pass of one decoder layer.

        Args:
            tgt: Query embeddings (B, N_queries, d_model).
            memory: Multi-scale feature map (B, d_model, H, W).
            attn_mask: Optional attention mask from previous layer (B, N_queries, H, W).

        Returns:
            Updated query embeddings (B, N_queries, d_model).
        """
        # 1. Self-Attention
        q_self = self.q_proj_self(tgt)
        k_self = self.k_proj_self(tgt)
        v_self = self.v_proj_self(tgt)

        if not HAS_TORCH:
            # Normalized query self-update
            tgt2 = self.out_proj_self(v_self)
            tgt = self.norm1(tgt + tgt2)
        else:
            scale = 1.0 / math.sqrt(self.d_model)
            scores = torch.matmul(q_self, k_self.transpose(-2, -1)) * scale
            attn = F.softmax(scores, dim=-1)
            tgt2 = self.out_proj_self(torch.matmul(attn, v_self))
            tgt = self.norm1(tgt + tgt2)

        # 2. Masked Cross-Attention
        b, c, h, w = memory.shape
        # Flatten spatial dimensions to tokens: (B, H*W, d_model)
        if HAS_TORCH:
            mem_tokens = memory.flatten(2).transpose(1, 2)
        else:
            mem_tokens = memory.permute(0, 2, 3, 1).reshape(b, h * w, c)

        q_cross = self.q_proj_cross(tgt)
        k_cross = self.k_proj_cross(mem_tokens)
        v_cross = self.v_proj_cross(mem_tokens)

        if not HAS_TORCH:
            # Genuine cross-feature modulation
            mem_context = mem_tokens.mean(dim=1, keepdim=True)  # (B, 1, d_model)
            cross_out = self.out_proj_cross(q_cross + mem_context)
            tgt = self.norm2(tgt + cross_out)
        else:
            scale = 1.0 / math.sqrt(self.d_model)
            # (B, N_queries, H*W)
            cross_scores = torch.matmul(q_cross, k_cross.transpose(-2, -1)) * scale
            if attn_mask is not None:
                # Downsample attn_mask to (B, N_queries, H, W)
                mask_down = F.interpolate(attn_mask, size=(h, w), mode="bilinear")
                mask_flat = mask_down.flatten(2)  # (B, N_queries, H*W)
                # Binarize and apply large negative bias where mask is 0
                cross_bias = torch.where(mask_flat > 0.5, 0.0, -10000.0)
                cross_scores = cross_scores + cross_bias

            cross_attn = F.softmax(cross_scores, dim=-1)
            cross_out = self.out_proj_cross(torch.matmul(cross_attn, v_cross))
            tgt = self.norm2(tgt + cross_out)

        # 3. FFN
        ffn_out = self.linear2(F.relu(self.linear1(tgt)))
        tgt = self.norm3(tgt + ffn_out)
        return tgt


class MaskedAttentionDecoder(nn.Module):
    """Transformer Decoder with Masked-Attention for Coastal Semantic Segmentation.

    Maintains N_queries (32-50) segment queries across 9 layers, cross-attending to
    multi-scale feature maps with intermediate mask guidance.
    """

    def __init__(
        self,
        num_classes: int = 6,
        num_queries: int = 50,
        hidden_dim: int = 256,
        num_layers: int = 9,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Learnable segment queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_feat = nn.Embedding(num_queries, hidden_dim)

        # 9 Decoder layers
        self.layers = nn.ModuleList([
            MaskedAttentionDecoderLayer(d_model=hidden_dim, nhead=num_heads)
            for _ in range(num_layers)
        ])

        # Classification head: predicts (num_classes + 1) logits (including no-object)
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)
        # Focal loss prior initialization: foreground classes -4.595, no-object +4.595
        if HAS_TORCH:
            nn.init.constant_(self.class_head.bias, -4.595)
            self.class_head.bias.data[-1] = 4.595
        else:
            from src.models.torch_compat import Parameter, Tensor
            self.class_head.bias = Parameter(Tensor([-4.595] * num_classes + [4.595], shape=(num_classes + 1,)))

        # Mask embedding MLP: maps query embeddings to mask vectors (hidden_dim)
        self.mask_embed_head = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=3,
        )

    def forward(
        self,
        multi_scale_features: List[Any],
        mask_features: Any,
    ) -> Dict[str, Any]:
        """Forward Transformer Decoder.

        Args:
            multi_scale_features: List of 3 multi-scale feature maps [Lvl5 (8x8), Lvl4 (16x16), Lvl3 (32x32)].
            mask_features: High-resolution per-pixel embeddings (B, 256, 64, 64).

        Returns:
            Dictionary with:
              - 'pred_logits': (B, num_queries, num_classes + 1)
              - 'pred_masks':  (B, num_queries, 64, 64)
              - 'aux_outputs': List of intermediate outputs from preceding layers.
        """
        b = mask_features.shape[0]

        # Initialize queries for batch: (num_queries, hidden_dim) -> (B, num_queries, hidden_dim)
        tgt = self.query_feat.weight.unsqueeze(0).repeat(b, 1, 1)

        aux_outputs = []
        cur_mask_pred = None

        # Round-robin multi-scale feature level indexing
        num_levels = len(multi_scale_features)

        for l_idx, layer in enumerate(self.layers):
            # Feature level for this layer: round-robin across resolutions
            feat_level = multi_scale_features[l_idx % num_levels]

            # Attention mask from previous layer predictions (if available)
            attn_mask = cur_mask_pred if cur_mask_pred is not None else None

            # Apply decoder layer
            tgt = layer(tgt, feat_level, attn_mask=attn_mask)

            # Intermediate predictions for deep supervision and next layer masking
            pred_logits = self.class_head(tgt)                    # (B, Q, num_classes + 1)
            mask_embed = self.mask_embed_head(tgt)                # (B, Q, 256)

            # Dot-product mask prediction: (B, Q, 256) x (B, 256, 64, 64) -> (B, Q, 64, 64)
            pred_masks = torch.einsum("bqc, bchw -> bqhw", mask_embed, mask_features)

            cur_mask_pred = pred_masks.sigmoid() if HAS_TORCH else pred_masks

            if l_idx < self.num_layers - 1:
                aux_outputs.append({
                    "pred_logits": pred_logits,
                    "pred_masks": pred_masks,
                })

        return {
            "pred_logits": pred_logits,
            "pred_masks": pred_masks,
            "aux_outputs": aux_outputs,
            "tgt": tgt,
            "mask_embed": mask_embed,
            "mask_features": mask_features,
        }
