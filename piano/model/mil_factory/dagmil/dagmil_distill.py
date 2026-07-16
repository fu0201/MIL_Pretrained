"""DAGMILDistill — DeformableGraphGNN adapted for feature distillation.

DeformableGraphGNN.forward(x, coords) returns raw logits (a plain tensor).
This wrapper:
  - accepts the standard distillation input dict {"features": [B,N,D], "coords": [B,N,2]}
  - replaces self.fc with nn.Identity so the readout embedding [dim_hidden] is
    returned as the slide feature instead of class logits
  - processes each slide in the batch independently (the base forward is B=1 only
    due to AttentionalAggregation's squeeze(0))
  - exposes output_dict["features"] → [B, dim_hidden] for the DDP training loop
"""

import torch
import torch.nn as nn

from .dagmil import DeformableGraphGNN


class DAGMILDistill(DeformableGraphGNN):
    """DeformableGraphGNN with a distillation-friendly interface.

    Parameters
    ----------
    dim_in : int
        Input patch-feature dimension.
    dim_hidden : int
        Internal GNN hidden dimension; also the distilled slide-feature dimension.
    num_classes : int
        Ignored — self.fc is always replaced with nn.Identity for distillation.
    topk : int
        Number of deformable neighbours per patch.
    stride : int
        Coordinate scale factor for offset computation (should match patch size).
    agg_type : str
        GNN aggregation type: 'bi-interaction' | 'gcn' | 'sage'.
    dropout : float
        Accepted for API compatibility; the base class uses a fixed 0.3 dropout.
    """

    def __init__(
        self,
        dim_in: int = 1024,
        dim_hidden: int = 256,
        num_classes: int = 0,
        topk: int = 6,
        stride: int = 512,
        agg_type: str = "bi-interaction",
        dropout: float = 0.25,
        **_extra,
    ):
        super().__init__(
            dim_in=dim_in,
            dim_hidden=dim_hidden,
            n_classes=max(num_classes, 2),  # parent requires >= 1 output
            topk=topk,
            stride=stride,
            agg_type=agg_type,
        )
        # Replace the classification head so the network returns the slide
        # embedding (shape [dim_hidden]) rather than class logits.
        self.fc = nn.Identity()

        # offset_net and expand_alpha have no differentiable gradient path:
        # their output feeds into argmin() which is non-differentiable, so
        # gradients are permanently blocked. Freeze them to avoid DDP errors
        # (unused-parameter detection) and wasted optimizer steps.
        for p in self.offset_net.parameters():
            p.requires_grad_(False)
        self.expand_alpha.requires_grad_(False)

    # ------------------------------------------------------------------
    def forward(self, input_dict: dict, return_loss: bool = True) -> dict:
        """Process a batched input dict and return slide-level features.

        Parameters
        ----------
        input_dict : dict
            "features" → [B, N, D_in] or [N, D_in]
            "coords"   → [B, N, 2]  (real WSI patch coordinates, required)
        return_loss : bool
            Ignored; kept for API compatibility.

        Returns
        -------
        dict with keys:
            "features" : [B, dim_hidden]
            "logits"   : [B, dim_hidden]  (same tensor, Identity head)
            "loss"     : None
        """
        features = input_dict["features"]   # [B, N, D] or [N, D]
        coords   = input_dict.get("coords", None)

        if features.ndim == 2:
            features = features.unsqueeze(0)
            if coords is not None and coords.ndim == 2:
                coords = coords.unsqueeze(0)

        B = features.shape[0]
        slide_feats = []

        for i in range(B):
            if coords is not None:
                coords_i = coords[i].float()            # [N, 2]
            else:
                # Fallback only used during infer_student_embed_dim probing.
                coords_i = torch.zeros(
                    features.shape[1], 2,
                    device=features.device, dtype=features.dtype,
                )

            # Parent forward expects (x:[1,N,D], coords:[1,N,2]) and returns
            # self.fc(h) — which is now nn.Identity, so shape = [dim_hidden]
            h = DeformableGraphGNN.forward(
                self,
                features[i].unsqueeze(0),    # [1, N, D]
                coords_i.unsqueeze(0),        # [1, N, 2]
            )
            # AttentionalAggregation returns [1, dim_hidden]; flatten to [dim_hidden]
            slide_feats.append(h.view(-1))   # [dim_hidden]

        slide_features = torch.stack(slide_feats, dim=0)   # [B, dim_hidden]

        return {
            "features": slide_features,
            "logits":   slide_features,   # Identity placeholder
            "loss":     None,
        }
