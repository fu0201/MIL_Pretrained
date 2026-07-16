"""GDFMILDistill — GDF_MIL adapted for feature distillation.

GDF_MIL.forward(X) takes a raw tensor and returns a dict.  This wrapper:
  - accepts the standard distillation input dict {"features": [B, N, D_in]}
  - replaces self.classifier with nn.Identity so the fused bag embedding
    (shape [1, out_dim]) is returned as the slide feature
  - processes each slide in the batch independently (BagPartition does
    squeeze(0) so it requires a single-sample call)
  - exposes output_dict["features"] → [B, out_dim] for the DDP training loop
"""

import torch
import torch.nn as nn

from .gdfmil import GDF_MIL


class GDFMILDistill(GDF_MIL):
    """GDF_MIL with a distillation-friendly interface.

    Parameters
    ----------
    dim_in : int
        Input patch-feature dimension.
    hid_dim : int
        Encoder hidden dimension.
    out_dim : int
        GNN output / slide-feature dimension.
    k_components : int
        Number of bag-partition clusters.
    k_neighbors : int
        k-NN for the dynamic graph.
    num_classes : int
        Ignored — self.classifier is always replaced with nn.Identity.
    dropout : float
        Dropout in the attention module.
    lambda_smooth : float
        Smoothness regularisation weight (not used during distillation).
    lambda_nce : float
        NCE loss weight (not used during distillation).
    act : str
        Activation function name.
    """

    def __init__(
        self,
        dim_in: int = 1024,
        hid_dim: int = 256,
        out_dim: int = 128,
        k_components: int = 10,
        k_neighbors: int = 10,
        num_classes: int = 0,
        dropout: float = 0.1,
        lambda_smooth: float = 0.0,
        lambda_nce: float = 0.0,
        act: str = "leaky_relu",
        **_extra,
    ):
        super().__init__(
            in_dim=dim_in,
            num_classes=max(num_classes, 2),  # parent Linear requires >= 1 out
            hid_dim=hid_dim,
            out_dim=out_dim,
            k_components=k_components,
            k_neighbors=k_neighbors,
            dropout=dropout,
            lambda_smooth=lambda_smooth,
            lambda_nce=lambda_nce,
            act=act,
        )
        # Replace the classification Sequential with Identity so the fused bag
        # embedding is returned as-is from the forward pass.
        self.classifier = nn.Identity()

    # ------------------------------------------------------------------
    def forward(self, input_dict: dict, return_loss: bool = True) -> dict:
        """Process a batched input dict and return slide-level features.

        Parameters
        ----------
        input_dict : dict
            "features" → [B, N, D_in] or [N, D_in]
            coords are not used by GDF_MIL and are silently ignored.
        return_loss : bool
            Ignored; kept for API compatibility.

        Returns
        -------
        dict with keys:
            "features" : [B, out_dim]
            "logits"   : [B, out_dim]   (same tensor, Identity head)
            "loss"     : None
        """
        features = input_dict["features"]   # [B, N, D_in] or [N, D_in]

        if features.ndim == 2:
            features = features.unsqueeze(0)

        B = features.shape[0]
        slide_feats = []

        for i in range(B):
            # GDF_MIL.forward takes a raw tensor (not a dict); BagPartition
            # internally does squeeze(0), so each call must be single-sample.
            out = GDF_MIL.forward(
                self,
                features[i],        # [N, D_in]
                return_WSI_feature=True,
            )
            # With self.classifier = Identity, out["logits"] = b (the fused bag
            # embedding) of shape [1, out_dim].  WSI_feature == b as well.
            slide_feats.append(out["WSI_feature"])   # [1, out_dim] or [out_dim]

        # Normalise shapes: WSI_feature may be [1, out_dim] or [out_dim]
        slide_features = torch.cat(
            [f.view(1, -1) for f in slide_feats], dim=0
        )   # [B, out_dim]

        return {
            "features": slide_features,
            "logits":   slide_features,
            "loss":     None,
        }
