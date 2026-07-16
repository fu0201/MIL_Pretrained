"""AEMMILDistill — AEM_MIL adapted for feature distillation.

Wraps AEM_MIL so that it:
  - accepts the standard distillation input dict  {"features": [B, N, D_in]}
  - processes each bag in the batch independently (AEM_MIL is inherently single-bag)
  - exposes output_dict["features"] → [B, L] slide embeddings expected by
    infer_student_embed_dim and the DDP training loop
  - replaces the classification head with nn.Identity so no class labels are
    needed during distillation
"""

import torch
import torch.nn as nn

from .aemmil import AEM_MIL


class AEMMILDistill(AEM_MIL):
    """AEM_MIL with a distillation-friendly interface.

    Parameters
    ----------
    dim_in : int
        Input patch-feature dimension (e.g. 768 for CONCH).
    L : int
        Internal feature projection dimension.
    D : int
        Attention hidden dimension.
    num_classes : int
        Ignored (kept for API compatibility); the classifier is always replaced
        with nn.Identity for distillation.
    dropout : float
        Dropout probability passed to the parent.
    temperature : float
        Attention temperature.
    lambda_entropy : float
        Entropy regularisation weight (not used during distillation forward,
        kept for parameter parity with the base class).
    """

    def __init__(
        self,
        dim_in: int = 1024,
        L: int = 512,
        D: int = 128,
        num_classes: int = 0,
        dropout: float = 0.1,
        temperature: float = 1.0,
        lambda_entropy: float = 0.1,
        **_extra,
    ):
        super().__init__(
            in_dim=dim_in,
            L=L,
            D=D,
            num_classes=2,   # placeholder — replaced immediately below
            dropout=dropout,
            act=nn.ReLU(),
            temperature=temperature,
            lambda_entropy=lambda_entropy,
        )
        # Replace the bag-level classifier with an identity so the model has no
        # class-specific parameters during distillation.
        self.classifier = nn.Sequential(nn.Identity())

    # ------------------------------------------------------------------
    # Distillation forward
    # ------------------------------------------------------------------
    def forward(self, input_dict: dict, return_loss: bool = True) -> dict:
        """Process a batched input dict and return slide-level features.

        Parameters
        ----------
        input_dict : dict
            Must contain key "features" with shape [B, N, D_in] or [N, D_in].
        return_loss : bool
            Ignored; kept for API compatibility with the training loop.

        Returns
        -------
        dict with keys:
            "features" : [B, L]   slide embeddings (used by DDP loop)
            "logits"   : [B, L]   same tensor (identity head placeholder)
            "loss"     : None
        """
        features = input_dict["features"]    # [B, N, D_in] or [N, D_in]
        if features.ndim == 2:
            features = features.unsqueeze(0) # [1, N, D_in]

        slide_feats = []
        for i in range(features.shape[0]):
            # AEM_MIL.forward expects a raw tensor x of shape [N, D_in] or [1, N, D_in]
            out = AEM_MIL.forward(
                self,
                features[i],            # [N, D_in]
                return_WSI_feature=True,
                return_WSI_attn=False,
                return_entropy=False,
            )
            slide_feats.append(out["WSI_feature"])  # [L]

        slide_features = torch.stack(slide_feats, dim=0)   # [B, L]

        return {
            "features": slide_features,
            "logits":   slide_features,   # identity placeholder
            "loss":     None,
        }
