"""MambaMIL2DDistill — MambaMIL_2D adapted for feature distillation.

Key differences from the base class:
  - accepts the standard distillation input dict {"features": [B, N, D_in]}
    coords are optional: when absent, sequential row-major grid coords are
    generated automatically so the model remains functional without spatial data
  - processes each bag in the batch independently (the base forward is single-bag)
  - exposes output_dict["features"] → [B, 128] slide embeddings expected by
    infer_student_embed_dim and the DDP training loop
  - initialised with num_classes=0, which already sets self.classifier to
    nn.Identity in the parent
"""

import math

import torch
import torch.nn.functional as F

from .mamba_2d import MambaMIL_2D


def _make_sequential_coords(n_patches: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return [n_patches, 2] integer grid coords in row-major order.

    The grid side length is ceil(sqrt(n_patches)) so every patch gets a unique
    (row, col) cell.  Used as a fallback when no real spatial coords are
    available during distillation.
    """
    side = math.ceil(math.sqrt(n_patches))
    idx  = torch.arange(n_patches, device=device)
    rows = idx // side
    cols = idx %  side
    return torch.stack([rows, cols], dim=1).to(dtype)


class MambaMIL2DDistill(MambaMIL_2D):
    """MambaMIL_2D with a distillation-friendly interface.

    Parameters
    ----------
    dim_in : int
        Input patch-feature dimension (e.g. 768 for CONCH).
    drop_out : float
        Dropout probability.
    num_classes : int
        Ignored — always forced to 0 so the parent sets self.classifier to
        nn.Identity.
    survival : bool
        Passed to parent (typically False for distillation).
    pos_emb_type : str or None
        Positional embedding type; None disables it.
    """

    def __init__(
        self,
        dim_in: int = 1024,
        drop_out: float = 0.25,
        num_classes: int = 0,
        survival: bool = False,
        pos_emb_type=None,
        dropout: float | None = None,   # alias accepted from shared student_kwargs
        **_extra,
    ):
        super().__init__(
            dim_in=dim_in,
            drop_out=dropout if dropout is not None else drop_out,
            num_classes=0,          # forces self.classifier = nn.Identity()
            survival=survival,
            pos_emb_type=pos_emb_type,
        )

    # ------------------------------------------------------------------
    # Single-bag helper (mirrors parent forward, exposes slide feature h)
    # ------------------------------------------------------------------
    def _forward_one_bag(
        self,
        features: torch.Tensor,     # [N, D_in]  (no batch dim)
        coords: torch.Tensor | None = None,  # [N, 2] or None
    ) -> torch.Tensor:
        """Run one slide through the network; return slide embedding [1, 128]."""
        h = features.unsqueeze(0)   # [1, N, D_in]

        if coords is None:
            # Fallback used only during infer_student_embed_dim probing (no label,
            # no real coords).  Real training always passes coords from the dataset.
            coords = _make_sequential_coords(features.shape[0], features.device, h.dtype)
        else:
            coords = coords.to(h.dtype)

        h = self._fc1(h)            # [1, N, 128]

        if self.pos_emb_type == "linear":
            pos_embs = self.pos_embs(coords)
            h = h + pos_embs.unsqueeze(0)
            h = self.pos_emb_dropout(h)

        h = self.layers(h, coords, self.pos_embs)   # [1, N, 128]
        h = self.norm(h)                            # [1, N, 128]
        A = self.attention(h)                       # [1, N, 1]

        if A.ndim == 3:
            A = A.transpose(1, 2)                   # [1, 1, N]
        else:
            A = A.permute(0, 3, 1, 2)
            A = A.view(1, 1, -1)
            h = h.view(1, -1, self.config.d_model)

        A = F.softmax(A, dim=-1)                    # [1, 1, N]
        h = torch.bmm(A, h)                         # [1, 1, 128]
        h = h.squeeze(0)                            # [1, 128]
        return h

    # ------------------------------------------------------------------
    # Distillation forward
    # ------------------------------------------------------------------
    def forward(self, input_dict: dict, return_loss: bool = True) -> dict:
        """Process a batched input dict and return slide-level features.

        Parameters
        ----------
        input_dict : dict
            Must contain "features" with shape [B, N, D_in] or [N, D_in].
            Optionally contains "coords" with shape [B, N, 2] or [N, 2].
        return_loss : bool
            Ignored; kept for API compatibility.

        Returns
        -------
        dict with keys:
            "features" : [B, 128]  slide embeddings
            "logits"   : [B, 128]  same tensor (identity head)
            "loss"     : None
        """
        features = input_dict["features"]    # [B, N, D_in] or [N, D_in]
        coords   = input_dict.get("coords", None)

        if features.ndim == 2:
            features = features.unsqueeze(0) # [1, N, D_in]
            if coords is not None and coords.ndim == 2:
                coords = coords.unsqueeze(0) # [1, N, 2]

        B = features.shape[0]
        slide_feats = []
        for i in range(B):
            coords_i = coords[i] if coords is not None else None
            h = self._forward_one_bag(features[i], coords_i)  # [1, 128]
            slide_feats.append(h)

        slide_features = torch.cat(slide_feats, dim=0)         # [B, 128]
        logits = self.classifier(slide_features)               # [B, 128] via Identity

        return {
            "features": slide_features,
            "logits":   logits,
            "loss":     None,
        }
