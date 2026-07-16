from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedSummaryLoss(nn.Module):
    def __init__(
        self,
        teacher_names,
        embed_dim: int,
        momentum: float = 0.99,
        eps: float = 1e-6,
        reduction: str = "mean",
        detach_teacher: bool = True,
        use_angle: bool = False,
        loss_type: Optional[str] = None,
        min_dispersion: float = 0.05,
        max_dispersion: Optional[float] = None,
    ):
        super().__init__()
        self.teacher_names = list(teacher_names)
        self.embed_dim = embed_dim
        self.momentum = momentum
        self.eps = eps
        self.reduction = reduction
        self.detach_teacher = detach_teacher
        if loss_type is None:
            loss_type = "angle" if use_angle else "cosine"
        if loss_type not in {"angle", "cosine", "mse"}:
            raise ValueError(
                f"Unsupported loss_type: {loss_type}. "
                "Expected one of: 'angle', 'cosine', 'mse'."
            )
        self.loss_type = loss_type
        self.use_angle = self.loss_type == "angle"
        self.min_dispersion = min_dispersion
        self.max_dispersion = max_dispersion

        # Running mean direction per teacher: [T, D]
        center = torch.zeros(len(self.teacher_names), embed_dim)
        center[:, 0] = 1.0  # simple valid init before first update
        self.register_buffer("running_centers", F.normalize(center, dim=-1))

        # Running angular dispersion per teacher: [T]
        # initialized to 1.0 so early training stays stable
        self.register_buffer("running_dispersion", torch.ones(len(self.teacher_names)))

        self.name_to_idx = {name: i for i, name in enumerate(self.teacher_names)}

    @torch.no_grad()
    def _update_teacher_stats(self, teacher_name: str, teacher_summary: torch.Tensor):
        """
        teacher_summary: [B, D], assumed already normalized
        """
        idx = self.name_to_idx[teacher_name]

        # Batch center direction
        batch_center = teacher_summary.mean(dim=0, keepdim=False)  # [D]
        batch_center = F.normalize(batch_center, dim=-1)

        # Update running center with EMA, then renormalize
        old_center = self.running_centers[idx]
        new_center = self.momentum * old_center + (1.0 - self.momentum) * batch_center
        new_center = F.normalize(new_center, dim=-1)
        self.running_centers[idx] = new_center

        # Compute teacher angular spread around updated center
        cos_to_center = (teacher_summary * new_center.unsqueeze(0)).sum(dim=-1)
        cos_to_center = cos_to_center.clamp(-1.0 + self.eps, 1.0 - self.eps)

        # angle in radians
        angles = torch.acos(cos_to_center)  # [B]
        batch_dispersion = angles.std(unbiased=False)

        old_disp = self.running_dispersion[idx]
        new_disp = self.momentum * old_disp + (1.0 - self.momentum) * batch_dispersion
        if self.max_dispersion is not None:
            new_disp = new_disp.clamp(min=self.min_dispersion, max=self.max_dispersion)
        else:
            new_disp = new_disp.clamp(min=self.min_dispersion)
        self.running_dispersion[idx] = new_disp

    def _pairwise_balanced_loss(
        self,
        student_summary: torch.Tensor,   # [B, D]
        teacher_summary: torch.Tensor,   # [B, D]
        teacher_name: str,
    ):
        idx = self.name_to_idx[teacher_name]

        s = F.normalize(student_summary, dim=-1)
        t = F.normalize(teacher_summary, dim=-1)
        if self.detach_teacher:
            t = t.detach()

        cos_st = (s * t).sum(dim=-1).clamp(-1.0 + self.eps, 1.0 - self.eps)  # [B]

        if self.loss_type == "angle":
            if self.training:
                self._update_teacher_stats(teacher_name, t)
            dispersion = self.running_dispersion[idx].detach()  # scalar
            raw_error = torch.acos(cos_st)  # angle distance, [B]
            return raw_error / (dispersion + self.eps)
        if self.loss_type == "cosine":
            return 1.0 - cos_st
        if self.loss_type == "mse":
            return F.mse_loss(s, t, reduction="none").mean(dim=-1)

        raise RuntimeError(f"Unexpected loss_type: {self.loss_type}")

    def forward(
        self,
        student_summary: torch.Tensor,
        teacher_summaries: Dict[str, torch.Tensor],
        teacher_weights: Optional[Dict[str, float]] = None,
        return_details: bool = False,
    ):
        """
        Args:
            student_summary: [B, D]
            teacher_summaries: dict[str, [B, D]]
            teacher_weights: optional dict[str, float]

        Returns:
            total_loss or (total_loss, details)
        """
        losses = {}
        total = 0.0

        for name, t_summary in teacher_summaries.items():
            if name not in self.name_to_idx:
                raise KeyError(f"Unknown teacher name: {name}")

            weight = 1.0 if teacher_weights is None else teacher_weights.get(name, 1.0)
            per_sample_loss = self._pairwise_balanced_loss(student_summary, t_summary, name)

            if self.reduction == "mean":
                loss_val = per_sample_loss.mean()
            elif self.reduction == "none":
                loss_val = per_sample_loss
            else:
                raise ValueError(f"Unsupported reduction: {self.reduction}")

            total = total + weight * loss_val
            losses[name] = loss_val

        if return_details:
            details = {
                "per_teacher_loss": losses,
                "running_dispersion": {
                    name: self.running_dispersion[self.name_to_idx[name]].item()
                    for name in self.teacher_names
                },
            }
            return total, details

        return total