import torch
import torch.nn as nn

from .clam import CLAM_MB, CLAM_SB


class CLAMDistill(CLAM_SB):
    def forward(self, input_dict, return_loss=True):
        features = input_dict["features"]
        if features.ndim == 2:
            features = features.unsqueeze(0)

        outputs = [
            CLAM_SB.forward(
                self,
                {"features": features[i].unsqueeze(0)},
                return_loss=False,
            )
            for i in range(features.shape[0])
        ]
        slide_features = torch.cat([output["features"] for output in outputs], dim=0)
        logits = torch.cat([output["logits"] for output in outputs], dim=0)
        output_dict = {
            "logits": logits,
            "features": slide_features,
            "loss": None,
        }
        if all("raw_attn" in output for output in outputs):
            output_dict["raw_attn"] = torch.stack(
                [output["raw_attn"] for output in outputs], dim=0
            )
        return output_dict


class CLAMMBDistill(CLAM_MB):
    def __init__(
        self,
        dim_in=1024,
        dim_hidden=512,
        dropout=0.25,
        num_classes=0,
        k_sample=8,
        instance_loss_fn=None,
        subtyping=False,
        survival=False,
        distill_branches=2,
    ):
        branch_count = distill_branches if num_classes == 0 else num_classes
        super().__init__(
            dim_in=dim_in,
            dim_hidden=dim_hidden,
            dropout=dropout,
            num_classes=branch_count,
            k_sample=k_sample,
            instance_loss_fn=instance_loss_fn,
            subtyping=subtyping,
            survival=survival,
        )
        # Also neutralise the singular CLAM_SB classifier that is never used
        # in _forward_one_bag; without this it becomes an unused DDP parameter.
        self.classifier = nn.Identity()
        self.classifiers = nn.ModuleList([nn.Identity() for _ in range(branch_count)])
        self.instance_classifiers = nn.ModuleList(
            [nn.Identity() for _ in range(branch_count)]
        )

    def _forward_one_bag(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        attn, h = self.attention_net(features)
        attn = torch.transpose(attn, 1, 0)
        raw_attn = attn
        attn = torch.softmax(attn, dim=1)
        class_features = torch.mm(attn, h)
        slide_features = class_features.mean(dim=0, keepdim=True)
        return {
            "logits": slide_features,
            "features": slide_features,
            "raw_attn": raw_attn,
            "class_features": class_features,
        }

    def forward(self, input_dict, return_loss=True):
        features = input_dict["features"]
        if features.ndim == 2:
            features = features.unsqueeze(0)

        outputs = [self._forward_one_bag(features[i]) for i in range(features.shape[0])]
        output_dict = {
            "logits": torch.cat([output["logits"] for output in outputs], dim=0),
            "features": torch.cat([output["features"] for output in outputs], dim=0),
            "raw_attn": torch.stack([output["raw_attn"] for output in outputs], dim=0),
            "class_features": torch.stack(
                [output["class_features"] for output in outputs], dim=0
            ),
            "loss": None,
        }
        return output_dict
