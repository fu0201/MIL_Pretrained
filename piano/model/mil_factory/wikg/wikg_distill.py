import torch

from .wikg import WiKG


class WiKGDistill(WiKG):
    def forward(self, input_dict, return_loss=True):
        features = input_dict["features"] if isinstance(input_dict, dict) else input_dict
        if features.ndim == 2:
            features = features.unsqueeze(0)

        outputs = [
            WiKG.forward(
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
            output_dict["raw_attn"] = [output["raw_attn"] for output in outputs]
        return output_dict
