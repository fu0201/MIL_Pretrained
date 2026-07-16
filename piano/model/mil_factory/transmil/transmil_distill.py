from .transmil import TransMIL


class TransMILDistill(TransMIL):
    def forward(self, input_dict, return_loss=True):
        output_dict = super().forward(input_dict, return_loss=return_loss)
        if "features" not in output_dict:
            output_dict["features"] = output_dict["logits"]
        return output_dict
