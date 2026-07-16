import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from piano.utils.wsi_finetune_tools import NLLSurvLoss


class ABMILDistill(nn.Module):
    def __init__(self, dim_in, dim_hidden=None, dropout=0.25, num_classes=1000, survival=False):
        super().__init__()
        if dim_hidden is None:
            self.dim_hidden = dim_in // 2
        else:
            self.dim_hidden = dim_hidden
        self.survival = survival

        self.attn_module = nn.Sequential(
            nn.Linear(dim_in, self.dim_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(self.dim_hidden, 1)
        )
        if num_classes == 0:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(dim_in, num_classes)

        if self.survival:
            self.loss_fn = NLLSurvLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_dict, return_loss=True):
        x = input_dict['features']
        attn = self.attn_module(x)
        A = torch.transpose(attn, -1, -2)
        A = torch.softmax(A, dim=-1)
        slide_embedding = torch.matmul(A, x).squeeze(1)
        logits = self.fc(slide_embedding)

        output_dict = {
            'logits': logits,
            'raw_attn': attn,
            'features': slide_embedding,
        }

        if self.survival:
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)

            output_dict.update({
                'Y_hat': Y_hat,
                'hazards': hazards,
                'S': S
            })

        if return_loss and 'labels' in input_dict:
            if self.survival and 'events' in input_dict:
                loss = self.loss_fn(
                    hazards=output_dict['hazards'],
                    S=output_dict['S'],
                    Y=input_dict['labels'],
                    c=input_dict['events'],
                )
            else:
                loss = self.loss_fn(logits, input_dict['labels'])
        else:
            loss = None

        output_dict['loss'] = loss
        return output_dict


class GatedABMILDistill(nn.Module):
    def __init__(self, dim_in, dim_hidden=None, dropout=0.25, num_classes=1000, survival=False):
        super().__init__()
        if dim_hidden is None:
            self.dim_hidden = dim_in // 2
        else:
            self.dim_hidden = dim_hidden

        self.survival = survival

        self.attn_1 = nn.Sequential(
            nn.Linear(dim_in, self.dim_hidden),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        self.attn_2 = nn.Sequential(
            nn.Linear(dim_in, self.dim_hidden),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )

        self.attn_3 = nn.Linear(self.dim_hidden, 1)

        if num_classes == 0:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(dim_in, num_classes)

        if self.survival:
            self.loss_fn = NLLSurvLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_dict, return_loss=True):
        x = input_dict['features']
        attn_1 = self.attn_1(x)
        attn_2 = self.attn_2(x)
        attn = attn_1.mul(attn_2)
        attn = self.attn_3(attn)
        A = torch.transpose(attn, -1, -2)
        A = torch.softmax(A, dim=-1)
        slide_embedding = torch.matmul(A, x).squeeze(1)
        logits = self.fc(slide_embedding)

        output_dict = {
            'logits': logits,
            'raw_attn': attn,
            'features': slide_embedding,
        }

        if self.survival:
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)

            output_dict.update({
                'Y_hat': Y_hat,
                'hazards': hazards,
                'S': S
            })

        if return_loss and 'labels' in input_dict:
            if self.survival and 'events' in input_dict:
                loss = self.loss_fn(
                    hazards=output_dict['hazards'],
                    S=output_dict['S'],
                    Y=input_dict['labels'],
                    c=input_dict['events'],
                )
            else:
                loss = self.loss_fn(logits, input_dict['labels'])
        else:
            loss = None

        output_dict['loss'] = loss
        return output_dict
