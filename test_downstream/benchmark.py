import os
import json
import csv
import argparse
import contextlib
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
)

from slide_encoder_models.model_registry import create_slide_encoder


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TileFeatDataset(Dataset):
    def __init__(self, data_dir: str, json_path: str, pfm_name: str, split: str, max_patches: int | None = None):
        self.data_dir = data_dir
        self.pfm_name = pfm_name
        self.max_patches = max_patches
        with open(json_path, "r") as f:
            data = json.load(f)
        self.items = data[split]

        # Build global label map from all split lists.
        all_labels = sorted(
            {
                it["label"]
                for v in data.values()
                if isinstance(v, list)
                for it in v
                if isinstance(it, dict) and "label" in it
            }
        )
        self.label2idx = {label: i for i, label in enumerate(all_labels)}
        self.num_classes = len(self.label2idx)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        data_path = item["path"].replace("<PFM_NAME>", self.pfm_name)
        pth = torch.load(os.path.join(self.data_dir, data_path), map_location="cpu", weights_only=False)
        feats, coords = pth["feats"], pth["coords"]
        if self.max_patches is not None and feats.shape[0] > self.max_patches:
            feats, coords = feats[: self.max_patches], coords[: self.max_patches]
        return {
            "feats": feats,
            "coords": coords,
            "patch_size_lv0": pth["patch_size_level0"],
            "labels": self.label2idx[item["label"]],
        }


class SlideEncoderForDownstream(nn.Module):
    def __init__(self, model_name: str, dim_in: int, dim_hidden: int, num_classes: int):
        super().__init__()
        if model_name in [
            "gigapath",
            "chief",
            "madeleine",
            "prism",
            "titan",
            "feather_uni_v1",
            "feather_uni_v2",
            "feather_conch_v1_5",
            "alice_slide",
            "alice_slide_v2",
            "care",
        ]:
            self.model = create_slide_encoder(model_name, num_classes=num_classes)
        else:
            self.model = create_slide_encoder(model_name, dim_in=dim_in, dim_hidden=dim_hidden, num_classes=num_classes)

    def forward(self, feats, coords=None, patch_size_lv0=None, labels=None):
        logits = self.model({"feats": feats, "coords": coords, "patch_size_lv0": patch_size_lv0})
        loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits, "labels": labels}


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int):
    # bf16/fp16 tensors cannot be converted to numpy directly in many setups
    logits = logits.detach().float().cpu()
    labels = labels.detach().cpu()
    y_true = labels.numpy()
    y_pred = torch.argmax(logits, dim=1).numpy()
    probs = F.softmax(logits, dim=1).numpy()

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    try:
        if num_classes > 2:
            auc = roc_auc_score(y_true=y_true, y_score=probs, average="macro", multi_class="ovr")
        else:
            auc = roc_auc_score(y_true=y_true, y_score=probs[:, 1])
    except ValueError:
        auc = float("nan")

    specificity = []
    for c in range(num_classes):
        tn = np.sum((y_true != c) & (y_pred != c))
        fp = np.sum((y_true != c) & (y_pred == c))
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    return {
        "accuracy": acc,
        "bal_accuracy": bal_acc,
        "auc": auc,
        "f1": f1,
        "kappa": kappa,
        "macro_specificity": float(np.mean(specificity)),
        "confusion_mat": confusion_matrix(y_true, y_pred).tolist(),
    }


def run_epoch(model, loader, device, optimizer=None, grad_accum_steps=1, dtype="fp32"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    losses = []
    all_labels = []
    all_logits = []

    if is_train:
        optimizer.zero_grad()

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(dtype)
    autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()

    with torch.set_grad_enabled(is_train):
        from tqdm import tqdm
        for idx, batch in tqdm(enumerate(loader), total=len(loader), desc="Training" if is_train else "Validation"):
            feats = batch["feats"].to(device)
            coords = batch["coords"].to(device)
            patch_size_lv0 = batch["patch_size_lv0"].to(device)
            labels = batch["labels"].to(device)

            with autocast_ctx:
                out = model(feats, coords, patch_size_lv0, labels)
            loss = out["loss"]
            losses.append(loss.item())
            all_labels.append(out["labels"].detach().cpu())
            all_logits.append(out["logits"].detach().cpu())

            if is_train:
                (loss / grad_accum_steps).backward()
                if (idx + 1) % grad_accum_steps == 0 or (idx + 1) == len(loader):
                    optimizer.step()
                    optimizer.zero_grad()

    labels = torch.cat(all_labels, dim=0)
    logits = torch.cat(all_logits, dim=0)
    return float(np.mean(losses)), labels, logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MIL with small-train/val and multi-scale tests")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--pfm_name", type=str, default="pfm_name")
    parser.add_argument("--slide_name", type=str, default="slide_name")
    parser.add_argument("--job_dir", type=str, default="./results")
    parser.add_argument("--dataset_name", type=str, default=None, help="Custom dataset name for output directory. If not set, uses json filename.")
    parser.add_argument("--seed", type=int, default=2077)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument(
        "--best_metrics",
        type=str,
        default="bal_accuracy",
        choices=["accuracy", "bal_accuracy", "auc", "f1", "kappa", "macro_specificity"],
    )
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument(
        "--max_patches",
        type=int,
        default=None,
        help="If set, truncate each slide to at most this many patches (dim 0 of feats/coords).",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    mp = args.max_patches
    train_set = TileFeatDataset(args.data_dir, args.json_path, args.pfm_name, "train", max_patches=mp)
    val_set = TileFeatDataset(args.data_dir, args.json_path, args.pfm_name, "val", max_patches=mp)
    test_splits = ["test"]
    test_sets = {s: TileFeatDataset(args.data_dir, args.json_path, args.pfm_name, s, max_patches=mp) for s in test_splits}

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    test_loaders = {
        s: DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        for s, ds in test_sets.items()
    }

    in_dim = train_set[0]["feats"].shape[1]
    num_classes = train_set.num_classes
    model = SlideEncoderForDownstream(args.slide_name, in_dim, 512, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dataset_name = args.dataset_name if args.dataset_name else os.path.basename(args.json_path).split(".")[0]
    output_dir = os.path.join(args.job_dir, dataset_name, args.pfm_name, args.slide_name, str(args.seed), "benchmark")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    val_csv = os.path.join(output_dir, "val_metrics.csv")
    with open(val_csv, "w") as f:
        csv.writer(f).writerow(
            ["epoch", "loss", "accuracy", "bal_accuracy", "auc", "f1", "kappa", "macro_specificity"]
        )

    best_score = -float("inf")
    best_epoch = -1
    no_improve = 0
    min_delta = 1e-3

    for epoch in range(args.epochs):
        train_loss, _, _ = run_epoch(
            model, train_loader, device, optimizer=optimizer, grad_accum_steps=max(1, args.grad_accum_steps), dtype=args.dtype
        )
        val_loss, val_labels, val_logits = run_epoch(model, val_loader, device, dtype=args.dtype)
        val_metrics = compute_metrics(val_logits, val_labels, num_classes)

        with open(val_csv, "a") as f:
            csv.writer(f).writerow(
                [
                    epoch + 1,
                    val_loss,
                    val_metrics["accuracy"],
                    val_metrics["bal_accuracy"],
                    val_metrics["auc"],
                    val_metrics["f1"],
                    val_metrics["kappa"],
                    val_metrics["macro_specificity"],
                ]
            )

        print(
            f"Epoch {epoch+1:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_{args.best_metrics}={val_metrics[args.best_metrics]:.4f}"
        )

        score = val_metrics[args.best_metrics]
        if score > (best_score + min_delta):
            best_score = score
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, f"best_{args.best_metrics}.pth"))
            with open(os.path.join(output_dir, "best_val_metrics.json"), "w") as f:
                json.dump({"epoch": best_epoch, **val_metrics}, f, indent=2)
        else:
            no_improve += 1

        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    model.load_state_dict(torch.load(os.path.join(output_dir, f"best_{args.best_metrics}.pth"), map_location=device))
    model.eval()

    test_csv = os.path.join(output_dir, "test_metrics.csv")
    with open(test_csv, "w") as f:
        csv.writer(f).writerow(
            ["split", "accuracy", "bal_accuracy", "auc", "f1", "kappa", "macro_specificity"]
        )

    all_test_metrics = {}
    for split_name, loader in test_loaders.items():
        _, test_labels, test_logits = run_epoch(model, loader, device, dtype=args.dtype)
        metrics = compute_metrics(test_logits, test_labels, num_classes)
        all_test_metrics[split_name] = metrics
        with open(test_csv, "a") as f:
            csv.writer(f).writerow(
                [
                    split_name,
                    metrics["accuracy"],
                    metrics["bal_accuracy"],
                    metrics["auc"],
                    metrics["f1"],
                    metrics["kappa"],
                    metrics["macro_specificity"],
                ]
            )
        print(
            f"[{split_name}] acc={metrics['accuracy']:.4f} bal_acc={metrics['bal_accuracy']:.4f} "
            f"auc={metrics['auc']:.4f} f1={metrics['f1']:.4f} kappa={metrics['kappa']:.4f} "
            f"spec={metrics['macro_specificity']:.4f}"
        )

    with open(os.path.join(output_dir, "all_test_metrics.json"), "w") as f:
        json.dump({"best_epoch": best_epoch, "best_val_metric": args.best_metrics, **all_test_metrics}, f, indent=2)
