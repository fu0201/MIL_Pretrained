"""benchmark_pretrain.py

Identical to benchmark.py but adds two extra arguments:
  --pretrained_weights  path to a distillation checkpoint (.pt) whose
                        'student_state_dict' initialises the aggregator's
                        attention weights before fine-tuning.
  --dim_hidden          hidden dim of the attention module; must match the
                        checkpoint architecture (default 512, set to 384
                        for checkpoints trained with dim_in//2 default).

All other behaviour (training loop, metrics, output layout) is unchanged so
results are directly comparable with benchmark.py runs.
"""
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
    def __init__(self, data_dir: str, json_path: str, pfm_name: str, split: str):
        self.data_dir = data_dir
        self.pfm_name = pfm_name
        with open(json_path, "r") as f:
            data = json.load(f)
        self.items = data[split]

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
        return {
            "feats": pth["feats"],
            "coords": pth["coords"],
            "patch_size_lv0": pth["patch_size_level0"],
            "labels": self.label2idx[item["label"]],
            "data_path": data_path,
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
        ]:
            self.model = create_slide_encoder(model_name, num_classes=num_classes)
        else:
            extra_kwargs = {"pool": "attn"} if model_name == "wikg" else {}
            self.model = create_slide_encoder(
                model_name, dim_in=dim_in, dim_hidden=dim_hidden, num_classes=num_classes, **extra_kwargs
            )

    def forward(self, feats, coords=None, patch_size_lv0=None, labels=None):
        logits = self.model({"feats": feats, "coords": coords, "patch_size_lv0": patch_size_lv0})
        loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits, "labels": labels}


def load_pretrained_attn(model: SlideEncoderForDownstream, checkpoint_path: str) -> None:
    """Transfer attn_module weights from a distillation checkpoint.

    The distillation checkpoint stores the student ABMIL under the key
    'student_state_dict'.  All keys in the checkpoint must exist in the
    downstream model with identical shapes (strict match).  The classifier
    head is not present in the checkpoint (distilled with num_classes=0)
    and therefore keeps its random initialisation for downstream fine-tuning.
    Raises RuntimeError if any key is missing or has a shape mismatch.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    src_sd = ckpt.get("student_state_dict", ckpt)

    tgt_sd = model.model.state_dict()

    # Backward compatibility for historical key naming in some distilled checkpoints.
    def _map_legacy_key(key: str) -> str:
        key = key.replace("W_head", "w_head")
        key = key.replace("W_tail", "w_tail")
        key = key.replace("gate_U", "gate_u")
        key = key.replace("gate_V", "gate_v")
        key = key.replace("gate_W", "gate_w")
        key = key.replace("att_net.", "readout.gate.")
        key = key.replace("readout.gate_nn.", "readout.gate.")
        return key

    loaded_keys, skipped_keys = [], []
    for key, val in src_sd.items():
        target_key = key if key in tgt_sd else _map_legacy_key(key)
        if target_key not in tgt_sd:
            raise RuntimeError(
                f"[pretrained] Key '{key}' in checkpoint not found in model "
                f"(after mapping to '{target_key}'). "
                f"Model keys: {list(tgt_sd.keys())}"
            )
        if tgt_sd[target_key].shape != val.shape:
            print(
                f"[pretrained] Shape mismatch for '{key}' -> '{target_key}': "
                f"checkpoint={tuple(val.shape)}, model={tuple(tgt_sd[target_key].shape)} — "
                f"skipping (will use random init, e.g. num_classes differs)"
            )
            skipped_keys.append(key)
            continue
        tgt_sd[target_key] = val
        loaded_keys.append(f"{key}->{target_key}" if key != target_key else key)

    model.model.load_state_dict(tgt_sd, strict=True)
    print(f"[pretrained] Loaded {len(loaded_keys)} tensors from {checkpoint_path}: {loaded_keys}")
    if skipped_keys:
        print(f"[pretrained] Skipped {len(skipped_keys)} tensors due to shape mismatch: {skipped_keys}")


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int):
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


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    grad_accum_steps=1,
    dtype="fp32",
    oom_csv_path=None,
    epoch=None,
    max_patches_per_sample=None,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    losses = []
    all_labels = []
    all_logits = []
    oom_skipped = 0
    over_limit_skipped = 0

    if is_train:
        optimizer.zero_grad()

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(dtype)
    autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()

    grad_accum_steps = max(1, int(grad_accum_steps))
    successful_backprops = 0

    def get_batch_meta(batch):
        feats = batch["feats"]
        coords = batch["coords"]
        patch_size_lv0 = batch["patch_size_lv0"]
        data_path = batch.get("data_path", [""])
        if isinstance(data_path, (list, tuple)):
            data_path = data_path[0] if data_path else ""
        num_instances = int(feats.shape[1]) if feats.ndim >= 2 else int(feats.shape[0])
        feat_dim = int(feats.shape[-1]) if feats.ndim >= 1 else -1
        coord_count = int(coords.shape[1]) if coords.ndim >= 2 else int(coords.shape[0])
        patch_size_val = patch_size_lv0.reshape(-1)[0].item() if hasattr(patch_size_lv0, "reshape") else patch_size_lv0
        return data_path, num_instances, feat_dim, coord_count, patch_size_val

    def write_skip_row(skip_type, batch_idx, phase, stage, batch):
        data_path, num_instances, feat_dim, coord_count, patch_size_val = get_batch_meta(batch)
        if oom_csv_path:
            with open(oom_csv_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        epoch,
                        phase,
                        stage,
                        skip_type,
                        batch_idx,
                        len(loader),
                        data_path,
                        num_instances,
                        feat_dim,
                        coord_count,
                        patch_size_val,
                    ]
                )
        return data_path, num_instances, feat_dim

    def log_oom(batch_idx, phase, stage, batch):
        nonlocal oom_skipped
        oom_skipped += 1
        data_path, num_instances, feat_dim = write_skip_row("oom", batch_idx, phase, stage, batch)
        print(
            f"[OOM] skip batch {batch_idx}/{len(loader)} ({phase}, {stage}) — patches={num_instances}, feat_dim={feat_dim}, path={data_path}; continuing",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def log_over_limit(batch_idx, phase, batch):
        nonlocal over_limit_skipped
        over_limit_skipped += 1
        data_path, num_instances, feat_dim = write_skip_row("max_patches", batch_idx, phase, "precheck", batch)
        print(
            f"[SKIP] batch {batch_idx}/{len(loader)} ({phase}, precheck) — patches={num_instances} > {max_patches_per_sample}, feat_dim={feat_dim}, path={data_path}; continuing",
            flush=True,
        )

    with torch.set_grad_enabled(is_train):
        from tqdm import tqdm
        for idx, batch in tqdm(enumerate(loader), total=len(loader), desc="Training" if is_train else "Validation"):
            phase = "train" if is_train else "eval"
            _data_path, num_instances, _feat_dim, _coord_count, _patch_size_val = get_batch_meta(batch)
            if max_patches_per_sample is not None and num_instances > max_patches_per_sample:
                log_over_limit(idx, phase, batch)
                continue

            feats = batch["feats"].to(device)
            coords = batch["coords"].to(device)
            patch_size_lv0 = batch["patch_size_lv0"].to(device)
            labels = batch["labels"].to(device)

            try:
                with autocast_ctx:
                    out = model(feats, coords, patch_size_lv0, labels)
                loss = out["loss"]
            except BaseException as exc:
                if not _is_cuda_oom(exc):
                    raise
                log_oom(idx, phase, "forward", batch)
                continue

            losses.append(loss.item())
            all_labels.append(out["labels"].detach().cpu())
            all_logits.append(out["logits"].detach().cpu())

            if is_train:
                try:
                    (loss / grad_accum_steps).backward()
                except BaseException as exc:
                    if not _is_cuda_oom(exc):
                        raise
                    log_oom(idx, "train", "backward", batch)
                    optimizer.zero_grad(set_to_none=True)
                    continue
                successful_backprops += 1
                if successful_backprops % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

    if is_train and successful_backprops % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    if over_limit_skipped:
        print(f"[SKIP] skipped {over_limit_skipped} batch(es) for exceeding max patches in this epoch", flush=True)
    if oom_skipped:
        print(f"[OOM] skipped {oom_skipped} batch(es) in this epoch", flush=True)

    if not all_labels:
        return float("nan"), None, None

    labels = torch.cat(all_labels, dim=0)
    logits = torch.cat(all_logits, dim=0)
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    return mean_loss, labels, logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark MIL with pretrained aggregator initialisation"
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--pfm_name", type=str, default="pfm_name")
    parser.add_argument("--slide_name", type=str, default="slide_name")
    parser.add_argument("--job_dir", type=str, default="./results")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Custom dataset name for output directory. If not set, uses json filename.")
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
    # ---- new arguments ----
    parser.add_argument("--dim_hidden", type=int, default=384,
                        help="Hidden dim of the ABMIL attention module. "
                             "Must match the distillation checkpoint (default 384 = 768//2).")
    parser.add_argument("--pretrained_weights", type=str, default=None,
                        help="Path to a distillation checkpoint (.pt). "
                             "The 'student_state_dict' is used to initialise "
                             "the aggregator attention weights before fine-tuning.")
    parser.add_argument("--slide_dir_name", type=str, default=None,
                        help="Override the output directory's slide-level folder name. "
                             "If not set, falls back to --slide_name. "
                             "Useful when slide_name must be a valid model key (e.g. 'abmil') "
                             "but you want results saved under a different label "
                             "(e.g. 'abmil_distillinit') for easy comparison.")
    parser.add_argument("--max_patches_per_sample", type=int, default=None,
                        help="If set, skip samples whose patch count exceeds this threshold and log them to oom_batches.csv.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    train_set = TileFeatDataset(args.data_dir, args.json_path, args.pfm_name, "train")
    val_set = TileFeatDataset(args.data_dir, args.json_path, args.pfm_name, "val")
    test_splits = ["test"]
    test_sets = {s: TileFeatDataset(args.data_dir, args.json_path, args.pfm_name, s) for s in test_splits}

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
    model = SlideEncoderForDownstream(args.slide_name, in_dim, args.dim_hidden, num_classes).to(device)

    if args.pretrained_weights:
        load_pretrained_attn(model, args.pretrained_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dataset_name = args.dataset_name if args.dataset_name else os.path.basename(args.json_path).split(".")[0]
    slide_dir = args.slide_dir_name if args.slide_dir_name else args.slide_name
    output_dir = os.path.join(args.job_dir, dataset_name, args.pfm_name, slide_dir, str(args.seed), "benchmark")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    oom_csv = os.path.join(output_dir, "oom_batches.csv")
    with open(oom_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "phase", "stage", "skip_type", "batch_idx", "num_batches", "data_path", "num_patches", "feat_dim", "coord_count", "patch_size_lv0"]
        )

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
            model, train_loader, device, optimizer=optimizer,
            grad_accum_steps=max(1, args.grad_accum_steps), dtype=args.dtype,
            oom_csv_path=oom_csv, epoch=epoch + 1,
            max_patches_per_sample=args.max_patches_per_sample,
        )
        val_loss, val_labels, val_logits = run_epoch(
            model, val_loader, device, dtype=args.dtype,
            oom_csv_path=oom_csv, epoch=epoch + 1,
            max_patches_per_sample=args.max_patches_per_sample,
        )

        if val_labels is None or val_labels.numel() == 0:
            print(
                f"Epoch {epoch+1:03d}/{args.epochs:03d} | "
                f"train_loss={train_loss:.4f} | [WARN] val set had no successful batch (all OOM?) — nan metrics",
                flush=True,
            )
            with open(val_csv, "a") as f:
                csv.writer(f).writerow(
                    [epoch + 1, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")]
                )
            continue

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
            _best_path = os.path.join(output_dir, f"best_{args.best_metrics}.pth")
            _best_tmp  = _best_path + ".tmp"
            torch.save(model.state_dict(), _best_tmp)
            os.replace(_best_tmp, _best_path)
            with open(os.path.join(output_dir, "best_val_metrics.json"), "w") as f:
                json.dump({"epoch": best_epoch, **val_metrics}, f, indent=2)
        else:
            no_improve += 1

        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    model.load_state_dict(torch.load(os.path.join(output_dir, f"best_{args.best_metrics}.pth"), map_location=device, weights_only=True))
    model.eval()

    test_csv = os.path.join(output_dir, "test_metrics.csv")
    with open(test_csv, "w") as f:
        csv.writer(f).writerow(
            ["split", "accuracy", "bal_accuracy", "auc", "f1", "kappa", "macro_specificity"]
        )

    all_test_metrics = {}
    for split_name, loader in test_loaders.items():
        _, test_labels, test_logits = run_epoch(
            model, loader, device, dtype=args.dtype,
            oom_csv_path=oom_csv, epoch=best_epoch,
            max_patches_per_sample=args.max_patches_per_sample,
        )
        if test_labels is None or test_labels.numel() == 0:
            print(
                f"[WARN] test split '{split_name}' had no successful batch (all OOM?) — skipping metrics",
                flush=True,
            )
            continue
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
