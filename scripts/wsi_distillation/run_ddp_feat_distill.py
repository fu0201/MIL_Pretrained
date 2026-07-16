"""DDP feature distillation training script.

Scales the single-WSI distillation experiment to the full TCGA-8K dataset
using PyTorch DistributedDataParallel.

Each .pth file in the feature directory contains exactly 1024 patch features
of dimension 768, so all samples in a batch have the same shape and no
padding is required.

Launch:
    torchrun --nproc_per_node=<NUM_GPUS> run_ddp_feat_distill.py [OPTIONS]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ── Repo root ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from piano.model.mil_factory import create_mil_model  # noqa: E402
from scripts.loss import BalancedSummaryLoss  # noqa: E402
from scripts.teacher_model import Conch15TitanTokenExtractor  # noqa: E402


DEFAULT_FEAT_DIR = REPO_ROOT / "data" / "features"
DEFAULT_CARE_ROOT = REPO_ROOT / "models" / "CARE"
DEFAULT_TITAN_ROOT = REPO_ROOT / "models" / "TITAN"
DEFAULT_CACHE_FILE = REPO_ROOT / "outputs" / "teacher_cache.pt"
DEFAULT_SAVE_PATH = REPO_ROOT / "outputs" / "ddp_feat_distill.pt"
DEFAULT_CKPT_DIR = REPO_ROOT / "outputs" / "checkpoints"
DEFAULT_COORD_UNIT = 512  # patch size in WSI level-0 pixels
DEFAULT_AMP = "bf16"
DEFAULT_EPOCHS = 300
DEFAULT_LR = 1e-4 #5e-5，在1024的batchsize下
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DROPOUT = 0.25
DEFAULT_BATCH_SIZE = 1024 #之前是1024
DEFAULT_NUM_WORKERS = 16
DEFAULT_WANDB_PROJECT = "wsi-feat-distill"
DEFAULT_SAVE_EVERY = 100
DEFAULT_DDP_TIMEOUT_MINUTES = 720
DEFAULT_CACHE_WAIT_TIMEOUT_MINUTES = 720
DEFAULT_MIL_NAME = "transmil"
DEFAULT_FIXED_GPU_IDS = (2)
TITAN_TARGET = "pooled_slide_embedding"
# Teacher ablation: both (default) | titan_only | care_only. Use distinct --cache_file per mode
# when building from scratch; a cache built with "both" can be reused for titan_only or care_only.
DEFAULT_TEACHER_MODE = "both"

MIL_DISTILL_NAME_MAP = {
    "abmil": "abmil_distill",
    "gated_abmil": "gated_abmil_distill",
    "amdmil": "amdmil_distill",
    "clam_sb": "clam_sb_distill",
    "clam_mb": "clam_mb_distill",
    "transmil": "transmil_distill",
    "wikg": "wikg_distill",
    "aemmil": "aemmil_distill",
    "2dmamba": "2dmamba_distill",
    "dagmil": "dagmil_distill",
    "gdfmil": "gdfmil_distill",
}


def _cache_teacher_keys_present(cache: dict) -> tuple[bool, bool]:
    return ("titan_pooled" in cache, "care_summary" in cache)


def _validate_cache_for_teacher_mode(
    cache: dict,
    teacher_mode: str,
    cache_path: Path,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return (titan_pooled_all, care_summary_all) tensors or None if unused."""
    has_titan, has_care = _cache_teacher_keys_present(cache)

    if teacher_mode == "both":
        if not has_titan or not has_care:
            raise RuntimeError(
                f"Teacher cache must contain both 'titan_pooled' and 'care_summary' "
                f"for --teacher_mode both: {cache_path}"
            )
        if cache.get("titan_target", TITAN_TARGET) != TITAN_TARGET:
            raise RuntimeError(
                "Teacher cache is not for TITAN pooled slide embeddings. "
                f"Expected titan_target={TITAN_TARGET!r}, got "
                f"{cache.get('titan_target')!r}. Delete/rebuild: {cache_path}"
            )
        return cache["titan_pooled"], cache["care_summary"]

    if teacher_mode == "titan_only":
        if not has_titan:
            raise RuntimeError(
                f"Teacher cache missing 'titan_pooled' for --teacher_mode titan_only: "
                f"{cache_path}"
            )
        if cache.get("titan_target", TITAN_TARGET) != TITAN_TARGET:
            raise RuntimeError(
                "Teacher cache is not for TITAN pooled slide embeddings. "
                f"Expected titan_target={TITAN_TARGET!r}, got "
                f"{cache.get('titan_target')!r}. Delete/rebuild: {cache_path}"
            )
        return cache["titan_pooled"], None

    # care_only
    if not has_care:
        raise RuntimeError(
            f"Teacher cache missing 'care_summary' for --teacher_mode care_only: "
            f"{cache_path}"
        )
    return None, cache["care_summary"]


# ── DDP helpers ────────────────────────────────────────────────────────────────
def get_fixed_cuda_device(local_rank: int) -> torch.device:
    if local_rank < 0 or local_rank >= len(DEFAULT_FIXED_GPU_IDS):
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is out of range for fixed GPU list {DEFAULT_FIXED_GPU_IDS}"
        )
    physical_gpu_id = DEFAULT_FIXED_GPU_IDS[local_rank]
    if physical_gpu_id >= torch.cuda.device_count():
        raise RuntimeError(
            f"Physical GPU {physical_gpu_id} is unavailable; visible device count is {torch.cuda.device_count()}"
        )
    return torch.device(f"cuda:{physical_gpu_id}")


def setup_ddp(timeout_minutes: int) -> tuple[int, int, int, torch.device]:
    if timeout_minutes <= 0:
        raise ValueError("--ddp_timeout_minutes must be positive")
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=timeout_minutes),
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    device = get_fixed_cuda_device(local_rank)
    torch.cuda.set_device(device)
    return local_rank, dist.get_rank(), dist.get_world_size(), device


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def is_main() -> bool:
    return dist.get_rank() == 0


def print_main(*args, **kwargs) -> None:
    if is_main():
        print(*args, **kwargs)


def torchrun_env() -> tuple[int, int, int]:
    required = ("LOCAL_RANK", "RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(f"Missing torchrun environment variables: {missing_str}")
    return (
        int(os.environ["LOCAL_RANK"]),
        int(os.environ["RANK"]),
        int(os.environ["WORLD_SIZE"]),
    )


def log_visible_gpu_mapping(local_rank: int, rank: int, world_size: int) -> None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(
        f"[Rank {rank}/{world_size}] fixed_gpu_ids={DEFAULT_FIXED_GPU_IDS} | "
        f"CUDA_VISIBLE_DEVICES={visible_devices} | LOCAL_RANK={local_rank} | "
        f"cuda:{current_device} -> {device_name}",
        flush=True,
    )


# ── Misc utils ─────────────────────────────────────────────────────────────────
def autocast_ctx(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "fp32":
        return nullcontext()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[amp]
    return torch.autocast(device_type="cuda", dtype=dtype)


def freeze_model(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def resolve_distill_mil_name(mil_name: str) -> str:
    if mil_name not in MIL_DISTILL_NAME_MAP:
        valid_names = ", ".join(sorted(MIL_DISTILL_NAME_MAP))
        raise ValueError(
            f"Unknown --mil_name '{mil_name}'. Valid options: {valid_names}"
        )
    return MIL_DISTILL_NAME_MAP[mil_name]


def infer_student_embed_dim(
    student: nn.Module,
    sample_features: torch.Tensor,
    device: torch.device,
) -> int:
    was_training = student.training
    student.eval()
    try:
        with torch.no_grad():
            output_dict = student(
                {"features": sample_features.unsqueeze(0).to(device)},
                return_loss=False,
            )
    finally:
        if was_training:
            student.train()

    if "features" not in output_dict:
        raise KeyError("Distill MIL model must return output_dict['features']")

    student_features = output_dict["features"]
    if student_features.ndim != 2 or student_features.shape[0] != 1:
        raise RuntimeError(
            "Distill MIL model must return slide features with shape [B, D]; "
            f"got {tuple(student_features.shape)}"
        )
    return int(student_features.shape[-1])


def normalize_coords_to_grid(coords: torch.Tensor, coord_unit: int) -> torch.Tensor:
    offset = coords.min(dim=1, keepdim=True).values
    return torch.floor_divide(coords - offset, coord_unit).long()


def set_hf_env() -> None:
    """Set offline flags only; never override HF_HOME or HF_MODULES_CACHE.

    The conda environment already has HF_HOME set to the system-wide cache.
    Overriding HF_MODULES_CACHE would cause a path mismatch between where
    configure_titan_local_runtime copies modules and where transformers looks
    for them (transformers resolves the path at import time from HF_HOME).
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def wait_for_cache_file(
    cache_path: Path,
    rank: int,
    timeout_minutes: int,
    poll_seconds: float = 5.0,
) -> None:
    """Wait for rank 0 to create the cache without touching the process group."""
    deadline = (
        None
        if timeout_minutes <= 0
        else time.monotonic() + timeout_minutes * 60.0
    )
    last_log = 0.0
    while not cache_path.exists():
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            raise TimeoutError(
                f"[Rank {rank}] Timed out after {timeout_minutes} minutes "
                f"waiting for teacher cache: {cache_path}"
            )
        if now - last_log >= 60.0:
            print(f"[Rank {rank}] Waiting for teacher cache: {cache_path}", flush=True)
            last_log = now
        time.sleep(poll_seconds)


# ── Teacher extraction ─────────────────────────────────────────────────────────
def _extract_titan_pooled(
    titan_teacher: Conch15TitanTokenExtractor,
    patch_features: torch.Tensor,
    coords: torch.Tensor,
    amp: str,
    device: torch.device,
) -> torch.Tensor:
    patch_features = patch_features.to(device, non_blocking=True)
    coords = coords.to(device, non_blocking=True)
    with torch.inference_mode():
        with autocast_ctx(device, amp):
            slide_embedding = titan_teacher.titan.encode_slide_from_patch_features(
                patch_features, coords, DEFAULT_COORD_UNIT
            )
    return slide_embedding.float().cpu()  # [1, D_t]


def _extract_care_summary(
    care_model: nn.Module,
    patch_features: torch.Tensor,
    coords: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    patch_features = patch_features.to(device, non_blocking=True)
    grid_coords = normalize_coords_to_grid(coords, DEFAULT_COORD_UNIT).to(
        device, non_blocking=True
    )
    n_values = torch.tensor(
        [patch_features.shape[1]], dtype=torch.long, device=device
    )
    with torch.inference_mode():
        output = care_model(patch_features, n_values, grid_coords)
    return output.wsi_embedding.float().cpu()  # [1, D_c]


def build_teacher_cache(
    feat_files: list[Path],
    cache_path: Path,
    amp: str,
    device: torch.device,
    care_root: Path,
    titan_root: Path,
    teacher_mode: str,
) -> None:
    """Extract and save teacher embeddings for every feature file (rank-0 only)."""
    if teacher_mode not in {"both", "titan_only", "care_only"}:
        raise ValueError(f"Invalid teacher_mode for cache build: {teacher_mode!r}")

    print(
        f"[Rank 0] Building teacher cache ({teacher_mode}) for {len(feat_files)} files → {cache_path}"
    )

    titan_teacher = None
    care_teacher = None
    if teacher_mode in ("both", "titan_only"):
        titan_teacher = freeze_model(
            Conch15TitanTokenExtractor(
                checkpoint_path=str(titan_root),
                local_dir=True,
                tile_size=DEFAULT_COORD_UNIT,
                patch_size_lv0=DEFAULT_COORD_UNIT,
            ).to(device)
        )
    if teacher_mode in ("both", "care_only"):
        from scripts.care import load_care_model  # noqa: PLC0415

        care_teacher = freeze_model(load_care_model(care_root).to(device))

    titan_pooled_list: list[torch.Tensor] = []
    care_list: list[torch.Tensor] = []

    for i, fp in enumerate(feat_files):
        data = torch.load(fp, map_location="cpu", weights_only=True)
        patch_features = data["feats"].float().unsqueeze(0)  # [1, N, C]
        coords = data["coords"].long().unsqueeze(0)           # [1, N, 2]

        if titan_teacher is not None:
            titan_pooled_list.append(
                _extract_titan_pooled(
                    titan_teacher, patch_features, coords, amp, device
                ).squeeze(0)
            )  # [D_t]
        if care_teacher is not None:
            care_list.append(
                _extract_care_summary(
                    care_teacher, patch_features, coords, device
                ).squeeze(0)
            )  # [D_c]

        if (i + 1) % 200 == 0 or i == len(feat_files) - 1:
            print(f"  [{i + 1}/{len(feat_files)}] extracted")

    payload: dict = {
        "feat_files": [str(f) for f in feat_files],
        "teacher_mode": teacher_mode,
    }
    if titan_teacher is not None:
        payload["titan_target"] = TITAN_TARGET
        payload["titan_pooled"] = torch.stack(titan_pooled_list)  # [N, D_t]
    if care_teacher is not None:
        payload["care_summary"] = torch.stack(care_list)  # [N, D_c]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_cache_path = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp_cache_path)
    os.replace(tmp_cache_path, cache_path)
    print(f"[Rank 0] Teacher cache saved: {cache_path}")

    del titan_teacher, care_teacher
    torch.cuda.empty_cache()


# ── Dataset ────────────────────────────────────────────────────────────────────
class FeatDistillDataset(Dataset):
    def __init__(
        self,
        feat_files: list[Path],
        teacher_mode: str,
        titan_pooled_all: torch.Tensor | None,  # [N, D_t]
        care_summary_all: torch.Tensor | None,  # [N, D_c]
    ) -> None:
        if teacher_mode not in {"both", "titan_only", "care_only"}:
            raise ValueError(f"Invalid teacher_mode: {teacher_mode!r}")
        n = len(feat_files)
        if teacher_mode in ("both", "titan_only"):
            if titan_pooled_all is None:
                raise ValueError("titan_pooled_all is required for teacher_mode both/titan_only")
            if titan_pooled_all.shape[0] != n:
                raise ValueError("titan_pooled_all length must match feat_files")
        if teacher_mode in ("both", "care_only"):
            if care_summary_all is None:
                raise ValueError("care_summary_all is required for teacher_mode both/care_only")
            if care_summary_all.shape[0] != n:
                raise ValueError("care_summary_all length must match feat_files")

        self.feat_files = feat_files
        self.teacher_mode = teacher_mode
        self.titan_pooled_all = titan_pooled_all
        self.care_summary_all = care_summary_all

    def __len__(self) -> int:
        return len(self.feat_files)

    def __getitem__(self, idx: int):
        data = torch.load(self.feat_files[idx], map_location="cpu", weights_only=True)
        patch_features = data["feats"].float()          # [N, D]  e.g. [1024, 768]
        coords = data["coords"].long()                  # [N, 2]  real WSI patch coords
        if self.teacher_mode == "both":
            return (
                patch_features,
                coords,
                self.titan_pooled_all[idx],  # type: ignore[index]
                self.care_summary_all[idx],  # type: ignore[index]
            )
        if self.teacher_mode == "titan_only":
            return (
                patch_features,
                coords,
                self.titan_pooled_all[idx],  # type: ignore[index]
            )
        # care_only
        return (
            patch_features,
            coords,
            self.care_summary_all[idx],  # type: ignore[index]
        )


# ── Training loop ──────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    # Set HF environment vars before any model loading
    set_hf_env()
    distill_mil_name = resolve_distill_mil_name(args.mil_name)
    args.distill_mil_name = distill_mil_name
    teacher_mode = args.teacher_mode

    # Discover feature files
    feat_files = sorted(args.feat_dir.glob("*.pth"))
    if not feat_files:
        raise FileNotFoundError(f"No .pth files found in {args.feat_dir}")

    # Phase 1: build teacher cache before DDP initialization. Cache extraction can
    # take much longer than the default process-group timeout on the first run.
    pre_local_rank, pre_rank, _ = torchrun_env()
    pre_device = get_fixed_cuda_device(pre_local_rank)
    if pre_rank == 0:
        print(f"Found {len(feat_files)} feature files. teacher_mode={teacher_mode}")
        if not args.cache_file.exists():
            torch.cuda.set_device(pre_device)
            build_teacher_cache(
                feat_files=feat_files,
                cache_path=args.cache_file,
                amp=args.amp,
                device=pre_device,
                care_root=args.care_root,
                titan_root=args.titan_root,
                teacher_mode=teacher_mode,
            )
    else:
        wait_for_cache_file(
            args.cache_file,
            rank=pre_rank,
            timeout_minutes=args.cache_wait_timeout_minutes,
        )

    # Phase 2: initialize DDP only after cache is available.
    local_rank, rank, world_size, device = setup_ddp(args.ddp_timeout_minutes)
    log_visible_gpu_mapping(local_rank, rank, world_size)
    dist.barrier(device_ids=[device.index])

    # Phase 3: load teacher cache
    print_main(f"Loading teacher cache from {args.cache_file} ...")
    cache = torch.load(args.cache_file, map_location="cpu", weights_only=True)
    titan_pooled_all, care_summary_all = _validate_cache_for_teacher_mode(
        cache, teacher_mode, args.cache_file
    )

    dataset = FeatDistillDataset(
        feat_files, teacher_mode, titan_pooled_all, care_summary_all
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    # All samples are (1024, 768) → default collate stacks them into [B, 1024, 768]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # Infer feature dimensions
    sample = torch.load(feat_files[0], map_location="cpu", weights_only=True)
    student_dim = int(sample["feats"].shape[-1])
    titan_dim = (
        int(titan_pooled_all.shape[-1]) if titan_pooled_all is not None else 0
    )
    care_dim = (
        int(care_summary_all.shape[-1]) if care_summary_all is not None else 0
    )
    args.student_dim = student_dim
    print_main(
        f"Dims → student: {student_dim}, titan: {titan_dim or '—'}, "
        f"care: {care_dim or '—'} (teacher_mode={teacher_mode})"
    )
    print_main(f"Student MIL: {args.mil_name} -> {distill_mil_name}")
    print_main(f"World size: {world_size}, steps/epoch: {len(loader)}")

    # Student model + projection heads
    student_kwargs = {
        "dim_in": student_dim,
        "num_classes": 0,
        "dropout": args.dropout,
    }
    if distill_mil_name in {"abmil_distill", "gated_abmil_distill"}:
        student_kwargs["dim_hidden"] = None
    student = create_mil_model(distill_mil_name, **student_kwargs).to(device)
    student_embed_dim = infer_student_embed_dim(
        student, sample["feats"].float(), device
    )
    args.student_embed_dim = student_embed_dim
    print_main(f"Student embedding dim: {student_embed_dim}")
    student_to_titan: nn.Module | None = None
    student_to_care: nn.Module | None = None
    if teacher_mode in ("both", "titan_only"):
        student_to_titan = nn.Linear(student_embed_dim, titan_dim, bias=False).to(
            device
        )
    if teacher_mode in ("both", "care_only"):
        student_to_care = nn.Linear(student_embed_dim, care_dim, bias=False).to(
            device
        )

    # Loss functions (no learnable params → do NOT wrap in DDP)
    # Each rank maintains independent running stats; slight divergence is acceptable.
    titan_loss_fn: BalancedSummaryLoss | None = None
    care_loss_fn: BalancedSummaryLoss | None = None
    if teacher_mode in ("both", "titan_only"):
        titan_loss_fn = BalancedSummaryLoss(
            teacher_names=["titan"],
            embed_dim=titan_dim,
            reduction="mean",
            detach_teacher=True,
            use_angle=True,
        ).to(device)

    if teacher_mode in ("both", "care_only"):
        care_loss_fn = BalancedSummaryLoss(
            teacher_names=["care"],
            embed_dim=care_dim,
            reduction="mean",
            detach_teacher=True,
            use_angle=True,
        ).to(device)

    # Wrap learnable modules in DDP.
    # find_unused_parameters=True is required for dagmil: offset_net and
    # expand_alpha have no differentiable gradient path (argmin breaks backprop)
    # so DDP must be told to skip those parameters during gradient reduction.
    student = DDP(student, device_ids=[device.index], output_device=device.index, find_unused_parameters=True)
    if student_to_titan is not None:
        student_to_titan = DDP(student_to_titan, device_ids=[device.index], output_device=device.index)
    if student_to_care is not None:
        student_to_care = DDP(student_to_care, device_ids=[device.index], output_device=device.index)

    trainable_params: list[nn.Parameter] = list(student.parameters())
    if student_to_titan is not None:
        trainable_params += list(student_to_titan.parameters())
    if student_to_care is not None:
        trainable_params += list(student_to_care.parameters())
    optimizer = optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )

    # ── wandb init (rank 0 only) ───────────────────────────────────────────────
    use_wandb = _WANDB_AVAILABLE and args.wandb_project and is_main()
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "dropout": args.dropout,
                "amp": args.amp,
                "world_size": world_size,
                "mil_name": args.mil_name,
                "distill_mil_name": distill_mil_name,
                "student_dim": student_dim,
                "student_embed_dim": student_embed_dim,
                "titan_dim": titan_dim,
                "care_dim": care_dim,
                "teacher_mode": teacher_mode,
                "feat_dir": str(args.feat_dir),
                "num_samples": len(dataset),
            },
        )
        print_main(f"wandb run: {wandb.run.url}")

    # ── Checkpoint helper (rank 0 only) ───────────────────────────────────────
    ckpt_dir = args.ckpt_dir
    if is_main():
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(epoch: int) -> None:
        is_last = (epoch == args.epochs - 1)
        tag = "_final" if is_last else ""
        path = ckpt_dir / f"epoch{epoch:04d}{tag}.pt"
        ckpt: dict = {
            "epoch": epoch,
            "student_state_dict": student.module.state_dict(),
            "feat_dir": str(args.feat_dir),
            "coord_unit": DEFAULT_COORD_UNIT,
            "titan_target": TITAN_TARGET,
            "teacher_mode": teacher_mode,
            "mil_name": args.mil_name,
            "distill_mil_name": distill_mil_name,
            "student_dim": student_dim,
            "student_embed_dim": student_embed_dim,
        }
        if student_to_titan is not None:
            ckpt["student_to_titan_state_dict"] = student_to_titan.module.state_dict()
        if student_to_care is not None:
            ckpt["student_to_care_state_dict"] = student_to_care.module.state_dict()
        if titan_loss_fn is not None:
            ckpt["titan_loss_state_dict"] = titan_loss_fn.state_dict()
        if care_loss_fn is not None:
            ckpt["care_loss_state_dict"] = care_loss_fn.state_dict()
        torch.save(ckpt, path)
        print(f"Saved checkpoint → {path}")

    global_step = 0

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        student.train()
        if student_to_titan is not None:
            student_to_titan.train()
        if student_to_care is not None:
            student_to_care.train()
        if titan_loss_fn is not None:
            titan_loss_fn.train()
        if care_loss_fn is not None:
            care_loss_fn.train()

        epoch_loss = 0.0
        epoch_titan_loss = 0.0
        epoch_care_loss = 0.0
        epoch_titan_cos = 0.0
        epoch_care_cos = 0.0
        n_steps = 0
        n_samples = 0  # total samples seen this epoch on this GPU

        for batch in loader:
            if teacher_mode == "both":
                patch_feats, coords, titan_pooled, care_summary = batch
            elif teacher_mode == "titan_only":
                patch_feats, coords, titan_pooled = batch
                care_summary = None
            else:
                patch_feats, coords, care_summary = batch
                titan_pooled = None

            patch_feats = patch_feats.to(device, non_blocking=True)   # [B, N, D]
            coords = coords.to(device, non_blocking=True)            # [B, N, 2]

            optimizer.zero_grad()

            student_out = student(
                {"features": patch_feats, "coords": coords}, return_loss=False
            )
            student_embed = student_out["features"].float()  # [B, D_s]

            loss = torch.zeros((), device=device, dtype=student_embed.dtype)
            titan_loss = torch.zeros((), device=device, dtype=student_embed.dtype)
            care_loss = torch.zeros((), device=device, dtype=student_embed.dtype)
            titan_details = None
            care_details = None
            pred_titan = pred_care = None

            if teacher_mode in ("both", "titan_only"):
                assert titan_pooled is not None and student_to_titan is not None
                assert titan_loss_fn is not None
                titan_pooled = titan_pooled.to(device, non_blocking=True)  # [B, D_t]
                pred_titan = student_to_titan(student_embed)  # [B, D_t]
                titan_loss, titan_details = titan_loss_fn(
                    pred_titan, {"titan": titan_pooled}, return_details=True
                )
                loss = loss + titan_loss

            if teacher_mode in ("both", "care_only"):
                assert care_summary is not None and student_to_care is not None
                assert care_loss_fn is not None
                care_summary = care_summary.to(device, non_blocking=True)  # [B, D_c]
                pred_care = student_to_care(student_embed)  # [B, D_c]
                care_loss, care_details = care_loss_fn(
                    pred_care, {"care": care_summary}, return_details=True
                )
                loss = loss + care_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            titan_cos = 0.0
            care_cos = 0.0
            with torch.no_grad():
                if pred_titan is not None and titan_pooled is not None:
                    titan_cos = torch.nn.functional.cosine_similarity(
                        pred_titan, titan_pooled
                    ).mean().item()
                if pred_care is not None and care_summary is not None:
                    care_cos = torch.nn.functional.cosine_similarity(
                        pred_care, care_summary
                    ).mean().item()

            bsz = patch_feats.shape[0]  # actual batch size (may differ at last step)
            # accumulate as sample-weighted sums so epoch avg = true per-sample mean
            epoch_loss += loss.item() * bsz
            epoch_titan_loss += titan_loss.item() * bsz
            epoch_care_loss += care_loss.item() * bsz
            epoch_titan_cos += titan_cos * bsz
            epoch_care_cos += care_cos * bsz
            n_steps += 1
            n_samples += bsz
            global_step += 1

            # Per-step wandb log (step-level metrics stay as-is: mean over batch)
            if use_wandb:
                log_payload: dict = {"step/loss": loss.item()}
                if teacher_mode in ("both", "titan_only"):
                    log_payload["step/titan_loss"] = titan_loss.item()
                    log_payload["step/titan_cos"] = titan_cos
                    if titan_details is not None:
                        log_payload["step/titan_dispersion"] = titan_details[
                            "running_dispersion"
                        ]["titan"]
                if teacher_mode in ("both", "care_only"):
                    log_payload["step/care_loss"] = care_loss.item()
                    log_payload["step/care_cos"] = care_cos
                    if care_details is not None:
                        log_payload["step/care_dispersion"] = care_details[
                            "running_dispersion"
                        ]["care"]
                wandb.log(log_payload, step=global_step)

        if is_main():
            # divide by total samples → true per-sample average
            avg = lambda v: v / max(n_samples, 1)  # noqa: E731
            metrics: dict = {
                "epoch": epoch,
                "epoch/loss": avg(epoch_loss),
            }
            if teacher_mode in ("both", "titan_only"):
                metrics["epoch/titan_loss"] = avg(epoch_titan_loss)
                metrics["epoch/titan_cos"] = avg(epoch_titan_cos)
            if teacher_mode in ("both", "care_only"):
                metrics["epoch/care_loss"] = avg(epoch_care_loss)
                metrics["epoch/care_cos"] = avg(epoch_care_cos)

            if epoch % 10 == 0 or epoch == args.epochs - 1:
                parts = [
                    f"epoch={epoch:03d}",
                    f"loss={metrics['epoch/loss']:.6f}",
                ]
                if teacher_mode in ("both", "titan_only"):
                    parts.append(f"titan_loss={metrics['epoch/titan_loss']:.6f}")
                    parts.append(f"titan_cos={metrics['epoch/titan_cos']:.6f}")
                if teacher_mode in ("both", "care_only"):
                    parts.append(f"care_loss={metrics['epoch/care_loss']:.6f}")
                    parts.append(f"care_cos={metrics['epoch/care_cos']:.6f}")
                print(" ".join(parts))

            if use_wandb:
                wandb.log(metrics, step=global_step)

            is_last_epoch = (epoch == args.epochs - 1)
            if is_main() and (
                (epoch + 1) % args.save_every == 0 or is_last_epoch
            ):
                save_checkpoint(epoch)

    if use_wandb:
        wandb.finish()

    cleanup_ddp()


# ── Argument parsing ───────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DDP WSI feature distillation (TITAN / CARE / both teachers)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--feat_dir", type=Path, default=DEFAULT_FEAT_DIR,
                        help="Directory containing .pth patch feature files")
    parser.add_argument("--care_root", type=Path, default=DEFAULT_CARE_ROOT,
                        help="Path to local CARE model directory")
    parser.add_argument("--titan_root", type=Path, default=DEFAULT_TITAN_ROOT,
                        help="Path to local TITAN model directory")
    parser.add_argument("--cache_file", type=Path, default=DEFAULT_CACHE_FILE,
                        help="Path to teacher embedding cache (auto-built if absent)")
    parser.add_argument("--save_path", type=Path, default=DEFAULT_SAVE_PATH,
                        help="Output checkpoint path (kept for compatibility)")
    parser.add_argument("--ckpt_dir", type=Path, default=DEFAULT_CKPT_DIR,
                        help="Directory to save periodic checkpoints")
    parser.add_argument("--save_every", type=int, default=DEFAULT_SAVE_EVERY,
                        help="Save a checkpoint every N epochs")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Number of WSI regions per GPU per step")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--mil_name", type=str, default=DEFAULT_MIL_NAME,
                        choices=sorted(MIL_DISTILL_NAME_MAP),
                        help="MIL student backbone to distill")
    parser.add_argument(
        "--teacher_mode",
        type=str,
        default=DEFAULT_TEACHER_MODE,
        choices=["both", "titan_only", "care_only"],
        help=(
            "Which teacher(s) to distill from. "
            "Use a separate --cache_file per mode when building cache from scratch; "
            "a cache built with 'both' can be reused for titan_only or care_only "
            "(skips re-extracting the unused teacher)."
        ),
    )
    parser.add_argument("--amp", type=str, default=DEFAULT_AMP,
                        choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--ddp_timeout_minutes", type=int,
                        default=DEFAULT_DDP_TIMEOUT_MINUTES,
                        help="NCCL/process-group timeout after cache is ready")
    parser.add_argument("--cache_wait_timeout_minutes", type=int,
                        default=DEFAULT_CACHE_WAIT_TIMEOUT_MINUTES,
                        help="How long non-zero ranks wait for rank-0 cache; <=0 disables timeout")
    # wandb
    parser.add_argument("--wandb_project", type=str, default=DEFAULT_WANDB_PROJECT,
                        help="wandb project name; set empty string to disable")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="wandb run name (auto-generated if not set)")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
