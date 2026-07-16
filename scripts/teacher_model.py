from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from piano.model.patch_encoder.conch.conch_v1_5_model import (  # noqa: E402
    configure_titan_local_runtime,
    resolve_titan_source,
)


DEFAULT_RESIZE = 2048
DEFAULT_TILE_SIZE = 512
DEFAULT_PATCH_SIZE_LV0 = 512
DEFAULT_AMP = "bf16"


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_patch_features_for_titan(
    features: torch.Tensor,
    coords: torch.Tensor,
    patch_size_lv0: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if features.ndim == 3 and features.shape[0] == 1:
        features = features.squeeze(0)
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords.squeeze(0)

    if features.ndim != 2:
        raise ValueError(f"Expected features to be [N, C], got {tuple(features.shape)}")
    if coords.ndim != 2 or coords.shape[-1] != 2:
        raise ValueError(f"Expected coords to be [N, 2], got {tuple(coords.shape)}")

    offset = coords.min(dim=0).values
    grid_coords = torch.floor_divide(coords - offset, patch_size_lv0)

    grid_offset = grid_coords.min(dim=0).values
    grid_coords = grid_coords - grid_offset

    grid_h, grid_w = grid_coords.max(dim=0).values + 1
    grid_h = int(grid_h.item())
    grid_w = int(grid_w.item())

    feature_grid = torch.zeros(
        (grid_h, grid_w, features.size(-1)),
        device=features.device,
        dtype=features.dtype,
    )
    coords_grid = torch.zeros(
        (grid_h, grid_w, 2),
        device=coords.device,
        dtype=torch.int64,
    )

    indices = grid_coords[:, 0] * grid_w + grid_coords[:, 1]
    feature_grid.view(-1, features.size(-1)).index_add_(0, indices, features)
    coords_grid.view(-1, 2).index_add_(0, indices, coords)

    feature_grid = feature_grid.permute(2, 0, 1)
    coords_grid = coords_grid.permute(2, 0, 1)
    bg_mask = torch.any(feature_grid != 0, dim=0)
    return feature_grid.unsqueeze(0), coords_grid.unsqueeze(0), bg_mask.unsqueeze(0)


class Conch15TitanTokenExtractor(nn.Module):
    def __init__(
        self,
        checkpoint_path: str | None = None,
        local_dir: bool = False,
        tile_size: int = DEFAULT_TILE_SIZE,
        patch_size_lv0: int = DEFAULT_PATCH_SIZE_LV0,
    ) -> None:
        super().__init__()
        checkpoint_path, local_files_only = resolve_titan_source(
            checkpoint_path=checkpoint_path,
            local_dir=local_dir,
        )
        if local_files_only:
            configure_titan_local_runtime(checkpoint_path)

        titan = AutoModel.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        conch, _ = titan.return_conch()

        self.titan = titan
        self.conch = conch
        self.vision_encoder = titan.vision_encoder
        self.tile_size = tile_size
        self.patch_size_lv0 = patch_size_lv0

        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        self.eval()

    def _to_tensor(self, image: Any) -> torch.Tensor:
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            tensor = torch.from_numpy(np.asarray(image, dtype=np.uint8))
            tensor = tensor.permute(2, 0, 1).float() / 255.0
            return tensor

        if not torch.is_tensor(image):
            raise TypeError(f"Unsupported image type: {type(image).__name__}")

        tensor = image
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError(f"Only batch size 1 is supported, got {tuple(tensor.shape)}")
            tensor = tensor[0]
        elif tensor.ndim == 3 and tensor.shape[0] not in (1, 3) and tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)

        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise ValueError(f"Expected image shape [3, H, W] or [1, 3, H, W], got {tuple(tensor.shape)}")

        tensor = tensor.float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor

    def _tile_image(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, height, width = image.shape
        if height % self.tile_size != 0 or width % self.tile_size != 0:
            raise ValueError(
                f"Image size {(height, width)} must be divisible by tile_size={self.tile_size}"
            )

        num_rows = height // self.tile_size
        num_cols = width // self.tile_size

        patches = image.unsqueeze(0).unfold(2, self.tile_size, self.tile_size).unfold(3, self.tile_size, self.tile_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(num_rows * num_cols, 3, self.tile_size, self.tile_size)

        coords = []
        for row in range(num_rows):
            for col in range(num_cols):
                coords.append([row * self.patch_size_lv0, col * self.patch_size_lv0])
        coords = torch.tensor(coords, dtype=torch.long, device=image.device)
        return patches, coords

    def _autocast_context(self, device: torch.device, amp: str):
        if device.type != "cuda" or amp == "fp32":
            return nullcontext()
        dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[amp]
        return torch.autocast(device_type="cuda", dtype=dtype)

    def forward(
        self,
        image: Any,
        amp: str = DEFAULT_AMP,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        image_tensor = self._to_tensor(image).to(device, non_blocking=True)
        patches, coords = self._tile_image(image_tensor)
        patches = (patches - self.pixel_mean) / self.pixel_std

        with torch.inference_mode():
            with self._autocast_context(device, amp):
                patch_features = self.conch(patches)

            patch_features = patch_features.float().unsqueeze(0)
            coords = coords.unsqueeze(0)
            feature_grid, coords_grid, bg_mask = preprocess_patch_features_for_titan(
                patch_features,
                coords,
                patch_size_lv0=self.patch_size_lv0,
            )

            with self._autocast_context(device, amp):
                x = self.vision_encoder.forward_features(
                    feature_grid,
                    coords=coords_grid,
                    bg_mask=bg_mask,
                )

        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]
        return cls_token, patch_tokens


def extract_cls_and_patch_tokens(
    image: Any,
    checkpoint_path: str | None = None,
    local_dir: bool = False,
    tile_size: int = DEFAULT_TILE_SIZE,
    patch_size_lv0: int = DEFAULT_PATCH_SIZE_LV0,
    device: str | torch.device = "auto",
    amp: str = DEFAULT_AMP,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(device, torch.device):
        device = resolve_device(device)

    model = Conch15TitanTokenExtractor(
        checkpoint_path=checkpoint_path,
        local_dir=local_dir,
        tile_size=tile_size,
        patch_size_lv0=patch_size_lv0,
    ).to(device)
    return model(image, amp=amp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract TITAN CLS and patch tokens from one image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to a local RGB image.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="TITAN checkpoint or model path.")
    parser.add_argument("--local_dir", action="store_true", help="Treat checkpoint_path as a local model directory.")
    parser.add_argument("--resize", type=int, default=DEFAULT_RESIZE, help="Resize image to this square size.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", type=str, default=DEFAULT_AMP, choices=["fp32", "fp16", "bf16"])
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image = image.resize((args.resize, args.resize), resample=Image.BILINEAR)

    device = resolve_device(args.device)
    cls_token, patch_tokens = extract_cls_and_patch_tokens(
        image=image,
        checkpoint_path=args.checkpoint_path,
        local_dir=args.local_dir,
        device=device,
        amp=args.amp,
    )
    cls_token = cls_token.detach().cpu()
    patch_tokens = patch_tokens.detach().cpu()

    print(f"image={image_path}")
    print(f"device={device}")
    print(f"original_size={original_size}")
    print(f"resized_size={image.size}")
    print(f"tile_size={DEFAULT_TILE_SIZE}")
    print(f"patch_size_lv0={DEFAULT_PATCH_SIZE_LV0}")
    print()
    print(f"cls_token.shape={tuple(cls_token.shape)}")
    print(f"patch_tokens.shape={tuple(patch_tokens.shape)}")


if __name__ == "__main__":
    main()
