import os
import re
import shutil

import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModel

from ..base_model import BaseModel, register_model


def resolve_titan_source(checkpoint_path=None, local_dir=False):
    def resolve_snapshot(path_str: str):
        path = Path(path_str).expanduser()
        if not path.exists():
            return None
        if path.is_file():
            return None
        if (path / 'config.json').exists():
            return str(path)

        ref_main = path / 'refs' / 'main'
        if ref_main.exists():
            snapshot_name = ref_main.read_text(encoding='utf-8').strip()
            snapshot_dir = path / 'snapshots' / snapshot_name
            if (snapshot_dir / 'config.json').exists():
                return str(snapshot_dir)

        snapshots_root = path / 'snapshots'
        if snapshots_root.is_dir():
            for snapshot_dir in sorted(snapshots_root.iterdir()):
                if snapshot_dir.is_dir() and (snapshot_dir / 'config.json').exists():
                    return str(snapshot_dir)
        return None

    if local_dir and checkpoint_path is not None:
        resolved = resolve_snapshot(checkpoint_path)
        if resolved is not None:
            return resolved, True

    for env_name in ('PIANO_CONCH_V15_PATH', 'PIANO_TITAN_PATH', 'HF_TITAN_PATH'):
        env_value = os.environ.get(env_name)
        if env_value:
            resolved = resolve_snapshot(env_value)
            if resolved is not None:
                return resolved, True

    default_candidates = [
        '/mnt/sdb/ljw/hf_cache/models--MahmoodLab--TITAN',
        str(Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--MahmoodLab--TITAN'),
    ]
    for candidate in default_candidates:
        resolved = resolve_snapshot(candidate)
        if resolved is not None:
            return resolved, True

    if checkpoint_path is not None:
        resolved = resolve_snapshot(checkpoint_path)
        if resolved is not None:
            return resolved, True

    from ..model_registry import get_model_hf_path
    return get_model_hf_path('conch_v1_5'), False


def configure_titan_local_runtime(snapshot_dir: str):
    snapshot_path = Path(snapshot_dir).expanduser().resolve()
    hub_cache_root = snapshot_path.parent.parent.parent

    os.environ.setdefault('HF_HUB_CACHE', str(hub_cache_root))
    os.environ.setdefault('HUGGINGFACE_HUB_CACHE', str(hub_cache_root))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

    hf_modules_root = os.environ.get('HF_MODULES_CACHE')
    hf_modules_root_path = Path(hf_modules_root).expanduser() if hf_modules_root else None

    def ensure_writable_modules_root(path: Path) -> Path | None:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / '.write_test'
            probe.touch(exist_ok=True)
            probe.unlink(missing_ok=True)
            return path
        except OSError:
            return None

    if hf_modules_root_path is None:
        candidates = []
        hf_home = Path(os.environ.get('HF_HOME', str(Path.home() / '.cache' / 'huggingface')))
        candidates.append(hf_home / 'modules')
        candidates.append(Path('/mnt/net_sda/fmx/hf_modules_cache'))
        candidates.append(Path('/tmp/piano_hf_modules'))

        for candidate in candidates:
            resolved = ensure_writable_modules_root(candidate.expanduser())
            if resolved is not None:
                hf_modules_root_path = resolved
                os.environ['HF_MODULES_CACHE'] = str(hf_modules_root_path)
                break
    else:
        writable = ensure_writable_modules_root(hf_modules_root_path)
        if writable is None:
            fallback = ensure_writable_modules_root(Path('/tmp/piano_hf_modules'))
            if fallback is None:
                raise OSError('Could not find a writable directory for HF_MODULES_CACHE.')
            hf_modules_root_path = fallback
            os.environ['HF_MODULES_CACHE'] = str(hf_modules_root_path)
        else:
            hf_modules_root_path = writable

    if hf_modules_root_path is None:
        raise OSError('Could not initialize HF_MODULES_CACHE.')

    einops_exts_stub = hf_modules_root_path / 'einops_exts.py'
    if not einops_exts_stub.exists():
        einops_exts_stub.write_text(
            "from einops import rearrange\n\n"
            "def rearrange_many(tensors, pattern, **axes_lengths):\n"
            "    return tuple(rearrange(tensor, pattern, **axes_lengths) for tensor in tensors)\n",
            encoding='utf-8',
        )

    modules_root = hf_modules_root_path / 'transformers_modules' / snapshot_path.name
    modules_root.mkdir(parents=True, exist_ok=True)
    (modules_root / '__init__.py').touch(exist_ok=True)

    for src in snapshot_path.glob('*.py'):
        shutil.copyfile(src, modules_root / src.name)

    # Remote TITAN code may still call `huggingface_hub.hf_hub_download(...)`
    # during model construction. Patch it process-wide so those calls resolve
    # to files inside the local snapshot instead of hitting the Hub.
    import huggingface_hub

    original_hf_hub_download = huggingface_hub.hf_hub_download
    if not getattr(original_hf_hub_download, '_piano_local_titan_patch', False):
        def local_hf_hub_download(repo_id, filename, *args, **kwargs):
            if repo_id == 'MahmoodLab/TITAN':
                local_file = snapshot_path / filename
                if local_file.exists():
                    return str(local_file)
            return original_hf_hub_download(repo_id, filename, *args, **kwargs)

        local_hf_hub_download._piano_local_titan_patch = True
        huggingface_hub.hf_hub_download = local_hf_hub_download

    # TITAN remote code hardcodes `PreTrainedTokenizerFast.from_pretrained("MahmoodLab/TITAN")`.
    # When a local gated snapshot is already available, rewrite that call to the resolved snapshot
    # so model construction never tries to hit the hub again.
    tokenizer_py = modules_root / 'conch_tokenizer.py'
    if tokenizer_py.exists():
        content = tokenizer_py.read_text(encoding='utf-8')
        remote_call = 'PreTrainedTokenizerFast.from_pretrained("MahmoodLab/TITAN")'
        local_call = (
            f'PreTrainedTokenizerFast.from_pretrained('
            f'"{snapshot_path.as_posix()}", local_files_only=True)'
        )
        if remote_call in content:
            tokenizer_py.write_text(
                content.replace(remote_call, local_call),
                encoding='utf-8',
            )

    conch_py = modules_root / 'conch_v1_5.py'
    if conch_py.exists():
        content = conch_py.read_text(encoding='utf-8')
        local_weight_path = (snapshot_path / 'conch_v1_5_pytorch_model.bin').as_posix()
        content = re.sub(
            r'from huggingface_hub import hf_hub_download\s+'
            r'checkpoint_path = hf_hub_download\(\s*'
            r'"MahmoodLab/TITAN",\s*'
            r'filename="conch_v1_5_pytorch_model\.bin",\s*'
            r'\)',
            f'checkpoint_path = "{local_weight_path}"',
            content,
            count=1,
            flags=re.MULTILINE,
        )
        conch_py.write_text(content, encoding='utf-8')


@register_model('conch_v1_5')
class CONCHV1_5Model(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        checkpoint_path, local_files_only = resolve_titan_source(
            checkpoint_path=checkpoint_path,
            local_dir=local_dir,
        )
        if local_files_only:
            configure_titan_local_runtime(checkpoint_path)
        self.titan = AutoModel.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        conch, eval_transform = self.titan.return_conch()
        self.backbone = conch
        self.image_preprocess = eval_transform

        self.output_dim = 768
    
    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            output = self.backbone(x)
        return output 
