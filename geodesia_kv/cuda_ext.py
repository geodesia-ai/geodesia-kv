"""Compilazione JIT del kernel di attenzione a precisione mista.

Autosufficiente: individua la toolchain CUDA di sistema oppure quella installata
via pip (layout `nvidia/cu13`), senza dipendere da altri pacchetti.
Con `GEODESIA_DISABLE_CUDA=1` si forza il percorso PyTorch.
"""
from __future__ import annotations

import os
import shutil
import sysconfig
from pathlib import Path
from typing import Any, Optional

_EXT: Any = None
_TRIED = False


def _pip_cuda_home() -> Optional[Path]:
    site = Path(sysconfig.get_paths()["purelib"])
    for sub in ("nvidia/cu13", "nvidia/cu12", "nvidia/cuda_nvcc"):
        cand = site / sub
        if (cand / "bin" / "nvcc").exists():
            return cand
    return None


def _ensure_cuda_home() -> bool:
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if home and (Path(home) / "bin" / "nvcc").exists():
        return True
    if shutil.which("nvcc"):
        return True
    pip_home = _pip_cuda_home()
    if pip_home is None:
        return False
    lib64 = pip_home / "lib64"          # torch si aspetta lib64/, il wheel ha lib/
    if not lib64.exists():
        try:
            lib64.symlink_to(pip_home / "lib")
        except OSError:
            return False
    os.environ["CUDA_HOME"] = str(pip_home)
    return True


def _ensure_ninja_path() -> None:
    if shutil.which("ninja"):
        return
    try:
        import ninja
        binary_dir = Path(ninja.BIN_DIR)
    except (ImportError, AttributeError):
        return
    if (binary_dir / "ninja").exists():
        cur = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{binary_dir}{os.pathsep}{cur}" if cur else str(binary_dir)


def get_ext() -> Any:
    """Compila e restituisce l'estensione CUDA, o None se non disponibile."""
    global _EXT, _TRIED
    if _TRIED:
        return _EXT
    _TRIED = True
    if os.environ.get("GEODESIA_DISABLE_CUDA"):
        return None
    try:
        if not _ensure_cuda_home():
            return None
        _ensure_ninja_path()
        from torch.utils.cpp_extension import load
        root = Path(__file__).resolve().parent.parent / "geodesia_kv_cuda"
        _EXT = load(
            name="geodesia_kv_ext",
            sources=[str(root / "binding.cpp"), str(root / "mixed_attn.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=False,
        )
    except Exception as exc:                       # pragma: no cover
        print(f"[geodesia-kv] kernel CUDA non disponibile: {exc}")
        _EXT = None
    return _EXT
