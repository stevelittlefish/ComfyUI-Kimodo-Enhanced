"""Bridge between ComfyUI's native 3D file types and Kimodo's export nodes.

Adapted from ComfyUI-SkinTokens-NoBlender (skintokens/comfy_types.py) — the
duck-typed helpers let a Kimodo node accept a ``File3D`` link straight from a
``Load3D`` node (or from SkinTokensRig's ``FILE_3D_GLB`` output) instead of a
hand-typed path string.

The helpers check attributes rather than ``isinstance`` so this module imports
without ``comfy`` present (keeps the logic unit-testable off the server).
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

# Socket type strings understood by ComfyUI. A comma-joined string in an input
# means "accept a link from any of these" (as the core Preview3D nodes declare).
FILE3D_TYPES = "FILE_3D_GLB,FILE_3D_GLTF,FILE_3D_OBJ,FILE_3D_STL,FILE_3D"
# A rigged character carries a skeleton + skin, so only glb/gltf make sense here.
RIGGED_FILE3D_TYPES = "FILE_3D_GLB,FILE_3D_GLTF,FILE_3D"


def is_file3d(obj) -> bool:
    """True for a ComfyUI ``File3D`` (file-backed 3D object)."""
    return hasattr(obj, "get_bytes") and hasattr(obj, "save_to")


def file3d_to_path(obj, tmp_dir: Optional[str] = None) -> str:
    """Return a filesystem path for a ``File3D`` (or pass through a plain string).

    Disk-backed objects return their existing path; memory-backed objects are
    written to a temp file (caller cleans up). The extension follows the
    object's declared format (default glb).
    """
    if isinstance(obj, str):
        return obj
    get_source = getattr(obj, "get_source", None)
    if callable(get_source):
        source = get_source()
        if isinstance(source, str) and os.path.exists(source):
            return source
    ext = (getattr(obj, "format", None) or "glb").lstrip(".")
    fd, path = tempfile.mkstemp(suffix=f".{ext}", dir=tmp_dir)
    os.close(fd)
    obj.save_to(path)
    return path


def make_file3d(path: str, file_format: str = "glb"):
    """Wrap an output file path in a ComfyUI ``File3D``.

    Falls back to returning the path string when ``comfy_api`` is unavailable
    (local dev / tests), so node logic stays exercisable without ComfyUI.
    """
    try:
        from comfy_api.latest import Types

        return Types.File3D(path, file_format=file_format)
    except Exception:
        return path
