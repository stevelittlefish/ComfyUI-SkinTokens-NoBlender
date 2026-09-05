"""Bridge between ComfyUI's native 3D types and our numpy engine (spec/05).

ComfyUI core has two native 3D socket types (seen in the Trellis workflow):

  * ``MESH``  — in-memory geometry: torch tensors ``vertices (B,N,3)``,
    ``faces (B,M,3)``, optional ``normals``/``uvs``/``texture``/... No armature or
    skin — so it is an *input* to rigging, never the rigged output.
  * ``File3D`` (sockets ``FILE_3D_GLB``/``FILE_3D_GLTF``/...) — a file wrapper with
    ``get_bytes()`` / ``save_to(path)`` / ``is_disk_backed`` / ``format``. A rigged
    result must be a ``File3D`` because the glb file carries the skeleton + skin.

These helpers duck-type the objects (checking attributes, not ``isinstance``) so
this module imports with neither ``comfy`` nor ``torch`` present — the GPU/ComfyUI
paths run only on the server, while the conversion logic stays unit-testable with
plain numpy stand-ins.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional, Tuple

import numpy as np

# Socket type strings understood by ComfyUI. A comma-joined string in an input
# means "accept a link from any of these" (as the core Preview3D nodes declare).
FILE3D_TYPES = "FILE_3D_GLB,FILE_3D_GLTF,FILE_3D_OBJ,FILE_3D_STL,FILE_3D"
MESH_OR_FILE3D_TYPES = "MESH," + FILE3D_TYPES
RIGGED_FILE3D_TYPES = "FILE_3D_GLB,FILE_3D_GLTF,FILE_3D"


def _to_numpy(t) -> np.ndarray:
    """Convert a torch tensor (possibly on GPU) or array-like to a numpy array."""
    if hasattr(t, "detach"):  # torch.Tensor
        return t.detach().cpu().numpy()
    return np.asarray(t)


def is_comfy_mesh(obj) -> bool:
    """True for a ComfyUI ``MESH`` (has vertex/face geometry, no file API)."""
    return hasattr(obj, "vertices") and hasattr(obj, "faces")


def is_file3d(obj) -> bool:
    """True for a ComfyUI ``File3D`` (file-backed 3D object)."""
    return hasattr(obj, "get_bytes") and hasattr(obj, "save_to")


def comfy_mesh_to_arrays(mesh) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Extract ``(vertices, faces, normals|None)`` from a ComfyUI ``MESH``.

    Takes batch item 0 (our pipeline rigs one mesh at a time) and trims any
    zero-padding using ``vertex_counts`` / ``face_counts`` when present.
    """
    verts = _to_numpy(mesh.vertices)
    faces = _to_numpy(mesh.faces)
    if verts.ndim == 3:  # (B, N, 3) -> item 0
        verts = verts[0]
        faces = faces[0]
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int64)

    # Trim padding for variable-size mesh batches.
    vc = getattr(mesh, "vertex_counts", None)
    fc = getattr(mesh, "face_counts", None)
    if vc is not None:
        verts = verts[: int(_to_numpy(vc).reshape(-1)[0])]
    if fc is not None:
        faces = faces[: int(_to_numpy(fc).reshape(-1)[0])]

    normals = getattr(mesh, "normals", None)
    if normals is not None:
        normals = _to_numpy(normals)
        if normals.ndim == 3:
            normals = normals[0]
        normals = np.ascontiguousarray(normals[: verts.shape[0]], dtype=np.float32)
    return verts, faces, normals


def file3d_to_path(obj, tmp_dir: Optional[str] = None) -> str:
    """Return a filesystem path for a ``File3D``.

    Disk-backed objects return their existing path; memory-backed objects are
    written to a temp file (caller cleans up). The extension follows the object's
    declared format (default glb).
    """
    source = obj.get_source()
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
