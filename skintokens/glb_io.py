"""Pure-Python glb/gltf import (replaces upstream's Blender-based load).

Phase 2 scope (see spec/02, spec/TODO): import a mesh from a glb/gltf/obj file
into the same ``Asset`` mesh fields that ``BpyParser.load`` produced, using
``trimesh`` instead of ``bpy``. Skinned *export* (the hard part) is Phase 3;
armature *import* (``use_skeleton``) is Phase 6.

Fields reproduced (mesh-only; no armature):
  vertices, faces, vertex_normals, face_normals, vertex_bias, face_bias,
  mesh_names. Normals are computed by trimesh from the merged mesh — matching
  upstream, which also recomputed them rather than reading them from the file.

AXIS CONVENTION (deferred to Phase 3): Blender's glTF importer converts glTF's
Y-up to Blender's Z-up, so upstream Assets are Z-up. trimesh keeps the file's
native glTF Y-up. We load in the file's native space here and DO NOT rotate;
the up-axis constant is calibrated in Phase 3 (spec/03) so import and export
agree. If a mesh rigs "sideways", this is the first suspect.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np
import trimesh

from .vendor.rig_package.info.asset import Asset

PathLike = Union[str, Path]
SUPPORTED_EXT = {".glb", ".gltf", ".obj", ".ply", ".stl"}


@dataclass
class LoadedMesh:
    """Merged mesh arrays, matching the fields BpyParser.load produced."""

    vertices: np.ndarray        # (N, 3) float32
    faces: np.ndarray           # (F, 3) int64
    vertex_normals: np.ndarray  # (N, 3) float32
    face_normals: np.ndarray    # (F, 3) float32
    vertex_bias: np.ndarray     # (P,) cumulative vertex counts per part
    face_bias: np.ndarray       # (P,) cumulative face counts per part
    mesh_names: List[str]       # (P,) part names, in load order

    @property
    def num_parts(self) -> int:
        return len(self.mesh_names)


def _iter_parts(loaded) -> List[tuple]:
    """Yield (name, Trimesh) parts in a stable order, with transforms applied.

    A glb with several meshes loads as a ``trimesh.Scene``; a single mesh loads
    as a ``trimesh.Trimesh``. For a scene we walk the graph so each geometry is
    placed by its node transform (as Blender would), and concatenate in graph
    order — reproducing upstream's "iterate mesh objects and stack" behavior.
    """
    if isinstance(loaded, trimesh.Trimesh):
        return [("mesh", loaded)]

    if isinstance(loaded, trimesh.Scene):
        parts = []
        for node_name in loaded.graph.nodes_geometry:
            transform, geom_name = loaded.graph[node_name]
            geom = loaded.geometry[geom_name]
            if not isinstance(geom, trimesh.Trimesh):
                continue  # skip non-mesh geometry (points, paths)
            part = geom.copy()
            part.apply_transform(transform)
            parts.append((geom_name or node_name, part))
        if not parts:
            raise ValueError("no mesh geometry found in scene")
        return parts

    raise TypeError(f"unsupported trimesh load result: {type(loaded)}")


def load_mesh(path: PathLike) -> LoadedMesh:
    """Load a mesh file into merged arrays matching upstream's mesh fields.

    Concatenates all mesh parts (offsetting faces by the running vertex count)
    and recomputes normals from the merged mesh with trimesh — as upstream did.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"file does not exist: {path}")
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"unsupported type: {ext} (supported: {sorted(SUPPORTED_EXT)})")

    # process=False / maintain_order=True: do not merge/reorder vertices, so the
    # arrays stay aligned with the file (upstream used the same trimesh flags).
    loaded = trimesh.load(path, process=False, maintain_order=True)
    parts = _iter_parts(loaded)

    vertices_list, faces_list, mesh_names, vertex_bias, face_bias = [], [], [], [], []
    cur_v, cur_f = 0, 0
    for name, part in parts:
        v = np.asarray(part.vertices, dtype=np.float32)
        f = np.asarray(part.faces, dtype=np.int64)
        vertices_list.append(v)
        faces_list.append(f + cur_v)  # offset faces into the merged vertex array
        mesh_names.append(str(name))
        cur_v += v.shape[0]
        cur_f += f.shape[0]
        vertex_bias.append(cur_v)
        face_bias.append(cur_f)

    vertices = np.vstack(vertices_list).astype(np.float32)
    faces = np.vstack(faces_list).astype(np.int64)

    merged = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, maintain_order=True)
    vertex_normals = np.asarray(merged.vertex_normals, dtype=np.float32)
    face_normals = np.asarray(merged.face_normals, dtype=np.float32)

    return LoadedMesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=vertex_normals,
        face_normals=face_normals,
        vertex_bias=np.array(vertex_bias, dtype=np.int64),
        face_bias=np.array(face_bias, dtype=np.int64),
        mesh_names=mesh_names,
    )


def load_asset(path: PathLike, cls: str = "articulation") -> Asset:
    """Load a mesh file into an unrigged ``Asset`` ready for inference.

    Populates the same mesh fields ``BpyParser.load`` produced, so the downstream
    transform + model code is unchanged. ``skin``/armature fields are left unset
    (armature import is Phase 6).
    """
    m = load_mesh(path)
    return Asset(
        vertices=m.vertices,
        faces=m.faces,
        vertex_normals=m.vertex_normals,
        face_normals=m.face_normals,
        vertex_bias=m.vertex_bias,
        face_bias=m.face_bias,
        mesh_names=m.mesh_names,
        cls=cls,
        path=str(path),
    )
