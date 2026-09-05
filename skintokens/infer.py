"""Run rigging inference on an in-memory mesh.

Phase 1 scope (see spec/TODO): given vertices/faces/normals already in memory,
build the upstream ``Asset``, run ``predict_step``, and return the rigged
``Asset`` (joints, parents, dense skin weights). glb import/export is Phase 2/3;
this module never touches files.

``build_asset`` and the transform pipeline are pure numpy/scipy and run on CPU.
The actual ``predict_step`` needs the ~14 GB model on a GPU (Gate B, server-side).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .model_loader import SkinTokensModel
from .vendor.model.spec import ModelInput
from .vendor.rig_package.info.asset import Asset

# Generation defaults, matching upstream demo.py.
DEFAULT_GENERATE_KWARGS = dict(
    max_length=2048,
    top_k=5,
    top_p=0.95,
    temperature=1.0,
    repetition_penalty=2.0,
    num_return_sequences=1,
    num_beams=10,
    do_sample=True,
)


def build_asset(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: Optional[np.ndarray] = None,
    cls: str = "articulation",
    path: Optional[str] = None,
    mesh_name: str = "mesh",
) -> Asset:
    """Build an unrigged ``Asset`` from in-memory mesh arrays.

    Reproduces the mesh fields that upstream ``BpyParser.load`` populated
    (vertices, faces, vertex/face normals, mesh_names, cls, path) so the
    downstream transform + model code is unchanged.

    ``vertices``: (N, 3) float; ``faces``: (F, 3) int; ``normals``: optional
    (N, 3) vertex normals — if omitted they are computed with trimesh.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must be (N, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (F, 3), got {faces.shape}")

    asset = Asset(
        vertices=vertices,
        faces=faces,
        mesh_names=[mesh_name],
        cls=cls,
        path=path,
    )
    # Face normals are always needed by the sampler; vertex normals too unless
    # provided. build_normals fills both from trimesh.
    asset.build_normals()
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float32)
        if normals.shape != vertices.shape:
            raise ValueError(
                f"normals must match vertices shape {vertices.shape}, got {normals.shape}"
            )
        asset.vertex_normals = normals
    return asset


def prepare_asset(bundle: SkinTokensModel, asset: Asset) -> Asset:
    """Apply the model's predict transform (normalize + surface sampling).

    Mutates and returns ``asset``; afterwards ``sampled_vertices`` /
    ``sampled_normals`` are populated and ``vertices`` are in the model's
    normalized space. Pure CPU — testable without the model weights.
    """
    bundle.transform.apply(asset)
    return asset


def rig_mesh(
    bundle: SkinTokensModel,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: Optional[np.ndarray] = None,
    cls: str = "articulation",
    generate_kwargs: Optional[dict] = None,
    use_skeleton: bool = False,
) -> Asset:
    """Rig an in-memory mesh: mesh arrays -> rigged ``Asset``.

    The returned Asset has ``joints``/``parents`` (skeleton) and dense
    ``skin`` weights over the original vertices, in the model's normalized space.
    ``use_skeleton`` (skin-only against an existing armature) is Phase 6.
    """
    if use_skeleton:
        raise NotImplementedError("use_skeleton (skin-only) is Phase 6")

    asset = build_asset(vertices, faces, normals=normals, cls=cls)
    return rig_asset(bundle, asset, generate_kwargs=generate_kwargs)


def rig_glb(
    bundle: SkinTokensModel,
    path,
    cls: str = "articulation",
    generate_kwargs: Optional[dict] = None,
    use_skeleton: bool = False,
) -> Asset:
    """Rig a mesh loaded from a glb/gltf/obj file: path -> rigged ``Asset``.

    Convenience wrapper: pure-Python glb import (Phase 2) + inference. Export back
    to a skinned glb is Phase 3.
    """
    if use_skeleton:
        raise NotImplementedError("use_skeleton (skin-only) is Phase 6")

    from .glb_io import load_asset

    asset = load_asset(path, cls=cls)
    return rig_asset(bundle, asset, generate_kwargs=generate_kwargs)


def rig_glb_to_file(
    bundle: SkinTokensModel,
    in_path,
    out_path,
    cls: str = "articulation",
    generate_kwargs: Optional[dict] = None,
) -> Asset:
    """Full pipeline: glb in -> rigged skinned glb out. Returns the rigged Asset.

    Import (Phase 2) + inference (Phase 1) + skinned export (Phase 3). Texture/
    material transfer onto the original glb is Phase 6; this writes the rigged
    mesh with a default material.
    """
    from .glb_io import export_glb

    rigged = rig_glb(bundle, in_path, cls=cls, generate_kwargs=generate_kwargs)
    export_glb(rigged, out_path)
    return rigged


def rig_asset(
    bundle: SkinTokensModel,
    asset: Asset,
    generate_kwargs: Optional[dict] = None,
) -> Asset:
    """Rig an already-built unrigged ``Asset`` -> rigged ``Asset``.

    Applies the predict transform (in place) and runs ``predict_step``. The
    returned Asset has ``joints``/``parents`` (skeleton) and dense ``skin``
    weights over the original vertices, in the model's normalized space.
    """
    prepare_asset(bundle, asset)

    gen = dict(DEFAULT_GENERATE_KWARGS)
    if generate_kwargs:
        gen.update(generate_kwargs)

    verts = torch.from_numpy(np.asarray(asset.sampled_vertices)).float().to(bundle.device)
    norms = torch.from_numpy(np.asarray(asset.sampled_normals)).float().to(bundle.device)

    batch = {
        "vertices": verts,
        "normals": norms,
        "cls": [asset.cls],
        "model_input": [ModelInput(asset=asset)],
        "generate_kwargs": gen,
    }

    with torch.no_grad():
        results = bundle.model.predict_step(batch, make_asset=True)["results"]

    rigged = results[0].asset
    if rigged is None:
        raise RuntimeError("predict_step returned no asset")
    return rigged
