"""Phase 2 local tests: pure-Python glb/gltf import (Gate A, self-consistency half).

No GPU, no Blender. Blender-parity (vs BpyParser.load) is a committed golden
fixture task for later; here we verify the importer's internal correctness:
round-trip counts, multi-part concatenation + bias, normals, and Asset fields.
"""

import pytest
import trimesh

from skintokens import glb_io


def _write_box_glb(path, extents=(1.0, 2.0, 3.0)):
    trimesh.creation.box(extents=extents).export(path)


def _write_two_box_scene(path):
    a = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    b = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    b.apply_translation([5.0, 0.0, 0.0])
    scene = trimesh.Scene()
    scene.add_geometry(a, geom_name="box_a", node_name="box_a")
    scene.add_geometry(b, geom_name="box_b", node_name="box_b")
    scene.export(path)


def test_load_single_mesh(tmp_path):
    p = tmp_path / "box.glb"
    _write_box_glb(p)
    ref = trimesh.creation.box(extents=(1.0, 2.0, 3.0))

    m = glb_io.load_mesh(p)

    assert m.vertices.shape[1] == 3 and m.faces.shape[1] == 3
    assert m.vertices.shape[0] == len(ref.vertices)
    assert m.faces.shape[0] == len(ref.faces)
    assert m.vertex_normals.shape == m.vertices.shape
    assert m.face_normals.shape == m.faces.shape
    assert m.num_parts == 1
    assert m.vertex_bias.tolist() == [m.vertices.shape[0]]
    assert m.face_bias.tolist() == [m.faces.shape[0]]


def test_faces_index_in_range(tmp_path):
    p = tmp_path / "box.glb"
    _write_box_glb(p)
    m = glb_io.load_mesh(p)
    assert m.faces.min() >= 0
    assert m.faces.max() < m.vertices.shape[0]


def test_load_multipart_scene(tmp_path):
    p = tmp_path / "two.glb"
    _write_two_box_scene(p)

    m = glb_io.load_mesh(p)

    assert m.num_parts == 2
    # cumulative biases are strictly increasing and end at the totals
    assert m.vertex_bias[-1] == m.vertices.shape[0]
    assert m.face_bias[-1] == m.faces.shape[0]
    assert m.vertex_bias[0] < m.vertex_bias[1]
    # second part's faces were offset into the merged vertex array and stay valid
    assert m.faces.max() < m.vertices.shape[0]
    assert set(m.mesh_names) == {"box_a", "box_b"}


def test_multipart_second_part_faces_offset(tmp_path):
    p = tmp_path / "two.glb"
    _write_two_box_scene(p)
    m = glb_io.load_mesh(p)

    v_split = int(m.vertex_bias[0])
    f_split = int(m.face_bias[0])
    second_faces = m.faces[f_split:]
    # every vertex referenced by the second part lives in the second block
    assert second_faces.min() >= v_split


def test_load_asset_fields(tmp_path):
    p = tmp_path / "box.glb"
    _write_box_glb(p)

    asset = glb_io.load_asset(p)

    assert asset.N == asset.vertices.shape[0]
    assert asset.F == asset.faces.shape[0]
    assert asset.vertex_normals.shape == asset.vertices.shape
    assert asset.cls == "articulation"
    assert asset.path == str(p)
    assert asset.skin is None  # armature import is Phase 6


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        glb_io.load_mesh(tmp_path / "nope.glb")


def test_unsupported_extension(tmp_path):
    p = tmp_path / "mesh.xyz"
    p.write_text("")
    with pytest.raises(ValueError):
        glb_io.load_mesh(p)


def test_loaded_asset_runs_through_transform(tmp_path):
    # The imported Asset must feed the inference transform (sampler) cleanly.
    from skintokens.vendor.data.sampler import SamplerMix
    from skintokens.vendor.data.transform import Transform

    p = tmp_path / "box.glb"
    _write_box_glb(p)
    asset = glb_io.load_asset(p)

    Transform(sampler=SamplerMix(num_samples=128, num_vertex_samples=0)).apply(asset)
    assert asset.sampled_vertices.shape == (128, 3)
    assert asset.sampled_normals.shape == (128, 3)
