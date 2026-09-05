# Test mesh fixtures

Sample meshes used as inputs for the GPU/server tests (rig import → inference →
export, and the ComfyUI node pipeline). Kept small so they can live in the repo.

## dummy.glb

A humanoid character used as a rigging input. Sourced from **Adobe Mixamo**
(https://www.mixamo.com/), used under Adobe's royalty-free license for use in
projects. Textures downscaled 4096² → 1024² to keep the file small; UVs, materials
(base color / normal / metallic-roughness) and the full geometry (20,568 verts /
37,668 faces) are unchanged, so it also exercises texture transfer (Phase 6).

- Original: ~43 MB (four 4096² PNG maps). This copy: ~3.7 MB.
- Being humanoid, it also validates the relabeler end to end (produces `mixamorig:*`).
