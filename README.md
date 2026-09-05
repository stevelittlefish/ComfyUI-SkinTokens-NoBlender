# ComfyUI-SkinTokens-NoBlender

SkinTokens rigging and skinning nodes for ComfyUI — no Blender required on the server.

Wraps the SkinTokens / TokenRig auto-rigger: a static mesh (`.glb`) goes in, a
rigged `.glb` (skeleton + skin weights) comes out, with an optional step that
relabels the skeleton with Mixamo bone names. glb import/export is pure Python —
no Blender / `bpy` dependency.

**Status:** early development. See [`spec/`](spec/) for the full build spec and
[`spec/TODO.md`](spec/TODO.md) for the phased task list.
