# Auto-Rig → Auto-Animate Pipeline — Findings

_Last updated: 2026-09-05_

## Goal

Fully automated, no-user-interaction pipeline running against local AI services on the
LAN (API calls only):

```
Trellis.2 (mesh) → SkinTokens (auto-rig) → Kimodo (motion) → animated character
```

The blocker: SkinTokens rigs a mesh with a generic, unnamed skeleton, but the Kimodo
animation stage expects a **Mixamo-named** skeleton. This document records how each
tool works and the solution.

---

## The files we started with

| File | What it is |
|------|-----------|
| `sci-fi-dude.glb` | Trellis.2 mesh, **unrigged** |
| `sci-fi-dude-rigged.glb` | Same mesh after **SkinTokens** auto-rig (52 bones, `bone_0…bone_51`) |
| `test.fbx` | An Adobe/Mixamo character (mesh "Ch50"), Mixamo skeleton |
| `kimodo_animated_test.fbx` | `test.fbx` animated via Kimodo (49-bone Mixamo skeleton, 150-frame clip) |
| `peasant.glb`, `knight.glb`, `robot_industrial.glb` | Extra SkinTokens rigs used to prove determinism (34 / 34 / 41 bones) |

---

## How the tools actually work

### Kimodo (`/home/steve/git/ComfyUI-Kimodo-Enhanced/`)
- NVIDIA **kinematic motion diffusion** model. Generates motion **from a text prompt**
  on its own internal skeleton (**SOMA**, 30 joints). It knows nothing about your character.
- `kimodo_retarget_fbx.py` then **retargets** that SOMA motion onto a target character
  and writes an animated FBX.
- **Key insight:** the retarget engine (`retarget_animation()`) is skeleton-agnostic.
  It takes a source skeleton, a target skeleton, and a `mapping` dict of
  source-bone-name → target-bone-name, and computes offsets from each skeleton's actual
  rest pose. The **only** Mixamo-specific thing in the whole file is one hardcoded dict,
  `SOMA_TO_MIXAMO` (line 146). It auto-scales by skeleton height and can apply a yaw offset.
- Gaps if we ever target `bone_N` directly: it loads/saves **FBX** (our rigs are `.glb`),
  and `_skeleton_height()` / root detection rely on named bones.

### SkinTokens = TokenRig (`SkinTokens/`)
- VAST-AI autoregressive rigger, successor to **UniRig**. Mesh in → skeleton + skin
  weights out as one token sequence. ~14 GB GPU. Runs as a Python process plus a
  `bpy_server` (Blender) for import/export — already headless/LAN-friendly.
- `bone_N` names are just a **fallback index** (`f"bone_{i}"`); the model does not emit
  semantic names.
- Ships skeleton **templates** (`configs/skeleton/mixamo.yaml`, `vroid.yaml`) with a
  canonical part order, and a `--use_skeleton` mode (generate skin only, against a
  supplied skeleton). The demo does **not** pass a class template, which is why our
  outputs came out as `bone_N`.

---

## The decisive finding: the skeleton is structurally deterministic

The generated skeletons are **not** fixed-index (bone counts vary: 52 / 34 / 34 / 41 —
the variation is entirely **finger count**). But the **topology and depth-first traversal
order are identical every time**.

The first 10 bones are byte-for-byte the same across all four characters:

```
bone_0 Hips → bone_1 Spine → bone_2 Spine1 → bone_3 Spine2(branch)
bone_4 Neck → bone_5 Head
bone_6 LeftShoulder → 7 LeftArm → 8 LeftForeArm → 9 LeftHand
```

…then the right-arm chain, then the **legs are always the last 8 bones** (the only
downward children of Hips). Only the finger fan-out off each Hand varies.

### Structural labeler rules (index-independent)
- **Hips** = the root (no parent).
- Hips' children split by height: the one going **up** is **Spine**; the two going
  **down** (±X) are the **legs**.
- Follow the spine up until a bone with 3 children = **Spine2 / chest**; earlier bones
  are Spine / Spine1.
- From the chest: center child → **Neck → Head**; +X child → **Left arm**; −X child →
  **Right arm**.
- Walk each limb 4 steps (choosing the deepest branch each step):
  Shoulder→Arm→ForeArm→**Hand**, and UpLeg→Leg→Foot→**ToeBase**.
- Fingers = whatever hangs off each Hand. Variable, and Kimodo's SOMA skeleton barely
  uses fingers, so they can be ignored for animation.

This labels the **22 core body bones** — exactly the ones Kimodo drives — and was
verified correct on all four characters despite their different finger counts.

---

## The solution

> The SkinTokens skeleton is structurally deterministic, so a small headless
> **bone-relabeler** converts any rigged output into a genuine Mixamo character.
> The existing Kimodo/ComfyUI pipeline then animates it **unchanged** — no retargeting
> engine changes, no manual per-model mapping, no Mixamo website.

```
Trellis.2 → SkinTokens (auto-rig) → [relabel bone_N → mixamorig:*] → Kimodo retarget → animated FBX
                                       ↑ new ~40-line headless step
```

### Caveats
1. **Left/Right = ±X** held for all four models (SkinTokens uses a canonical
   orientation). If a first real animation comes out mirrored, swap the L/R rule once.
2. The relabeler must rename the **vertex groups / skin weights**, not just the bones,
   or the mesh detaches from the renamed skeleton.
3. Still needs one end-to-end run through Kimodo on the GPU box to confirm motion lands.
4. Non-humanoid rigs (quadrupeds, etc.) are out of scope for this humanoid labeler.

---

## Decided architecture (2026-09-05)

Rig and animate stages **always run separately** (different ComfyUI workflows) — no
simultaneous VRAM. Kimodo stays in its own fork (`ComfyUI-Kimodo-Enhanced`), edited later
if needed. SkinTokens becomes a **new, separate, pure-Python ComfyUI node pack** (glb in →
rigged glb out, optional relabel), idiomatic and Manager-installable, using ComfyUI's
`ModelPatcher`/`model_management` for load/unload. **No `bpy`, no `bpy_server`, no CLI
subprocess** (a subprocess is invisible to ComfyUI's VRAM accounting and awkward to
distribute via Manager).

Reference repos:
- Torch core (copied verbatim): `~/workspace/ai-rigging/SkinTokens` (upstream).
- `~/workspace/ai-rigging/skin-tokens.cpp` — C++/GGML port; **spec** for pure-Python glb
  export + structural relabel + SOMA30 retarget (`src/glb.cpp`, `src/retarget.cpp`).
  Not the runtime (we want pure Python; speed is not the priority).

### Scope: `bpy` is confined to one file — `src/rig_package/parser/bpy.py` (`BpyParser`), 3 ops
| Op | Python replacement | Notes |
|----|--------------------|-------|
| `load` (mesh) | `trimesh` | trivial; sampling/normalize already pure torch/numpy |
| `load` (armature, `use_skeleton` only) | `pygltflib` skins/nodes | moderate |
| `export` (mesh+skeleton+skin→glb) | `pygltflib` | **hard part — see risk below** |
| `transfer` (`--use_transfer`, keep textures) | `pygltflib`, carry buffers | phase 2, don't skip |

Also swap the `BpyParser` import in `src/model/tokenrig.py:313` (predict_step make_asset).
Relabeler = trivial pure Python over the `joints`/`parents` arrays (no bpy).

### Top correctness risk
`export` bone convention: upstream extrudes bones along **local Y "in accordance with
Blender"** (`asset.py:158`). Exported `inverseBindMatrices` must be consistent with joint
node transforms or the mesh **detaches/deforms when posed** (looks rigged, explodes on
animation). `glb.cpp` is the spec for this convention + the top-4 weight packing
(`group_per_vertex=4`; glTF allows max 4 influences/vertex, model emits dense weights).

## Open / next steps (not started — awaiting go-ahead)
- [ ] Port `BpyParser` (load/export/transfer) to trimesh+pygltflib per `glb.cpp`.
- [ ] Reimplement the relabeler in pure Python (walk joints/parents arrays).
- [ ] Wrap as a ComfyUI node (glb in → rigged glb out, relabel toggle) with ModelPatcher.
- [ ] Validate: same Trellis mesh through torch demo.py vs new node — skeleton/weights parity,
      and re-import keeps mesh attached AND survives posing/animation.
- [ ] End-to-end validation through Kimodo on ai.lemon.com.
</content>
</invoke>
