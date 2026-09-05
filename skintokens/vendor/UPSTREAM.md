# Vendored upstream: SkinTokens (torch core)

This directory is a verbatim copy of the `src/` tree from upstream **SkinTokens**,
with the Blender/server pieces removed. Do not hand-edit vendored files except for
the documented import fixes below; re-vendoring should reproduce them.

- Upstream: https://github.com/VAST-AI-Research/SkinTokens
- Commit: `273b691d35989d71cd17ff2895fdc735097b92d1` ("modify post-sampling strategy", 2026-05-12)
- Copied from: `references/SkinTokens/src/` → `skintokens/vendor/`

## Changes applied on vendoring

1. **Removed** `rig_package/parser/bpy.py` — the Blender (`bpy`) mesh/armature
   parser + exporter. Its behavior is reimplemented in pure Python in
   `skintokens/glb_io.py` (see `spec/02`, `spec/03`).
2. **Removed** `server/bpy_server.py` — the bottle/tornado HTTP server. We call
   functions directly; no server.
3. **Added** `data/__init__.py` (empty) — upstream relied on it being an implicit
   namespace package; made explicit so it imports as a regular package here.
4. **Fixed** `model/skin_vae_model.py`: `from src.rig_package.info.asset` →
   `from ..rig_package.info.asset` (the only absolute intra-package import).

5. **Patched** `model/tokenrig.py`: the Qwen backbone was built with a hardcoded
   `attn_implementation="flash_attention_2"`. Now it uses `"flash_attention_2"`
   only if `flash_attn` imports, else `"sdpa"` — so the model loads on servers
   (and CPU dev boxes) without flash-attn. Same policy as the SDPA fallback in (3-ish).

6. **Added** `configs/skeleton/{mixamo,vroid}.yaml` — copied verbatim from
   upstream `configs/`. The checkpoint's transform config references these skeleton
   part-order tables by a CWD-relative path; `model_loader._rewrite_skeleton_paths`
   rewrites those to this vendored location at load time.

## Known lazy references to the removed `bpy.py`

These are inside function bodies and do NOT run at import time. They will be
rewired to the pure-Python exporter in later phases:

- `model/tokenrig.py` (`predict_step` / make_asset) — `from ..rig_package.parser.bpy import BpyParser`
- `data/datapath.py` (`BpyLazyAsset.load`) — same import
