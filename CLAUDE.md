# CLAUDE.md

Guidance for Claude Code working in this repo.

## What this is

A pure-Python ComfyUI node pack wrapping the SkinTokens / TokenRig auto-rigger:
static mesh (`.glb`) → rigged `.glb` (skeleton + skin weights), with an optional
Mixamo relabel step. The whole point is doing glb I/O **without Blender / `bpy`**.
See `spec/00-overview.md`.

## The spec (read these — don't go searching)

The `spec/` directory is a complete, self-contained build spec. Start with the
overview and TODO, then the doc relevant to the phase you're on.

- [spec/README.md](spec/README.md) — how the spec is organized.
- [spec/00-overview.md](spec/00-overview.md) — what we're building and why; glossary.
- [spec/01-architecture.md](spec/01-architecture.md) — decisions, rejected alternatives, VRAM/ComfyUI integration.
- [spec/02-scope-bpyparser-port.md](spec/02-scope-bpyparser-port.md) — exactly what to port vs copy verbatim.
- [spec/03-glb-export-spec.md](spec/03-glb-export-spec.md) — **the critical correctness path**: bone convention, inverse bind matrices, top-4 weight packing.
- [spec/04-relabeler-spec.md](spec/04-relabeler-spec.md) — structural humanoid relabeler.
- [spec/05-comfyui-node-spec.md](spec/05-comfyui-node-spec.md) — node interface, model loading, ModelPatcher, repo layout.
- [spec/06-references.md](spec/06-references.md) — what to copy, what to reference, links, weights.
- [spec/07-validation.md](spec/07-validation.md) — validation gates (A–F).
- [spec/08-prior-art.md](spec/08-prior-art.md) — prior art (`Aero-Ex/ComfyUI-SkinTokens`).
- [spec/TODO.md](spec/TODO.md) — **ordered, phased build task list.** Work phases in order.
- [spec/RIGGING_ANIMATION_NOTES.md](spec/RIGGING_ANIMATION_NOTES.md) — pipeline findings.

## Reference code (read, never depend on)

`references/` holds cloned upstream projects — gitignored, read-only, never
imported or vendored from directly at runtime. See [references/README.md](references/README.md).

- `references/SkinTokens/` — upstream torch reference implementation (VAST-AI).
- `references/skin-tokens.cpp/` — C++/GGML port; the spec for pure-Python glb export + relabeler.
- `references/ComfyUI-SkinTokens/` — prior-art ComfyUI node pack (`Aero-Ex`).

Run `references/pull.sh` to clone/update them.

## Git workflow

- **Commit directly to `main`.**
- **Push after every commit.**
- **Never rewrite history** — no amend, no rebase, no force-push.
- Fix mistakes with follow-up commits, not by editing past commits. Only if a
  whole commit is genuinely wrong, revert it (a new commit).
