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

## Local dev environment

- **Always use the venv** at `./.venv` (gitignored). It was created with
  `uv venv` and runs **CPython 3.12** (ComfyUI-compatible; the server's exact
  version is TBD once we have access). **Never install into system Python.**
- Install deps: `uv pip install -r requirements-dev.txt` then
  `uv pip install -e . --no-deps` (all into `.venv`).
- `requirements.txt` = runtime deps for ComfyUI Manager (torch NOT listed —
  ComfyUI provides it). `requirements-dev.txt` adds a CPU torch build + pytest
  for GPU-less local work.
- Run tests: `source .venv/bin/activate && python -m pytest -q`.

## Testing strategy (no GPU / no ComfyUI here)

- Do as much as possible with local pure-Python `pytest` — especially the
  highest-risk paths: **glb export (Gate C)** and the **relabeler (Gate D)**.
- Blender-parity checks become committed golden fixtures (no `bpy` locally).
- GPU/server-only gates (inference B, ComfyUI/VRAM F, end-to-end E) are deferred
  until we wire up access to the Comfy server (https://comfy.seaslug.ai/).
- User provides test data (sample meshes/rigs) on request.

## Vendored upstream

`skintokens/vendor/` is the upstream SkinTokens torch core, copied verbatim minus
Blender/server. See `skintokens/vendor/UPSTREAM.md` for the commit and the exact
edits applied (import fixes, flash-attn → SDPA fallback, CPU guards). Don't
hand-edit vendored files beyond those documented portability fixes.

## Keep TODO.md current

`spec/TODO.md` is the living build tracker. **Whenever a task lands (or gets
blocked), update it in the same commit as the work** — tick `[x]` done,
`[~]` partial/blocked (say why), and add a short parenthetical noting what was
actually delivered or where it lives. Don't let it drift from reality.

## Git workflow

- **Commit directly to `main`.**
- **Push after every commit.**
- **Never rewrite history** — no amend, no rebase, no force-push.
- Fix mistakes with follow-up commits, not by editing past commits. Only if a
  whole commit is genuinely wrong, revert it (a new commit).
