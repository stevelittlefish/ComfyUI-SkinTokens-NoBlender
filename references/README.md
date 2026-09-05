# Reference projects

Projects kept here to **read, not to depend on**. Everything in this folder is
**cloned and gitignored** — nothing here is part of the project, and nothing
here should ever be imported or vendored from at runtime.

```sh
./pull.sh      # clone anything missing, pull anything present
```

## What's here

- **`SkinTokens/`** — upstream torch reference implementation (VAST-AI). The
  model, tokenizer, transforms, FSQ, SkinVAE, sampling, and normalization are
  copied verbatim into our package; only its Blender-based glb I/O is replaced.
- **`skin-tokens.cpp/`** — C++/GGML port (LocalAI). Reference only; it is the
  spec for the pure-Python glb export, the structural relabeler, and retargeting.
- **`ComfyUI-SkinTokens/`** — prior-art ComfyUI node pack (`Aero-Ex`). Shows one
  way to wire SkinTokens into ComfyUI (uses Blender). See `spec/08-prior-art.md`.
