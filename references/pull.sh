#!/usr/bin/env bash
set -euo pipefail

REPOS=(
  "git@github.com:VAST-AI-Research/SkinTokens.git"
  "git@github.com:localai-org/skin-tokens.cpp.git"
  "git@github.com:Aero-Ex/ComfyUI-SkinTokens.git"
)

cd "$(dirname "$0")"

for repo in "${REPOS[@]}"; do
  dir="${repo##*/}"; dir="${dir%.git}"
  if [ -d "$dir" ]; then
    echo "Pulling $dir..."
    git -C "$dir" pull
  else
    echo "Cloning $repo..."
    git clone "$repo"
  fi
done
