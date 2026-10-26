#!/usr/bin/env bash
# Build llama.cpp's server for the NVIDIA GPU of this machine (Kaggle), from the release pinned
# in configs/config.yaml. The prebuilt CUDA binaries need glibc 2.38 (Ubuntu 24.04), so building
# is the reliable route; only the GPU's own architecture is compiled, which takes ~10 minutes.
# Usage: scripts/build_llama.sh TAG OUT_DIR  -> OUT_DIR/llama-server
set -euo pipefail
tag="$1"; out="$2"
arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "llama.cpp $tag for compute capability $arch"
src="$out/llama.cpp"
if [ ! -d "$src" ]; then
  git clone --depth 1 --branch "$tag" https://github.com/ggml-org/llama.cpp "$src"
fi
# Static libraries, so the one binary can be copied; no HTTPS (the server only listens locally).
cmake -S "$src" -B "$src/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$arch" \
  -DBUILD_SHARED_LIBS=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$src/build" --target llama-server -j "$(nproc)"
cp "$src/build/bin/llama-server" "$out/llama-server"
"$out/llama-server" --version
