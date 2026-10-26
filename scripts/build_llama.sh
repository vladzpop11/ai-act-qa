#!/usr/bin/env bash
# Build llama.cpp's server for the NVIDIA GPU of this machine (Kaggle), from the release pinned
# in configs/config.yaml. The prebuilt CUDA binaries need glibc 2.38 (Ubuntu 24.04), so building
# is the reliable route; only the GPU's own architecture is compiled, which takes ~10 minutes.
# Usage: scripts/build_llama.sh TAG OUT_DIR  -> OUT_DIR/llama-server
set -euo pipefail
tag="$1"; out="$2"
# The CUDA compiler is often installed but not on PATH (Kaggle: /usr/local/cuda/bin).
if ! command -v nvcc >/dev/null; then
  for dir in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
    if [ -x "$dir/nvcc" ]; then export PATH="$dir:$PATH"; break; fi
  done
fi
command -v nvcc >/dev/null || { echo "no CUDA compiler (nvcc) found"; exit 1; }
command -v cmake >/dev/null || pip install -q cmake
echo "nvcc: $(nvcc --version | tail -1); $(cmake --version | head -1); $(nproc) cores"
export CUDACXX="$(command -v nvcc)"
arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "llama.cpp $tag for compute capability $arch"
src="$out/llama.cpp"
if [ ! -d "$src" ]; then
  git -c advice.detachedHead=false clone --depth 1 --branch "$tag" https://github.com/ggml-org/llama.cpp "$src"
fi
# Static libraries, so the one binary can be copied; no HTTPS (the server only listens locally).
# GGML_CUDA_NO_VMM: Kaggle's toolkit has no libcuda.so to link against at build time (CMake:
# "Target ggml-cuda links to CUDA::cuda_driver but the target was not found", 2026-10-26).
# Without virtual memory management, llama.cpp does not link the driver library directly; the
# CUDA runtime still loads it when the server starts. Only GPU memory pooling is affected.
cmake -S "$src" -B "$src/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$arch" \
  -DGGML_CUDA_NO_VMM=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$src/build" --target llama-server -j "$(nproc)"
cp "$src/build/bin/llama-server" "$out/llama-server"
"$out/llama-server" --version
