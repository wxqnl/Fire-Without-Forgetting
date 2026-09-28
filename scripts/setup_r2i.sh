#!/usr/bin/env bash
set -euo pipefail
cd /data/Minko
root=/data/Minko/fire-without-forgetting
[ -x "$root/envs/r2i/bin/python" ] || /root/.local/bin/uv venv --python /root/miniconda3/bin/python "$root/envs/r2i"
/root/.local/bin/uv pip install --python "$root/envs/r2i/bin/python" -r "$root/envs/r2i-requirements.txt" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
/root/.local/bin/uv pip freeze --python "$root/envs/r2i/bin/python" > "$root/envs/r2i-freeze.txt"
