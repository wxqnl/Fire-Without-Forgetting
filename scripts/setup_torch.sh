#!/usr/bin/env bash
set -euo pipefail
cd /data/Minko
root=/data/Minko/fire-without-forgetting
/root/.local/bin/uv venv --python /root/miniconda3/bin/python "$root/envs/torch"
/root/.local/bin/uv pip install --python "$root/envs/torch/bin/python" -r "$root/envs/torch-requirements.txt"
/root/.local/bin/uv pip freeze --python "$root/envs/torch/bin/python" > "$root/envs/torch-freeze.txt"
