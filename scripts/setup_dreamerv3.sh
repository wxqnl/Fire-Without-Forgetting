#!/usr/bin/env bash
set -euo pipefail
cd /data/Minko
root=/data/Minko/fire-without-forgetting
[ -x "$root/envs/dreamerv3/bin/python" ] || /root/.local/bin/uv venv --python /root/miniconda3/bin/python "$root/envs/dreamerv3"
/root/.local/bin/uv pip install --python "$root/envs/dreamerv3/bin/python" -r "$root/envs/dreamerv3-requirements.txt"
/root/.local/bin/uv pip freeze --python "$root/envs/dreamerv3/bin/python" > "$root/envs/dreamerv3-freeze.txt"
