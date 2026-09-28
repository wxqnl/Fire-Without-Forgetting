#!/usr/bin/env bash
set -euo pipefail
cd /data/Minko
root=/data/Minko/fire-without-forgetting
for name in original_tau2_seed0 persistent_tau2_seed0 no_dendrite_reset_tau2_seed0 persistent_no_dendrite_reset_tau2_seed0; do
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' "$root/envs/torch/bin/python" -u "$root/scripts/probe_memory.py" --code "$root/code/spiking-wm" --run "$root/runs/phase0_initial/$name" --reference "$root/runs/phase0_initial/original_tau2_seed0/eval_eps" --output "$root/docs/probes/$name.json" > "$root/logs/probe_$name.log" 2>&1
done
