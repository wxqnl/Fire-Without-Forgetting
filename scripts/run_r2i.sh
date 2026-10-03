#!/bin/bash
# R2I (Samsami et al., ICLR 2024), official code (external/Recall2Imagine, commit 6e317e7) and its
# POPGym configuration, unchanged except: run.steps (our 2M budget), replay_size 2e6 (the default
# FIFO buffer of 1e7 never fills within 2M steps, so sampling is the same) and jax.prealloc False
# (the GPUs are shared). Logs the training-episode score, as in the R2I paper.
# usage: scripts/run_r2i.sh <popgym task, e.g. RepeatPreviousEasy> <gpu> <logdir>
set -u
task=$1; gpu=$2; logdir=$3
mkdir -p "$logdir"
cd "$(dirname "$0")/../external/Recall2Imagine"
echo $$ > "$logdir/.pid"; echo "gpu$gpu" > "$logdir/.gpu"
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled \
  /home/user01/miniconda3/envs/r2i/bin/python recall2imagine/train.py \
  --configs popgym --task "gym_popgym:popgym-$task-v0" --logdir "$logdir" \
  --run.steps 2e6 --replay_size 2e6 --jax.prealloc False --seed 0 > "$logdir/stdout.log" 2>&1
echo $? > "$logdir/.exit"
