#!/bin/bash
# SD-Dreamer on one task with the settings of the paper (the commands of scripts/make_jobs.py).
# The full grid, every model on every task, is written by scripts/make_jobs.py and run by scripts/queue.py.
#
# usage: bash scripts/train.sh <task> [seed] [logdir]
#   RL          bsuite_memlen-{8,16,32,64}      BSuite memory length
#               bsuite_memlen-{51,71,101}       BSuite memory length, protocol of R2I
#               popgym_RepeatPrevious{Easy,Medium}
#               dmc_{walker_walk,cheetah_run,cartpole_swingup}   DMC from proprioception
#               dmcpix_walker_walk              DMC walker from pixels
#   probes      bsuite_memlen-{8,16,32,64,101}-recall            world model alone, cue recall
#               memory_maze_9x9                 Memory Maze offline probing (data/memory_maze_9x9_official)
set -eu
cd "$(dirname "$0")/.."
task=$1; seed=${2:-0}; logdir=${3:-runs/sd-dreamer/$task/s$seed}
PY=${PYTHON:-python}
OURS="ours resonanthalf policymem aclowrank somathermo pdfeedback pdch2"  # the final model
WM="ours resonanthalf somathermo pdfeedback pdch2"                        # its world model (probes)
BSUITE="--train_ratio 64 --reward_head symlog_mse --env_seed_per_env True"
case $task in
  bsuite_memlen-8)   cmd="dreamer.py --configs bsuite $OURS rewardbyact --steps 150000 --eval_every 10000 --batch_length 32 $BSUITE";;
  bsuite_memlen-16)  cmd="dreamer.py --configs bsuite $OURS rewardbyact --steps 200000 --eval_every 10000 $BSUITE";;
  bsuite_memlen-32)  cmd="dreamer.py --configs bsuite $OURS rewardbyact --steps 300000 --eval_every 10000 $BSUITE";;
  bsuite_memlen-64)  cmd="dreamer.py --configs bsuite $OURS rewardbyact --steps 500000 --eval_every 10000 --batch_length 80 $BSUITE";;
  bsuite_memlen-51)  cmd="dreamer.py --configs bsuite $OURS rewardbyact --steps 1e6 --eval_every 20000 --batch_length 64 $BSUITE";;
  bsuite_memlen-71)  cmd="dreamer.py --configs bsuite $OURS rewardbyact --steps 1e6 --eval_every 20000 --batch_length 80 $BSUITE";;
  bsuite_memlen-101) cmd="dreamer.py --configs bsuite $OURS rewardbyact --steps 1e6 --eval_every 20000 --batch_length 128 $BSUITE";;
  popgym_*)          cmd="dreamer.py --configs popgym $OURS --steps 2e6 --train_ratio 64 --envs 8 --batch_length 64 --env_seed_per_env True --eval_every 5e4";;
  dmc_*)             cmd="dreamer.py --configs dmc_proprio $OURS --steps 500000 --train_ratio 512 --eval_every 2e4 --env_seed_per_env True";;
  dmcpix_*)          export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa; task=dmc_${task#dmcpix_}
                     cmd="dreamer.py --configs dmc_vision $OURS --steps 1e6 --train_ratio 512 --eval_every 2e4 --env_seed_per_env True";;
  bsuite_memlen-*-recall)
    bl=$(case $task in *-8-*) echo "--batch_length 32";; *-64-*) echo "--batch_length 80";; *-101-*) echo "--batch_length 128";; esac)
    cmd="probe_wm.py --configs bsuite $WM --updates 12000 --eval_every 500 --reward_head symlog_mse $bl";;
  memory_maze_9x9)
    exec $PY probe_mm.py --configs mmprobe ours resonanthalf policymem somathermo pdfeedback pdch2 \
      --data data/memory_maze_9x9_official --seed "$seed" --mm_updates 1000000 --mm_eval_every 10000 --logdir "$logdir";;
  *) echo "unknown task: $task" >&2; exit 1;;
esac
exec $PY $cmd --task "$task" --seed "$seed" --logdir "$logdir"
