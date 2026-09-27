#!/bin/bash
# Hard stop for the GPU time limit: releases GPUs 1-6 at a fixed time.
# usage: setsid nohup bash scripts/deadline_stop.sh "2026-09-27 18:50" > runs/deadline.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
STOP_AT="$1"
PY=/home/user01/miniconda3/envs/fwf/bin/python
now=$(date +%s); target=$(date -d "$STOP_AT" +%s)
echo "$(date '+%F %T') watchdog armed, stop at $STOP_AT ($(( (target - now) / 60 )) min)"
sleep $(( target > now ? target - now : 0 ))

echo "$(date '+%F %T') stopping queue and jobs"
pkill -f "envs/fwf/bin/python scripts/queu[e]"
sleep 2
pkill -TERM -f "fwf/bin/python (probe_wm|dreamer)[.]py"
sleep 20
pkill -KILL -f "fwf/bin/python (probe_wm|dreamer)[.]py"
sleep 5
for g in 1 2 3 4 5 6; do
  CUDA_MPS_PIPE_DIRECTORY=$PWD/.mps/gpu$g/pipe bash -c 'echo quit | nvidia-cuda-mps-control' 2>/dev/null
done
sleep 10
# only our daemons, addressed through our pipe dirs (other jobs of this account may use MPS)
alive=0
for g in 1 2 3 4 5 6; do
  CUDA_MPS_PIPE_DIRECTORY=$PWD/.mps/gpu$g/pipe bash -c 'echo get_server_list | nvidia-cuda-mps-control' >/dev/null 2>&1 && alive=$((alive + 1))
done
echo "$(date '+%F %T') remaining job processes: $(pgrep -fc 'fwf/bin/python (probe_wm|dreamer)[.]py')"
echo "$(date '+%F %T') our MPS daemons still alive: $alive"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
$PY scripts/summarize.py > /dev/null 2>&1 && echo "$(date '+%F %T') summary written to runs/summary.md"
echo "$(date '+%F %T') done"
