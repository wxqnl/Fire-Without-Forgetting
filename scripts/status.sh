#!/usr/bin/env bash
set -euo pipefail
cd /data/Minko
/data/Minko/fire-without-forgetting/envs/torch/bin/python - <<'PYTHON'
import json,pathlib,subprocess
root=pathlib.Path('/data/Minko/fire-without-forgetting')
state=json.loads((root/'runs/queue_status.json').read_text())
print(json.dumps(state,indent=2))
for worker in state.get('active',[]):
    print('GPU',worker['gpu'],worker['job'])
    path=pathlib.Path(worker['log'])
    if path.exists(): print('\n'.join(path.read_text().splitlines()[-3:]))
if state.get('active'):
    subprocess.run(['ps','-p',','.join(str(w['pid']) for w in state['active']),'-o','pid,etime,%cpu,rss,args'])
subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv'])
PYTHON
