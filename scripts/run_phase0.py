"""Dispatch the four phase-0 conditions across the authorized idle GPUs."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/data/Minko/fire-without-forgetting')
os.chdir('/data/Minko')
parser = argparse.ArgumentParser()
parser.add_argument('--adopt', type=Path)
args = parser.parse_args()
lock = (ROOT/'runs/queue.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
active = {}
last_report = None
handoff = None

def manifest():
    return json.loads((ROOT/'configs/phase0_initial.json').read_text())

def proc_info(pid):
    raw = Path(f'/proc/{pid}/stat').read_text()
    fields = raw[raw.rfind(')')+2:].split()
    return dict(state=fields[0], start_ticks=int(fields[19]), wait_status=int(fields[49]))

def runpath(name):
    return ROOT/'runs/phase0_initial'/name

def result(name):
    path = runpath(name)/'result.json'
    return json.loads(path.read_text()) if path.exists() else None

def publish(config):
    global last_report
    running = [dict(job=v['name'], gpu=g, pid=v['pid'],
                    started_at=datetime.datetime.fromtimestamp(v['start']).astimezone().isoformat(),
                    log=str(runpath(v['name'])/'train.log')) for g,v in sorted(active.items())]
    finished = {j['name']:result(j['name']) for j in config['jobs']}
    completed = [name for name,r in finished.items() if r is not None and r['returncode']==0]
    failed = [name for name,r in finished.items() if r is not None and r['returncode']!=0]
    running_names = {v['name'] for v in active.values()}
    pending = [j['name'] for j in config['jobs'] if finished[j['name']] is None and j['name'] not in running_names]
    state = 'running' if running else ('failed' if failed else ('waiting_for_gpu' if pending else 'completed'))
    report = dict(state=state, queue_pid=os.getpid(), total=len(config['jobs']), gpus=config['gpus'],
                  active=running, pending=pending, completed=completed, failed=failed,
                  seed_policy='seed 0 only')
    if report != last_report:
        print(json.dumps(report),flush=True)
        last_report = report
    report = dict(report, updated_at=datetime.datetime.now().astimezone().isoformat())
    path = ROOT/'runs/queue_status.json'
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(report,indent=2)); tmp.replace(path)
    return report

if args.adopt:
    handoff = json.loads(args.adopt.read_text())
    parent = proc_info(handoff['old_queue_pid'])
    if parent['state'] not in ('T','t') or parent['start_ticks'] != handoff['old_queue_start_ticks']:
        raise RuntimeError('The original controller must remain individually stopped during adoption.')
    child = proc_info(handoff['pid'])
    if child['start_ticks'] != handoff['child_start_ticks']:
        raise RuntimeError('The original training process identity changed.')
    if result(handoff['job']) is not None:
        raise RuntimeError('Adopted job already has a result; inspect before proceeding.')
    active[0] = dict(name=handoff['job'], pid=handoff['pid'], process=None, logfile=None,
                     start=datetime.datetime.fromisoformat(handoff['started_at']).timestamp(),
                     command=json.loads((runpath(handoff['job'])/'command.json').read_text()),
                     terminated_at=None, timed_out=False)

while True:
    config = manifest()
    for gpu, worker in list(active.items()):
        if worker['process'] is None:
            state = proc_info(worker['pid'])
            if state['start_ticks'] != handoff['child_start_ticks']:
                raise RuntimeError('Adopted process identity changed.')
            code = os.waitstatus_to_exitcode(state['wait_status']) if state['state']=='Z' else None
        else:
            code = worker['process'].poll()
        if code is None:
            if time.time()-worker['start'] > config['timeout_hours_per_run']*3600:
                if worker['terminated_at'] is None:
                    os.kill(worker['pid'], signal.SIGTERM)
                    worker['terminated_at'] = time.time()
                    worker['timed_out'] = True
                elif time.time()-worker['terminated_at'] > 30:
                    os.kill(worker['pid'], signal.SIGKILL)
            continue
        if worker['timed_out']: code = 124
        record = dict(returncode=code, elapsed_seconds=time.time()-worker['start'],
                      command=worker['command'], gpu=gpu)
        path = runpath(worker['name'])/'result.json'
        tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(record,indent=2)); tmp.replace(path)
        if worker['logfile']: worker['logfile'].close()
        if worker['process'] is None:
            parent = proc_info(handoff['old_queue_pid'])
            if parent['start_ticks'] != handoff['old_queue_start_ticks']:
                raise RuntimeError('Refusing to retire an unrelated controller.')
            # Its original training child has now exited; no training is interrupted.
            os.kill(handoff['old_queue_pid'], signal.SIGKILL)
        del active[gpu]
    failed = any((r:=result(j['name'])) is not None and r['returncode']!=0 for j in config['jobs'])
    if not failed:
        running_names = {w['name'] for w in active.values()}
        pending = [j for j in config['jobs'] if j['name'] not in running_names and result(j['name']) is None]
        free_slots = [g for g in config['gpus'] if g not in active]
        if pending and free_slots:
            usage = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                             '--format=csv,noheader,nounits'],text=True)
            availability = {int(parts[0]):(int(parts[1]),int(parts[2]))
                            for line in usage.strip().splitlines() if (parts:=line.split(','))}
            for gpu in free_slots:
                if not pending: break
                memory, utilization = availability[gpu]
                if memory>=128 or utilization!=0: continue
                job = pending.pop(0); name=job['name']; run=runpath(name)
                if (run/'latest_model.pt').exists():
                    raise RuntimeError(f'{name} was interrupted; upstream checkpoints lack optimizer state.')
                run.mkdir(parents=True,exist_ok=True)
                command=[str(ROOT/'envs/torch/bin/python'),'-u',str(ROOT/'code/spiking-wm/dreamer.py'),
                         '--configs','memory','--logdir',str(run)]
                for key,value in job.items():
                    if key!='name': command+=['--'+key,str(value)]
                (run/'command.json').write_text(json.dumps(command,indent=2))
                link=ROOT/'models'/name
                if not link.is_symlink() and not link.exists(): link.symlink_to(run/'latest_model.pt')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                         OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONWARNINGS='ignore',
                         WANDB_MODE='disabled',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID=str(gpu))
                logfile=(run/'train.log').open('w'); start=time.time()
                process=subprocess.Popen(command,cwd='/data/Minko',env=env,stdout=logfile,stderr=subprocess.STDOUT)
                active[gpu]=dict(name=name,pid=process.pid,process=process,logfile=logfile,
                                 start=start,command=command,terminated_at=None,timed_out=False)
    report=publish(config)
    if not active and (failed or not report['pending']):
        sys.exit(1 if failed else 0)
    time.sleep(5)
