import json, glob, os, sys
import numpy as np
import sys
R = sys.argv[1] if len(sys.argv)>1 else "runs/recall"
def load(p):
    rows=[]
    for l in open(p):
        try: rows.append(json.loads(l))
        except: pass
    return rows
for task in sorted(os.listdir(R)):
    for cond in sorted(os.listdir(f"{R}/{task}")):
        out=[]
        for s in sorted(glob.glob(f"{R}/{task}/{cond}/s*")):
            m=f"{s}/metrics.jsonl"
            if not os.path.exists(m): out.append(f"{os.path.basename(s)}:none"); continue
            ev=[r for r in load(m) if "eval_auc" in r]
            if not ev: out.append(f"{os.path.basename(s)}:noeval"); continue
            hit=next((r["step"] for r in ev if r["eval_auc"]>0.9),None)
            fin=np.mean([r["eval_auc"] for r in ev[-3:]])
            rate=ev[-1].get("eval_deter_rate",float('nan'))
            ex=open(f"{s}/.exit").read().strip() if os.path.exists(f"{s}/.exit") else "?"
            out.append(f"{os.path.basename(s)}:{hit if hit is not None else '--'}@{ev[-1]['step']} fin{fin:.3f} r{rate:.3f} x{ex}")
        print(f"{task.replace('bsuite_memlen-','L').replace('-recall','')} {cond}: "+" | ".join(out))
