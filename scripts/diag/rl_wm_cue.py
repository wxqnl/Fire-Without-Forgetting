"""Does the world model of an RL run keep the L101 cue? Loads the world model of an R2I L101 RL checkpoint and runs
probe_wm.evaluate on held-out random-policy L101 episodes (reward by action, as in RL), without training.
usage: python scripts/diag/rl_wm_cue.py <rl logdir> <configs...>"""
import json, sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np, torch
import probe_wm, models, dreamer
logdir, cfgs = sys.argv[1], sys.argv[2:]
config, args = probe_wm.parse_config(["--configs", *cfgs, "--task", "bsuite_memlen-101", "--batch_length", "128",
                                      "--reward_head", "symlog_mse", "--device", "cpu", "--eval_episodes", "256"])
eval_eps, env = probe_wm.collect(config, args.eval_episodes, args.data_seed + 1, "eval")
config.num_actions = env.action_space.shape[0]
wm = models.WorldModel(env.observation_space, env.action_space, 0, config).to("cpu")
sd = torch.load(f"{logdir}/latest_model.pt", map_location="cpu", weights_only=False)
sd = sd.get("agent_state_dict", sd)
wsd = {k[len("_wm."):].replace("_orig_mod.", ""): v for k, v in sd.items() if k.startswith("_wm.")}
missing, unexpected = wm.load_state_dict(wsd, strict=False)
print("missing", missing[:5], len(missing), "unexpected", unexpected[:5], len(unexpected))
out = probe_wm.evaluate(wm, eval_eps, config)
mem = out.get("eval_cue_mem", [])
print(json.dumps({k: out[k] for k in ("eval_auc", "eval_bal_acc", "eval_reward_mse", "eval_deter_rate")}))
print("cue decodable from dendrites, rows 0,1,2,5,10,25,50,75,100,end:", [mem[i] for i in (0, 1, 2, 5, 10, 25, 50, 75, 100)], out.get("eval_cue_mem_end"))
h = out.get("eval_cue_h", [])
print("cue decodable from head features, same rows:", [h[i] for i in (0, 1, 2, 5, 10, 25, 50, 75, 100)], out.get("eval_cue_h_end"))
