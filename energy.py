"""Operation counts and energy per environment step of the world-model filter (E5).

For a trained world model (probe run: <logdir>/wm.pt; RL run: <logdir>/latest_model.pt) runs the
encoder and the posterior filter (dynamics.observe), the part that runs at every step on the
agent, over held-out episodes and counts with forward hooks:

  AC   accumulate operations of Linear layers whose input is binary spikes (nonzero inputs x
       fan-out) or a spike rate over the T SNN steps (spike count x fan-out)
  MAC  multiply-accumulate operations: Linear layers with real-valued input (rows x in x out),
       normalisation layers (2 per element) and neuron state updates per element and call
       (LIF 1 per SNN step, MCN 4 per SNN step, PDRNN 3 per integrator and 6 per resonant dendritic
       channel, GRSN 4, GRU 4, DH-SNN 2 per dendritic branch and 2 per soma, TC-LIF 2)

Energy with the 45 nm figures common in SNN papers (Horowitz 2014): 0.9 pJ per AC, 4.6 pJ per
MAC. For RL runs the actor is counted as well (actor_*, agent_energy_nj_per_step).
Memory access is not counted; the hardware-aware accounting of research plan §4.4 is a
later step.

usage: python energy.py --configs bsuite ours --task bsuite_memlen-32-recall --logdir <run dir>
       (the configs and task of the run; writes <logdir>/energy.json)
"""
import json
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent))

import numpy as np
import torch
from torch import nn

import models
import networks
import node
import normalization
import probe_wm
import tools

E_AC, E_MAC = 0.9e-12, 4.6e-12


def spike_count(x, T):
    """Number of spikes if x is binary spikes or a rate over T steps, else None."""
    y = x * T
    if bool((x >= 0).all() and (x <= 1).all() and ((y - y.round()).abs() < 1e-4).all()):
        return float(y.sum())
    return None


class Counter:
    def __init__(self, T):
        self.T = T
        self.ac = self.mac = 0.0
        self.spikes = self.neurons = 0.0

    def linear(self, mod, inp, out):
        x = inp[0]
        tail = getattr(mod, "_real_tail", 0)
        if tail:  # spikes followed by `tail` real-valued inputs (first layer behind networks.LowRankInput)
            head = x[..., :-tail]
            n = float(head.sum()) if bool(((head == 0) | (head == 1)).all()) else spike_count(head, self.T)
            assert n is not None, "the inputs before the low-rank projection must be spikes"
            self.ac += n * mod.out_features
            self.mac += x[..., 0].numel() * tail * mod.out_features
            return
        if not x.dtype.is_floating_point:
            n = spike_count(x, 1)
        elif getattr(mod, "_rate_input", False):  # reads the rate of the T input frames (PDRNN, GRSN synapses)
            n = spike_count(x, self.T)
        elif bool(((x == 0) | (x == 1)).all()):  # one frame of binary spikes (fixed 2026-10-04: was counted T times)
            n = float(x.sum())
        else:  # a rate over the T frames, or real-valued (None: multiply-accumulates)
            n = spike_count(x, self.T)
        if n is not None:
            self.ac += n * mod.out_features
        else:
            self.mac += x[..., 0].numel() * mod.in_features * mod.out_features

    def norm(self, mod, inp, out):
        self.mac += 2 * out.numel()

    def elementwise(self, per_element):
        def hook(mod, inp, out):
            out = out[0][0] if isinstance(out, tuple) else out  # cells return (outputs, [state])
            self.mac += per_element * out.numel()
        return hook

    def dhsnn(self, mod, inp, out):
        """DH-SNN: the masked synapses run through F.linear, not mod.dense, so they are counted here. Each input (x: the
        rate of the T input frames; s_{t-1}: one binary frame) reaches the branches the mask connects it to (AC). At
        the first step s_{t-1} is the learned initial state, real-valued: multiply-accumulates, as for nn.Linear."""
        inputs, state = inp
        x, s = sum(inputs), state[0][-1]
        assert bool(((x - x.round()).abs() < 1e-4).all()), "DH-SNN inputs must be spikes"
        fan = mod.mask.sum(0)
        n_x = x.shape[-1]
        self.ac += float((x * fan[:n_x]).sum())
        if bool(((s == 0) | (s == 1)).all()):
            self.ac += float((s * fan[n_x:]).sum())
        else:
            self.mac += s[..., 0].numel() * float(fan[n_x:].sum())
        self.mac += (2 * mod.branches + 2) * out[0][0].numel()  # beta i + (1 - beta) W x per branch, alpha u + (1 - alpha) sum

    def spiking(self, per_element):
        def hook(mod, inp, out):
            self.mac += per_element * out.numel()
            self.spikes += float(out.sum())
            self.neurons += out.numel()
        return hook


def attach(wm, counter, parts=None):
    handles = []
    for part in parts or (wm.encoder, wm.dynamics):
        for mod in part.modules():
            if isinstance(mod, networks.LowRankInput):
                # the projection P reads the dendritic spikes (AC); the first layer reads the spikes of (z, h)
                # (AC) and the rank-r output of P, which is real-valued (MAC)
                first = next(m for m in mod.net.modules() if isinstance(m, nn.Linear))
                first._real_tail = mod.proj.out_features
            if isinstance(mod, nn.Linear):
                handles.append(mod.register_forward_hook(counter.linear))
            elif isinstance(mod, (normalization.PopNorm, nn.LayerNorm)):
                handles.append(mod.register_forward_hook(counter.norm))
            elif isinstance(mod, node.MCNode):
                handles.append(mod.register_forward_hook(counter.spiking(4)))
            elif isinstance(mod, node.LIFNode):
                handles.append(mod.register_forward_hook(counter.spiking(1)))
            elif isinstance(mod, node.ANNAct):
                handles.append(mod.register_forward_hook(counter.elementwise(1)))
            elif isinstance(mod, networks.PDRNN):
                for lin in list(mod.B) + list(mod.D) + list(getattr(mod, "R", [])):
                    lin._rate_input = True
                # 3 per integrator channel; a resonant channel rotates (a, b) and is read by C and C': 6
                handles.append(mod.register_forward_hook(counter.elementwise(3 * (mod._channels + mod._n_res) * mod._layers)))
            elif isinstance(mod, (networks.GRSNCell, networks.GRUCtrl)):
                if isinstance(mod, networks.GRSNCell):
                    mod.forget._rate_input = mod.inp._rate_input = True
                handles.append(mod.register_forward_hook(counter.elementwise(4)))
            elif isinstance(mod, networks.DHSNNCell):
                handles.append(mod.register_forward_hook(counter.dhsnn))
            elif isinstance(mod, networks.TCLIFCell):
                mod.inp._rate_input = True  # hid reads s_{t-1}, one binary frame
                handles.append(mod.register_forward_hook(counter.elementwise(2)))  # v1 -= g1 v2, v2 += g2 v1
            elif isinstance(mod, networks.LIFRNN):
                handles.append(mod.register_forward_hook(counter.elementwise(counter.T)))  # T updates per call
    return handles


def episodes(config, logdir, n):
    eval_dir = logdir / "eval_eps"
    if eval_dir.exists():  # RL run: the agent's own evaluation episodes
        eps = tools.load_episodes(eval_dir, limit=None)
        eps = list(eps.values())[:n]
        length = min(len(ep["reward"]) for ep in eps)
        return [{k: v[:length] for k, v in ep.items()} for ep in eps]
    eps, _ = probe_wm.collect(config, n, 12345 + 1, "eval")  # probe: same held-out data as training
    return list(eps.values())


@torch.no_grad()
def main():
    argv = sys.argv[1:]
    logdir = pathlib.Path(argv[argv.index("--logdir") + 1])
    config, args = probe_wm.parse_config(argv + ["--train_episodes", "1"])
    eps = episodes(config, logdir, 64)
    env = probe_wm.dreamer.make_env(config, "eval")
    config.num_actions = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
    wm = models.WorldModel(env.observation_space, env.action_space, 0, config).to(config.device)
    if (logdir / "wm.pt").exists():
        wm.load_state_dict(torch.load(logdir / "wm.pt", map_location=config.device))
    else:
        state = torch.load(logdir / "latest_model.pt", map_location=config.device)
        wm.load_state_dict({k[len("_wm."):].replace("_orig_mod.", "", 1): v  # torch.compile prefix
                            for k, v in state.items() if k.startswith("_wm.")})
    wm.eval()
    counter = Counter(config.spike_times)
    attach(wm, counter)
    batch = {k: np.stack([ep[k] for ep in eps]) for k in eps[0]}
    data = wm.preprocess(batch)
    post, _ = wm.dynamics.observe(wm.encoder(data), data["action"], data["is_first"])
    steps = data["action"].shape[0] * data["action"].shape[1]
    out = {
        "ac_per_step": counter.ac / steps,
        "mac_per_step": counter.mac / steps,
        "energy_nj_per_step": (E_AC * counter.ac + E_MAC * counter.mac) / steps * 1e9,
        "ff_spike_rate": counter.spikes / max(counter.neurons, 1),  # LIF/MCN neurons of the filter
        "deter_rate": float(post["deter"].float().mean()),
        "steps": steps,
    }
    if not (logdir / "wm.pt").exists():
        # RL run: also the actor, which runs at every step on the agent (Stage 3 review, RM-9).
        # It reads get_policy_feat: (z_t, h_t), plus the dendritic ON/OFF spikes with policy_mem.
        beh = models.ImagBehavior(config, wm).to(config.device)
        prefix = next(p for p in ("_task_behavior._orig_mod.actor.", "_task_behavior.actor.") if any(k.startswith(p) for k in state))
        beh.actor.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
        beh.eval()
        actor_counter = Counter(config.spike_times)
        attach(wm, actor_counter, parts=(beh.actor,))
        beh.actor(wm.dynamics.get_policy_feat(post))
        out["actor_ac_per_step"] = actor_counter.ac / steps
        out["actor_mac_per_step"] = actor_counter.mac / steps
        out["actor_energy_nj_per_step"] = (E_AC * actor_counter.ac + E_MAC * actor_counter.mac) / steps * 1e9
        out["agent_energy_nj_per_step"] = out["energy_nj_per_step"] + out["actor_energy_nj_per_step"]
    print(json.dumps(out))
    (logdir / "energy.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
