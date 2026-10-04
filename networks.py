import math
import numpy as np
import re

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as td
import node
import normalization
from einops import rearrange
import tools


class RSSM(nn.Module):
    def __init__(
        self,
        config,
        stoch=30,
        deter=200,
        hidden=200,
        layers_input=1,
        layers_output=1,
        rec_depth=1,
        shared=False,
        discrete=False,
        act="LIFNode",
        norm="PopNorm",
        mean_act="none",
        std_act="softplus",
        temp_post=True,
        min_std=0.1,
        cell="MCRNN",
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,
        embed=None,
        device=None
    ):
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._layers_input = layers_input
        self._layers_output = layers_output
        self._rec_depth = rec_depth
        self._shared = shared
        self._discrete = discrete
        act_p = eval("config."+act) 
        norm_p = eval("config."+norm)
        act = getattr(node, act)
        norm = getattr(normalization, norm)
        self._mean_act = mean_act
        self._std_act = std_act
        self._temp_post = temp_post
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._embed = embed
        self._device = device

        self.T = config.spike_times
        # persist_state=True carries the recurrent cell's membrane potentials across
        # environment steps (intervention A); they are cleared only at episode starts
        self._persist = getattr(config, "persist_state", False)

        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        if self._shared:
            inp_dim += self._embed
        for i in range(self._layers_input):
            inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            inp_layers.append(norm(self._hidden, **norm_p))
            inp_layers.append(act(**act_p))
            if i == 0:
                inp_dim = self._hidden
        self._inp_layers = nn.Sequential(*inp_layers)
        self._inp_layers.apply(tools.weight_init)

        self._cell = eval(cell)(config, self._hidden, self._deter, **config.dyn_cell_p)
        self._cell.apply(tools.weight_init)
        self._mem_keys = list(self._cell.neuron.mems.keys()) if self._persist else []
        # policy_mem 'onoff' (EXPERIMENTS.md §18.2): the actor and critic also read the cell's state
        # (policy_mem_layers: the last layer's or all), each value standardised over its layer and sent
        # as T ON and T OFF spike frames, thresholds ±(k - 1/2)·2/T. The world-model heads read get_feat.
        self._policy_mem = getattr(config, "policy_mem", "none")
        last = getattr(config, "policy_mem_layers", "last") == "last"
        self._policy_keys = self._mem_keys[-1:] if last else self._mem_keys
        index = getattr(self._cell, "state_index", None)  # drops the b of integrator channels, always 0
        size = self._deter if index is None else len(index)
        if index is None and hasattr(self._cell, "mem_sizes"):
            size = next(iter(self._cell.mem_sizes.values()))
        self.policy_extra = 0 if self._policy_mem == "none" else 2 * size * len(self._policy_keys)
        # if cell == "gru":
        #     self._cell = GRUCell(self._hidden, self._deter)
        #     self._cell.apply(tools.weight_init)
        # elif cell == "gru_layer_norm":
        #     self._cell = GRUCell(self._hidden, self._deter, norm=True)
        #     self._cell.apply(tools.weight_init)
        # else:
        #     raise NotImplementedError(cell)

        img_out_layers = []
        inp_dim = self._deter
        for i in range(self._layers_output):
            img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            img_out_layers.append(norm(self._hidden, **norm_p))
            img_out_layers.append(act(**act_p))
            if i == 0:
                inp_dim = self._hidden
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(tools.weight_init)

        obs_out_layers = []
        if self._temp_post:
            inp_dim = self._deter + self._embed
        else:
            inp_dim = self._embed
        for i in range(self._layers_output):
            obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            obs_out_layers.append(norm(self._hidden, **norm_p))
            obs_out_layers.append(act(**act_p))
            if i == 0:
                inp_dim = self._hidden
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)

        if self._discrete:
            self._ims_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._ims_stat_layer.apply(tools.weight_init)
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(tools.weight_init)
        else:
            self._ims_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._ims_stat_layer.apply(tools.weight_init)
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer.apply(tools.weight_init)

        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

 
    def initial(self, batch_size):
        # deter = torch.zeros(batch_size, self._deter).to(self._device)
        deter = torch.zeros(self.T, batch_size, self._deter).to(self._device)
        if self._discrete:
            state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch]).to(self._device),
                std=torch.zeros([batch_size, self._stoch]).to(self._device),
                stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
                deter=deter,
            )
        for c in self._mem_keys:
            size = getattr(self._cell, "mem_sizes", {}).get(c, self._deter)
            state["mem_" + c] = torch.zeros([batch_size, size]).to(self._device)
        if self._initial == "zeros":
            return state
        # TODO
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(self.T, batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        if state is None:
            state = self.initial(action.shape[0])
        # (batch, time, ch) -> (time, batch, ch)
        embed = rearrange(embed, "st b t d -> t st b d") 
        # print("$"*100)
        # print("embed in observation shape: ", embed.shape)
        action, is_first = tools.swap(action), tools.swap(is_first)
        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ),
            (action, embed, is_first),
            (state, state),
        )

        # (time, batch, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        # deter [times, spike_time, batch, deter_num] -> [spike_time, batch, time, deter_num]
        post = {k: tools.swap(v, (0, 1, 2), (1, 2, 0)) if k=="deter" else tools.swap(v) for k, v in post.items()}
        prior = {k: tools.swap(v, (0, 1, 2), (1, 2, 0)) if k=="deter" else tools.swap(v) for k, v in prior.items()}
        return post, prior

    def imagine(self, action, state=None):
        if state is None:
            state = self.initial(action.shape[0])
        assert isinstance(state, dict), state
        action = action
        action = tools.swap(action)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: tools.swap(v) for k, v in prior.items()}
        return prior
    

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        stoch= torch.unsqueeze(stoch, 0)
        re_shape =  [1] * len(stoch.shape)
        re_shape[0] = state["deter"].shape[0]  # spike_times
        return torch.cat([stoch.repeat(*re_shape), state["deter"]], -1)

    def get_policy_feat(self, state):
        feat = self.get_feat(state)
        if self._policy_mem == "none":
            return feat
        z = []
        index = getattr(self._cell, "state_index", None)
        for c in self._policy_keys:
            m = state["mem_" + c].float()
            if index is not None:
                m = m[..., index.to(m.device)]
            z.append((m - m.mean(-1, keepdim=True)) / (m.std(-1, keepdim=True) + 1e-3))
        z = torch.cat(z, -1)
        frames = [torch.cat([(z > t).float(), (-z > t).float()], -1) for t in (torch.arange(self.T) + 0.5) * 2 / self.T]
        return torch.cat([feat, torch.stack(frames, 0).to(feat.dtype)], -1)

    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state["logit"]
            dist = td.independent.Independent(
                tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
        else:
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(
                td.independent.Independent(td.normal.Normal(mean, std), 1)
            )
        return dist

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        # if shared is True, prior and post both use same networks(inp_layers, _img_out_layers, _ims_stat_layer)
        # otherwise, post use different network(_obs_out_layers) with prior[deter] and embed as inputs
        self.reset()
        prev_action *= (1.0 / torch.clip(torch.abs(prev_action), min=1.0)).detach()
        # print("#"*100)
        # print("#"*100)
        # print("prev_state deter:  ", prev_state["deter"].shape)
        # 初始化 is_firest 对应 idx的 state, 更新prev_state内容
        if torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                if key == "deter":
                    is_first_r = torch.reshape(
                        is_first, 
                        (1,) + is_first.shape + (1,) * (len(val.shape) -1 - len(is_first.shape))
                    )
                else:
                    is_first_r = torch.reshape(
                        is_first,
                        is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                    )
                # print("#"*100)
                # print("key: ", key)
                # print("prev_state[key] shape: ", prev_state[key].shape)
                # print("val shape: ", val.shape)
                # print("is_first_r shape: ", is_first_r.shape)
                # print("init_state[key] shape: ", init_state[key].shape)
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )
        # print("prev_state deter:  ", prev_state["deter"].shape)
        prior = self.img_step(prev_state, prev_action, None, sample)
        if self._shared:
            post = self.img_step(prev_state, prev_action, embed, sample)
        else:
            if self._temp_post:
                # print("#"*100)
                # print("deter shape: ", prior["deter"].shape)
                # print("stoch shape: ", prior["stoch"].shape)
                # print("embed shape: ", embed.shape)
                x = torch.cat([prior["deter"], embed], -1)
            else:
                x = embed
            # (batch_size, prior_deter + embed) -> (batch_size, hidden)
            xs = []
            for step in range(self.T):
                xs.append(self._obs_out_layers(x[step]))
            # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
            xs = sum(xs) /self.T
            stats = self._suff_stats_layer("obs", xs)
            if sample:
                stoch = self.get_dist(stats).sample()
            else:
                stoch = self.get_dist(stats).mode()
            mems = {k: v for k, v in prior.items() if k.startswith("mem_")}
            post = {"stoch": stoch, "deter": prior["deter"], **stats, **mems}
        return post, prior

    # this is used for making future image

    def img_step(self, prev_state, prev_action, embed=None, sample=True):
        # (batch, stoch, discrete_num)
        self.reset()
        prev_action *= (1.0 / torch.clip(torch.abs(prev_action), min=1.0)).detach()
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)
        if self._shared:
            if embed is None:
                shape = list(prev_action.shape[:-1]) + [self._embed]
                embed = torch.zeros([self.T] + shape)
            # TODO 这里没用被执行
            # embed [s_t, bacth, dimes]
            # (batch, stoch * discrete_num) -> (self.T, batch, stoch * discrete_num + action, embed)
            x = torch.cat([prev_stoch.expand(self.T, *list(prev_stoch.shape)), 
                           prev_action.expand(self.T, *list(prev_action.shape)),
                            embed], -1)
        else:
            x = torch.cat([prev_stoch, prev_action], -1)
            # x = x.expand(self.T, *list(x.shape))
        # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
        # 需要变成 spike
        x_spikes = []
        if self._shared:
            for step in range(self.T):
                x_spikes.append(self._inp_layers(x[step]))
        else:
            for step in range(self.T):
                x_spikes.append(self._inp_layers(x))
        deter = prev_state["deter"]   
        if self._persist:
            self._cell.neuron.mems = {c: prev_state["mem_" + c] for c in self._mem_keys}
        # 这里 rec_depth 被设置成1
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            # deter 为[x_spikes]
            x_spikes, deter = self._cell(x_spikes, [deter])
            deter = deter[0]  # Keras wraps the state in a list.
        # (batch, deter) -> (batch, hidden)
        xs = []
        for step in range(self.T):
            xs.append(self._img_out_layers(x_spikes[step]))
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        xs = sum(xs) / self.T
        # TODO 
        stats = self._suff_stats_layer("ims", xs)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        for c in self._mem_keys:
            prior["mem_" + c] = self._cell.neuron.mems[c]
        return prior


    def get_stoch(self, deter):
        self.reset()
        spikes = []
        for step in range(self.T):
            x = self._img_out_layers(deter[step])
            spikes.append(x)
        x = sum(spikes) / self.T
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()


    def _suff_stats_layer(self, name, x):
        if self._discrete:
            if name == "ims":
                x = self._ims_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._ims_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = td.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        rep_loss = torch.mean(torch.clip(rep_loss, min=free))
        dyn_loss = torch.mean(torch.clip(dyn_loss, min=free))
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()

class MultiEncoder(nn.Module):
    def __init__(
        self,
        shapes,
        config,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        symlog_inputs,
    ):
        super(MultiEncoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = SpikeConvEncoder(
                input_shape, cnn_depth, act, eval("config."+act), norm, eval("config."+norm), kernel_size, minres, config.spike_times
            )
            self.outdim += self._cnn.outdim
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self._mlp = SpikeMLP(
                input_size,
                None,
                mlp_layers,
                mlp_units,
                act,
                getattr(config, act),
                norm,
                getattr(config, norm),
                symlog_inputs=symlog_inputs,
                spike_times=config.spike_times
            )
            self.outdim += mlp_units

    def forward(self, obs):
        outputs = []
        if self.cnn_shapes:
            inputs = torch.cat([obs[k] for k in self.cnn_shapes], -1)
            cnn_out = self._cnn(inputs) # cnn output spike list []
            outputs.append(torch.stack(cnn_out, dim=0))   
        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            mlp_out = self._mlp(inputs, unroll=True, spikes=True)  # mlp out spike list []
            outputs.append(torch.stack(mlp_out, dim=0))

        outputs = torch.cat(outputs, -1)

        return outputs



class MultiDecoder(nn.Module):
    def __init__(
        self,
        config,
        feat_size,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        cnn_sigmoid,
        image_dist,
        vector_dist,
    ):
        super(MultiDecoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)

        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = SpikeConvDecoder(
                feat_size,
                shape,
                cnn_depth,
                act,
                eval("config."+act),
                norm,
                eval("config."+norm),
                kernel_size,
                minres,
                cnn_sigmoid=cnn_sigmoid,
                spike_times=config.spike_times
            )
        if self.mlp_shapes:
            self._mlp = SpikeMLP(
                feat_size,
                self.mlp_shapes,
                mlp_layers,
                mlp_units,
                act,
                eval("config."+act),
                norm,
                eval("config."+norm),
                vector_dist,
                spike_times=config.spike_times
            )
        self._image_dist = image_dist

    def forward(self, features):
        dists = {}
        if self.cnn_shapes:
            feat = features
            outputs = self._cnn(feat)
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update(
                {
                    key: self._make_image_dist(output)
                    for key, output in zip(self.cnn_shapes.keys(), outputs)
                }
            )
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, mean):
        if self._image_dist == "normal":
            return tools.ContDist(
                td.independent.Independent(td.normal.Normal(mean, 1), 3)
            )
        if self._image_dist == "mse":
            return tools.MSEDist(mean)
        raise NotImplementedError(self._image_dist)


class SpikeConvEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        depth=32,
        act="LIFNode",
        act_p = {},
        norm="PopNorm",
        norm_p = {},
        kernel_size=4,
        minres=4,
        spike_times=5
    ):
        super().__init__()
        self.T = spike_times
        act = getattr(node, act)
        norm = getattr(normalization, norm)
        h, w, input_ch = input_shape
        layers = []
        
        for i in range(int(np.log2(h) - np.log2(minres))):
            if i == 0:
                in_dim = input_ch
            else:
                in_dim = 2 ** (i - 1) * depth
            out_dim = 2**i * depth
            layers.append(
                Conv2dSame(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            layers.append(ChNorm(out_dim, norm, norm_p))
            layers.append(act(**act_p))
            h, w = h // 2, w // 2

        self.outdim = out_dim * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        self.reset()
        outs =[]
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)

        inp = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        inp = inp.permute(0, 3, 1, 2)

        for step in range(self.T):
            # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
            # x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
            # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
            # x = x.permute(0, 3, 1, 2)
            x = self.layers(inp)
            # (batch * time, ...) -> (batch * time, -1)
            x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
            # (batch * time, -1) -> (batch, time, -1)
            outs.append(x.reshape(list(obs.shape[:-3]) + [x.shape[-1]]))
        return outs
    
    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()



class SpikeConvDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shape=(3, 64, 64),
        depth=32,
        act="LIF",
        act_p={},
        norm="PopNorm",
        norm_p = {},
        kernel_size=4,
        minres=4,
        outscale=1.0,
        cnn_sigmoid=False,
        spike_times=5
    ):
        super(SpikeConvDecoder, self).__init__()
        self.T = spike_times
        act = getattr(node, act)
        norm = getattr(normalization, norm)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = int(np.log2(shape[1]) - np.log2(minres))
        self._minres = minres
        self._embed_size = minres**2 * depth * 2 ** (layer_num - 1)

        self._linear_layer = nn.Linear(feat_size, self._embed_size)
        self._linear_layer.apply(tools.weight_init)
        in_dim = self._embed_size // (minres**2)

        layers = []
        h, w = minres, minres
        for i in range(layer_num):
            out_dim = self._embed_size // (minres**2) // (2 ** (i + 1))
            bias = False
            initializer = tools.weight_init
            if i == layer_num - 1:
                out_dim = self._shape[0]
                act = False
                bias = True
                norm = False
                initializer = tools.uniform_weight_init(outscale)

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ChNorm(out_dim, norm, norm_p))
            if act:
                layers.append(act(**act_p))
            [m.apply(initializer) for m in layers[-3:]]
            h, w = h * 2, w * 2

        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features, dtype=None):
        # TODO feature shape
        self.reset()
        spikes = []
        for step in range(self.T):
            x = self._linear_layer(features[step])
        # (batch, time, -1) -> (batch , time, h, w, ch)
            x = x.reshape(
                [-1, self._minres, self._minres, self._embed_size // self._minres**2]
            )
            # (batch, time, -1) -> (batch * time, ch, h, w)
            x = x.permute(0, 3, 1, 2)
            x = self.layers(x)
            spikes.append(x)
        mean = sum(spikes)/self.T
        # print("#"*100)
        # print("mean shape: ", mean.shape)
        # print("feature shape: ", features.shape)
        # print("self._shape: ", self._shape)
        # (batch,*time, ch, h, w) -> (batch, time, ch, h, w) necessary???
        mean = mean.reshape(features.shape[1:-1] + self._shape)
        # (batch * time, ch, h, w) -> (batch * time, h, w, ch)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean) - 0.5
        return mean
    
    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()


class SpikeMLP(nn.Module):
    def __init__(
        self,
        inp_dim,
        shape,
        layers,
        units,
        act="LIF",
        act_p={},
        norm="PopNorm",
        norm_p={},
        dist="normal",
        std=1.0,
        outscale=1.0,
        symlog_inputs=False,
        device="cuda",
        spike_times=5
    ):
        super(SpikeMLP, self).__init__()
        self.T = spike_times
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        self._layers = layers
        act = getattr(node, act)
        norm = getattr(normalization, norm)
        self._dist = dist
        self._std = std
        self._symlog_inputs = symlog_inputs
        self._device = device

        layers = []
        for index in range(self._layers):
            layers.append(nn.Linear(inp_dim, units, bias=False))
            layers.append(norm(units, **norm_p))
            layers.append(act(**act_p))
            if index == 0:
                inp_dim = units
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None, unroll=False, pr=False, spikes=False):
        self.reset()
        if unroll:
            shape = [self.T] + [1] * len(features.shape)
            features = features.unsqueeze(0).repeat(*shape)
        if self._symlog_inputs:
            features = tools.symlog(features)
        outs = []
        if pr:
            print("features shape: ", features.shape)
        for step in range(self.T):
            outs.append(self.layers(features[step]))
        if spikes:
            # per-step spike list, the same interface as SpikeConvEncoder
            return outs
        outs =  sum(outs) / self.T
        
        if self._shape is None:
            return outs
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](outs)
                if self._std == "learned":
                    std = self.std_layer[name](outs)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(outs)
            if self._std == "learned":
                std = self.std_layer(outs)
            else:
                std = self._std
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        if dist == "normal":
            return tools.ContDist(
                td.independent.Independent(
                    td.normal.Normal(mean, std), len(shape)
                )
            )
        if dist == "huber":
            return tools.ContDist(
                td.independent.Independent(
                    tools.UnnormalizedHuber(mean, std, 1.0), len(shape)
                )
            )
        if dist == "binary":
            return tools.Bernoulli(
                td.independent.Independent(
                    td.bernoulli.Bernoulli(logits=mean), len(shape)
                )
            )
        if dist == "symlog_disc":
            return tools.DiscDist(logits=mean, device=self._device)
        if dist == "symlog_mse":
            return tools.SymlogDist(mean)
        raise NotImplementedError(dist)
    
    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()


class ActionHead(nn.Module):
    def __init__(
        self,
        inp_dim,
        size,
        layers,
        units,
        act="LIF",
        act_p={},
        norm="PopNorm",
        norm_p={},
        dist="trunc_normal",
        init_std=0.0,
        min_std=0.1,
        max_std=1.0,
        temp=0.1,
        outscale=1.0,
        unimix_ratio=0.01,
        spike_times=5
    ):
        super(ActionHead, self).__init__()
        self.T= spike_times
        self._size = size
        self._layers = layers
        self._units = units
        self._dist = dist
        act = getattr(node, act)
        norm = getattr(normalization, norm)
        self._min_std = min_std
        self._max_std = max_std
        self._init_std = init_std
        self._unimix_ratio = unimix_ratio
        self._temp = temp() if callable(temp) else temp


        pre_layers = []
        for index in range(self._layers):
            pre_layers.append(nn.Linear(inp_dim, self._units, bias=False))
            pre_layers.append(norm(self._units, **norm_p))
            pre_layers.append(act(**act_p))
            if index == 0:
                inp_dim = self._units
        self._pre_layers = nn.Sequential(*pre_layers)
        self._pre_layers.apply(tools.weight_init)

        if self._dist in ["tanh_normal", "tanh_normal_5", "normal", "trunc_normal"]:
            self._dist_layer = nn.Linear(self._units, 2 * self._size)
            self._dist_layer.apply(tools.uniform_weight_init(outscale))

        elif self._dist in ["normal_1", "onehot", "onehot_gumbel"]:
            self._dist_layer = nn.Linear(self._units, self._size)
            self._dist_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        # x = features
        self.reset()
        xs = []
        for step in range(self.T):
            s = self._pre_layers(features[step])
            xs.append(s)
        x = sum(xs) / self.T
        if self._dist == "tanh_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = torch.tanh(mean)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = td.normal.Normal(mean, std)
            dist = td.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = td.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "tanh_normal_5":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = 5 * torch.tanh(mean / 5)
            std = F.softplus(std + 5) + 5
            dist = td.normal.Normal(mean, std)
            dist = td.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = td.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size] * 2, -1)
            std = (self._max_std - self._min_std) * torch.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = td.normal.Normal(torch.tanh(mean), std)
            dist = tools.ContDist(td.independent.Independent(dist, 1))
        elif self._dist == "normal_1":
            x = self._dist_layer(x)
            dist = td.normal.Normal(mean, 1)
            dist = tools.ContDist(td.independent.Independent(dist, 1))
        elif self._dist == "trunc_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size] * 2, -1)
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(td.independent.Independent(dist, 1))
        elif self._dist == "onehot":
            x = self._dist_layer(x)
            dist = tools.OneHotDist(x, unimix_ratio=self._unimix_ratio)
        elif self._dist == "onehot_gumble":
            x = self._dist_layer(x)
            temp = self._temp
            dist = tools.ContDist(td.gumbel.Gumbel(x, 1 / temp))
        else:
            raise NotImplementedError(self._dist)
        return dist

    def reset(self):
            for mod in self.modules():
                if hasattr(mod, 'n_reset'):
                    mod.n_reset()

class MCRNN(nn.Module): 
    def __init__(self, 
                 config,
                 input_size, 
                 size, 
                 act="MCNode",
                 norm="PopNorm",
                 device="cuda"):
        super().__init__()
        self._device = device
        self._norm = norm
        self.neuron = getattr(node, act)(**eval("config."+act))
        self._size = size
        self.apcial_w = nn.Linear(input_size+size, size)
        self.basal_w = nn.Linear(input_size+size, size) 
        self.soma_w = nn.Linear(input_size, size)
        # self.h2o = nn.Linear(size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1) 
        self.T = config.spike_times
        self._persist = getattr(config, "persist_state", False)
        # TODO
        if norm:
            norm = getattr(normalization, norm)
            self.apical_norm=norm(size, **eval("config."+self._norm))
            self.basal_norm=norm(size, **eval("config."+self._norm))
            self.soma_norm=norm(size, **eval("config."+self._norm))
    

    def forward(self, inputs, state):
        # input spike list or tensor [spike_T, batch, ...]
        # state spike list or tensor [spike_T, batch, ...]
        if not self._persist:
            self.reset()
        # combined = torch.cat((inputs, hidden), -1) # [T, batch, dims]
        # assert combined.shape[0] == self.T
        outputs = []
        # print("inputs device: ", inputs.device)
        for step in range(self.T):
            inp_spike = inputs[step]
            # print("#"*100)
            # print("state shape: ", state.shape)
            # print("inspike sahpe: ", inp_spike.shape)
            # print("state[step] shape: ", state[step].shape)
            combined = torch.cat((inp_spike, state[0][step]), -1)
            apcial_inp = self.apcial_w(combined)
            basal_inp = self.basal_w(combined)
            soma_inp = self.soma_w(inp_spike)
            if self._norm:
                apcial_inp = self.apical_norm(apcial_inp)
                basal_inp = self.basal_norm(basal_inp)
                soma_inp = self.soma_norm(soma_inp)
            neu_input= {"basal_inputs":basal_inp, "apical_inputs": apcial_inp, "soma_inputs": soma_inp}
            spike = self.neuron(neu_input)
            # print("spike shape: ", spike.shape)
            outputs.append(spike)
        # out
        # output = self.h2o(sum(outputs)/self.T)
        # output = self.softmax(output)
        return outputs, [torch.stack(outputs, dim=0)]

    def initHidden(self):
        return torch.zeros(self.T, 1, self.hidden_size).to(self._device)

    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset() 



# class GRUCell(nn.Module):
#     def __init__(self, inp_size, size, norm=False, act=torch.tanh, update_bias=-1):
#         super(GRUCell, self).__init__()
#         self._inp_size = inp_size
#         self._size = size
#         self._act = act
#         self._norm = norm
#         self._update_bias = update_bias
#         self._layer = nn.Linear(inp_size + size, 3 * size, bias=False)
#         if norm:
#             self._norm = nn.LayerNorm(3 * size, eps=1e-03)

#     @property
#     def state_size(self):
#         return self._size

#     def forward(self, inputs, state):
#         state = state[0]  # Keras wraps the state in a list.
#         parts = self._layer(torch.cat([inputs, state], -1))
#         if self._norm:
#             parts = self._norm(parts)
#         reset, cand, update = torch.split(parts, [self._size] * 3, -1)
#         reset = torch.sigmoid(reset)
#         cand = self._act(reset * cand)
#         update = torch.sigmoid(update + self._update_bias)
#         output = update * cand + (1 - update) * state
#         return output, [output]


class GRUCtrl(nn.Module):
    """Positive control for the Phase 0 probe: a continuous-state GRU core inside the
    otherwise spiking world model. It keeps the MCRNN interface (list of T input spike
    tensors in, T identical outputs out) so that everything else stays unchanged."""

    def __init__(self, config, input_size, size, act=None, norm=None, device="cuda"):
        super().__init__()
        self.T = config.spike_times
        self._layer = nn.Linear(input_size + size, 3 * size, bias=False)
        self._norm = nn.LayerNorm(3 * size, eps=1e-03)

    def forward(self, inputs, state):
        x = sum(inputs) / self.T
        h = state[0][-1]
        parts = self._norm(self._layer(torch.cat([x, h], -1)))
        reset, cand, update = torch.split(parts, [h.shape[-1]] * 3, -1)
        cand = torch.tanh(torch.sigmoid(reset) * cand)
        update = torch.sigmoid(update - 1)
        h = update * cand + (1 - update) * h
        outputs = [h] * self.T
        return outputs, [torch.stack(outputs, dim=0)]


class PDState(nn.Module):
    """The dendritic states of PDRNN, under the interface RSSM uses for persist_state."""

    def __init__(self, keys):
        super().__init__()
        self.keys = keys
        self.mems = {k: None for k in keys}

    def n_reset(self):
        self.mems = {k: None for k in self.keys}


class PDRNN(nn.Module):
    """Method prototype (research plan §4.2): persistent, non-resetting linear dendrites and a
    stateless spiking soma, updated once per environment step. Per layer, with M dendritic
    channels per neuron:

        d_t = λ ⊙ d_{t-1} + g(λ) ⊙ LN(B x_t)            λ = exp(-1/τ), τ log-uniform in [tau_min, tau_max]
        u_t = PopNorm(Σ_m C_m ⊙ d_t,m + D x_t)       s_t = H(u_t - θ)

    g(λ) = sqrt(1 - λ²) (input_norm 'lru', the LRU normalisation; intervention G is the same up to
    a factor 2) or 1 - λ (input_norm 'euler', the gain of the MCNode Euler update). x_t is the
    rate of the T input spikes; later layers read the spikes of the previous one. There is no
    feedback from the spikes into the dendrites, so d is linear in the inputs and a parallel scan
    could train it (not used here). Use with persist_state (configs block pdcell).

    The soma turns u_t into the T spike frames that the rest of the spiking world model reads:
      soma 'step'  one frame s_t, repeated T times (512 binary values per environment step)
      soma 'if'    integrate-and-fire over the T SNN steps of the environment step: the membrane
                   starts at rest, integrates the constant current u_t - and is reset by
                   subtraction of θ after each spike, so the spike count encodes u_t (rate code,
                   T + 1 levels). The first frame is the 'step' frame. The soma holds no state
                   across environment steps: the memory is in the dendrites.
    Later layers read the rate of the previous layer's frames.

    resonant > 0 makes that fraction of each neuron's channels resonant: the channel is a damped
    oscillator with two state variables (a, b), rotated by its phase θ at every step and decaying
    with λ,

        (a, b)_t = λ R(θ) (a, b)_{t-1} + (g(λ) LN(B x_t), 0)      u_t = Σ_m C_m a_t,m + C'_m b_t,m + D x_t

    (the complex recurrence d_t = λ e^{iθ} d_{t-1} + g(λ) LN(B x_t) in real form). An integrator
    channel (θ = 0) holds a value; a resonant one also encodes how long ago its input arrived, which
    is what recalling the input of exactly k steps ago needs. θ ~ U[0, max_phase] (the LRU's
    initialisation) and is learned with τ.

    Design ablations (EXPERIMENTS.md §19.8), both off in the model:
      soma 'lif'          the soma keeps a membrane across environment steps, v_t = β v_{t-1} + PopNorm(u_t)
                          with β = exp(-1/soma_tau), hard reset to 0 by its spike (one frame, repeated T times)
      spike_feedback      the dendrites also receive the layer's own spikes of the previous step,
                          LN(B x_t + R s_{t-1}), so the recurrence also passes through the threshold"""

    def __init__(self, config, input_size, size, act=None, norm="PopNorm", device="cuda"):
        super().__init__()
        p = config.PDRNN
        self.T = config.spike_times
        self._layers, self._channels, self._size = p["layers"], p["channels"], size
        self._input_norm = p["input_norm"]
        self.threshold = p["threshold"]
        keys = [f"d{i}" for i in range(self._layers)]
        self.neuron = PDState(keys)
        self.mem_sizes = {k: size * self._channels for k in keys}
        norm_cls = getattr(normalization, norm)
        self.B, self.D = nn.ModuleList(), nn.ModuleList()
        self.in_norm, self.soma_norm = nn.ModuleList(), nn.ModuleList()
        self.log_tau, self.C = nn.ParameterList(), nn.ParameterList()
        in_dim = input_size
        for _ in range(self._layers):
            n = size * self._channels
            self.B.append(nn.Linear(in_dim, n, bias=False))
            self.in_norm.append(nn.LayerNorm(n, eps=1e-3))
            self.D.append(nn.Linear(in_dim, size, bias=False))
            self.soma_norm.append(norm_cls(size, **getattr(config, norm)))
            log_tau = torch.empty(n).uniform_(math.log(p["tau_min"]), math.log(p["tau_max"] + 1e-6))
            self.log_tau.append(nn.Parameter(log_tau))
            self.C.append(nn.Parameter(torch.randn(size, self._channels) / math.sqrt(self._channels)))
            in_dim = size
        self.act_fun = node.QGateGrad(alpha=2.0, requires_grad=False)
        self._learn_tau = p["learn_tau"]
        self._soma = p.get("soma", "step")
        self._n_res = round(p.get("resonant", 0.0) * self._channels)  # resonant channels per neuron
        if self._n_res:
            self.mem_sizes = {k: 2 * size * self._channels for k in keys}
            self.theta, self.C2 = nn.ParameterList(), nn.ParameterList()
            for _ in range(self._layers):
                theta = torch.zeros(size, self._channels)
                theta[:, : self._n_res].uniform_(0, p["max_phase"])
                self.theta.append(nn.Parameter(theta.flatten()))
                self.C2.append(nn.Parameter(torch.randn(size, self._channels) / math.sqrt(self._channels)))
            self.register_buffer("res_mask", (torch.arange(self._channels) < self._n_res).float().repeat(size))
            # entries of the stored state that can be nonzero: all a, and b of the resonant channels
            self.state_index = torch.cat([torch.arange(size * self._channels),
                                          size * self._channels + torch.nonzero(self.res_mask)[:, 0]])
        # states of the design ablations; listed first so that the last key stays the last layer's
        # dendrites (what policy_mem reads)
        self._feedback = p.get("spike_feedback", False)
        extra = {}
        if self._soma == "lif":
            self.soma_decay = math.exp(-1.0 / p.get("soma_tau", 2.0))
            extra.update({f"v{i}": size for i in range(self._layers)})
        if self._feedback:
            self.R = nn.ModuleList([nn.Linear(size, size * self._channels, bias=False) for _ in range(self._layers)])
            extra.update({f"s{i}": size for i in range(self._layers)})
        if extra:
            self.neuron = PDState(list(extra) + keys)
            self.mem_sizes = {**extra, **self.mem_sizes}

    def soma(self, u):
        if self._soma == "step":
            return [self.act_fun(u - self.threshold)] * self.T
        if self._soma == "thermo":  # diagnostic: the IF spike count without the IF dynamics, frame k: H(u - kθ/T)
            return [self.act_fun(u - self.threshold * k / self.T) for k in range(1, self.T + 1)]
        v, frames = torch.zeros_like(u), []
        for _ in range(self.T):
            v = v + u
            s = self.act_fun(v - self.threshold)
            v = v - s.detach() * self.threshold
            frames.append(s)
        return frames

    def forward(self, inputs, state):
        x = sum(inputs) / self.T
        for i in range(self._layers):
            key = f"d{i}"
            n = self._size * self._channels
            d = self.neuron.mems[key]
            if d is None:
                d = torch.zeros(x.shape[0], 2 * n if self._n_res else n, device=x.device)
            # detach instead of requires_grad=False: tools.RequiresGrad turns on every parameter in training
            log_tau = self.log_tau[i] if self._learn_tau else self.log_tau[i].detach()
            lam = torch.exp(-torch.exp(-log_tau))
            gain = torch.sqrt(1 - lam**2) if self._input_norm == "lru" else 1 - lam
            pre = self.B[i](x)
            if self._feedback and self.neuron.mems[f"s{i}"] is not None:
                pre = pre + self.R[i](self.neuron.mems[f"s{i}"])
            inp = gain * self.in_norm[i](pre)
            if self._n_res:
                theta = self.theta[i] if self._learn_tau else self.theta[i].detach()
                theta = theta * self.res_mask  # integrator channels stay at θ = 0
                a, b = d[:, :n], d[:, n:]
                c, s = lam * torch.cos(theta), lam * torch.sin(theta)
                a, b = c * a - s * b + inp, s * a + c * b
                d = torch.cat([a, b], -1)
                u = ((a.view(-1, self._size, self._channels) * self.C[i]).sum(-1)
                     + (b.view(-1, self._size, self._channels) * self.C2[i]).sum(-1) + self.D[i](x))
            else:
                d = lam * d + inp
                u = (d.view(-1, self._size, self._channels) * self.C[i]).sum(-1) + self.D[i](x)
            self.neuron.mems[key] = d
            if self._soma == "lif":
                v = self.neuron.mems[f"v{i}"]
                v = self.soma_norm[i](u) + (0 if v is None else self.soma_decay * v)
                spike = self.act_fun(v - self.threshold)
                self.neuron.mems[f"v{i}"] = v * (1 - spike.detach())
                outputs = [spike] * self.T
            else:
                outputs = self.soma(self.soma_norm[i](u))
            x = sum(outputs) / self.T
            if self._feedback:
                self.neuron.mems[f"s{i}"] = x
        return outputs, [torch.stack(outputs, dim=0)]


class LIFRNN(nn.Module):
    """Baseline: a recurrent LIF layer, the usual spiking-RL practice. Each environment step runs
    T SNN steps, v += (PopNorm(W [x, s]) - v) / tau, hard reset on a spike; the recurrent input
    is the previous environment step's spikes at the same SNN step, as in MCRNN. The membrane
    is cleared at every environment step unless persist_state is on (intervention A for LIF).
    gain=True multiplies the input by sqrt(tau / 2), intervention G for LIF."""

    def __init__(self, config, input_size, size, act=None, norm="PopNorm", device="cuda"):
        super().__init__()
        p = config.LIFRNN
        self.T, self._size = config.spike_times, size
        self.tau, self.threshold = p["tau"], p["threshold"]
        self.gain = math.sqrt(self.tau / 2.0) if p["gain"] else 1.0
        self.neuron = PDState(["v"])
        self.mem_sizes = {"v": size}
        self.w = nn.Linear(input_size + size, size)
        self.norm = getattr(normalization, norm)(size, **getattr(config, norm))
        self.act_fun = node.QGateGrad(alpha=2.0, requires_grad=False)

    def forward(self, inputs, state):
        v = self.neuron.mems["v"]
        if v is None:
            v = torch.zeros(inputs[0].shape[0], self._size, device=inputs[0].device)
        outputs = []
        for step in range(self.T):
            v = v + (self.gain * self.norm(self.w(torch.cat((inputs[step], state[0][step]), -1))) - v) / self.tau
            spike = self.act_fun(v - self.threshold)
            v = v * (1.0 - spike.detach())
            outputs.append(spike)
        self.neuron.mems["v"] = v
        return outputs, [torch.stack(outputs, dim=0)]


class GRSNCell(nn.Module):
    """Baseline: GRSN (Qin et al., AAAI 2025) in its temporal alignment paradigm, one update per
    environment step with the current c and the membrane u carried across steps:

        c_t = F ⊙ c_{t-1} + (1 - F) ⊙ I      F = sigmoid(W_f x + b_f), I = ReLU(W_i x + b_i)
        u_t = β ⊙ û_{t-1} + (1 - β) ⊙ c_t    o_t = H(u_t - θ)    û_t = u_t - θ o_t (soft reset)

    β is learnable per neuron. x is the rate of the T input spikes and o_t is repeated T times,
    as in PDRNN, so that the rest of the world model is shared. Adaptation: a LayerNorm on the
    input-gate pre-activation puts I at the input scale of the other cells; without it the
    neurons never reach θ in this world model (firing rate 0). Use with persist_state."""

    def __init__(self, config, input_size, size, act=None, norm="PopNorm", device="cuda"):
        super().__init__()
        p = config.GRSN
        self.T, self._size, self.threshold = config.spike_times, size, p["threshold"]
        self.neuron = PDState(["c", "u"])
        self.mem_sizes = {"c": size, "u": size}
        self.forget = nn.Linear(input_size, size)
        self.inp = nn.Linear(input_size, size)
        self.inp_norm = nn.LayerNorm(size, eps=1e-3)
        beta = torch.full((size,), float(p["beta_init"]))
        self.beta_logit = nn.Parameter(torch.log(beta / (1 - beta)))
        self.act_fun = node.QGateGrad(alpha=2.0, requires_grad=False)

    def forward(self, inputs, state):
        x = sum(inputs) / self.T
        c, u = self.neuron.mems["c"], self.neuron.mems["u"]
        if c is None:
            c = torch.zeros(x.shape[0], self._size, device=x.device)
            u = torch.zeros_like(c)
        f = torch.sigmoid(self.forget(x))
        c = f * c + (1 - f) * F.relu(self.inp_norm(self.inp(x)))
        beta = torch.sigmoid(self.beta_logit)
        u = beta * u + (1 - beta) * c
        spike = self.act_fun(u - self.threshold)
        u = u - self.threshold * spike
        self.neuron.mems["c"], self.neuron.mems["u"] = c, u
        outputs = [spike] * self.T
        return outputs, [torch.stack(outputs, dim=0)]


class Conv2dSame(torch.nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret



class ChNorm(nn.Module):
    def __init__(self, ch, norm, norm_p):
        super(ChNorm, self).__init__()
        self.norm = norm(ch, **norm_p)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


# 
