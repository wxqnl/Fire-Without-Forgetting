"""Persistent dendritic world-model dynamics, with exact affine scan recurrence.

Each environment step is one neuronal step. Only binary spikes pass between
SSM layers. The complex dendritic state never receives spike-reset feedback.
Observation-only posteriors make teacher-forced training parallel in time.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from networks import RSSM, SpikeMLP
from normalization import PopNorm
from surrogate import QGateGrad
import tools


def affine_scan(a, b, initial):
    """Inclusive differentiable scan of x_t = a_t*x_(t-1) + b_t.

    a has shape [batch,time,state]; episode starts set a_t=0. The same
    recurrence is used by step(); no temporal convolution or approximation.
    """
    aa, bb = a, b
    stride = 1
    while stride < b.shape[1]:
        bb = torch.cat((bb[:, :stride], bb[:, stride:] + aa[:, stride:] * bb[:, :-stride]), dim=1)
        aa = torch.cat((aa[:, :stride], aa[:, stride:] * aa[:, :-stride]), dim=1)
        stride *= 2
    return bb + aa * initial[:, None]


class DendriticLayer(nn.Module):
    def __init__(self, inputs, width, states, min_tau, max_tau, threshold):
        super().__init__()
        # Parameterize decay in environment steps; |lambda| always below 1.
        self.log_tau = nn.Parameter(torch.linspace(math.log(min_tau), math.log(max_tau), states))
        self.phase = nn.Parameter(torch.linspace(0.0, math.pi, states))
        self.B_re = nn.Parameter(torch.randn(states, inputs) / math.sqrt(2 * inputs))
        self.B_im = nn.Parameter(torch.randn(states, inputs) / math.sqrt(2 * inputs))
        self.C_re = nn.Parameter(torch.randn(width, states) / math.sqrt(states))
        self.C_im = nn.Parameter(torch.randn(width, states) / math.sqrt(states))
        self.D = nn.Linear(inputs, width, bias=False)
        nn.init.xavier_uniform_(self.D.weight)
        # Local somatic normalization; never fed back into the dendritic state.
        self.soma_norm = PopNorm(width, threshold=threshold, v_reset=0.0, eps=1e-3)
        self.spike = QGateGrad(alpha=2.0, requires_grad=False)
        self.threshold = threshold
        self.states = states

    def coefficients(self):
        tau = self.log_tau.clamp(math.log(1.0), math.log(1e6)).exp()
        radius = torch.exp(-1.0 / tau)
        lam = torch.polar(radius, self.phase)
        # Variance-preserving input scaling over different decay constants.
        scale = torch.sqrt((1.0 - radius.square()).clamp_min(1e-8))
        return lam, scale

    def input_current(self, x, scale):
        return torch.complex(F.linear(x, self.B_re), F.linear(x, self.B_im)) * scale

    def readout(self, state, x):
        u = F.linear(state.real, self.C_re) - F.linear(state.imag, self.C_im) + self.D(x)
        return self.spike(self.soma_norm(u) - self.threshold)

    def forward(self, x, initial, is_first):
        lam, scale = self.coefficients()
        a = lam[None, None] * (1.0-is_first[..., None])
        b = self.input_current(x, scale)
        states = affine_scan(a, b, initial)
        return self.readout(states, x), states

    def step(self, x, initial, is_first):
        lam, scale = self.coefficients()
        state = lam * initial * (1.0-is_first[..., None]) + self.input_current(x, scale)
        return self.readout(state, x), state


class PersistentSSM(nn.Module):
    get_feat = RSSM.get_feat
    get_dist = RSSM.get_dist
    kl_loss = RSSM.kl_loss

    def __init__(self, config, embed):
        super().__init__()
        if config.spike_times != 1 or config.dyn_temp_post:
            raise ValueError('Persistent SSM requires T=1 and an observation-only posterior.')
        self._stoch, self._discrete = config.dyn_stoch, config.dyn_discrete
        self._deter, self._hidden = config.dyn_deter, config.dyn_hidden
        self._unimix_ratio, self._device = config.unimix_ratio, config.device
        self.T = 1
        self.depth, self.states = config.ssm_layers, config.ssm_states
        inputs = self._stoch*self._discrete + config.num_actions
        self.layers = nn.ModuleList([
            DendriticLayer(inputs if i==0 else self._deter, self._deter,
                           self.states, config.ssm_min_tau, config.ssm_max_tau,
                           config.ssm_threshold) for i in range(self.depth)])
        kw = dict(shape=None,layers=config.dyn_output_layers,units=self._hidden,
                  act=config.act,act_p=getattr(config,config.act),norm=config.norm,
                  norm_p=getattr(config,config.norm),spike_times=1)
        self.posterior = SpikeMLP(embed, **kw)
        self.prior = SpikeMLP(self._deter, **kw)
        self.post_logits = nn.Linear(self._hidden,self._stoch*self._discrete)
        self.prior_logits = nn.Linear(self._hidden,self._stoch*self._discrete)
        self.post_logits.apply(tools.weight_init)
        self.prior_logits.apply(tools.weight_init)

    def initial(self, batch_size):
        ref = next(self.parameters())
        return dict(logit=ref.new_zeros(batch_size,self._stoch,self._discrete),
                    stoch=ref.new_zeros(batch_size,self._stoch,self._discrete),
                    deter=ref.new_zeros(1,batch_size,self._deter),
                    ssm_real=ref.new_zeros(batch_size,self.depth,self.states),
                    ssm_imag=ref.new_zeros(batch_size,self.depth,self.states))

    def posterior_stats(self, embed):
        x = self.posterior(embed)[0]
        return self.post_logits(x).reshape(*x.shape[:-1],self._stoch,self._discrete)

    def prior_stats(self, spikes):
        x = self.prior(spikes.unsqueeze(0))[0]
        return self.prior_logits(x).reshape(*x.shape[:-1],self._stoch,self._discrete)

    def sample(self, logit, sample):
        dist=self.get_dist({'logit':logit})
        return dist.sample() if sample else dist.mode()

    def filter(self, x, state, is_first, parallel=True):
        ds=[]
        for i,layer in enumerate(self.layers):
            init=torch.complex(state['ssm_real'][...,i,:],state['ssm_imag'][...,i,:])
            if parallel:
                x,d=layer(x,init,is_first)
            else:
                x,d=layer.step(x,init,is_first)
            ds.append(d)
        d=torch.stack(ds,dim=-2)
        return x,d.real,d.imag

    def observe(self, embed, action, is_first, state=None, sample=True):
        if state is None: state=self.initial(action.shape[0])
        logit=self.posterior_stats(embed)
        stoch=self.sample(logit,sample)
        prev=torch.cat((state['stoch'][:,None],stoch[:,:-1]),dim=1)
        mask=1.0-is_first[...,None]
        x=torch.cat((prev.flatten(-2),action),dim=-1)*mask
        spikes,re,im=self.filter(x,state,is_first,parallel=True)
        prior_logit=self.prior_stats(spikes)
        shared=dict(deter=spikes.unsqueeze(0),ssm_real=re,ssm_imag=im)
        return (dict(logit=logit,stoch=stoch,**shared),
                dict(logit=prior_logit,stoch=self.sample(prior_logit,sample),**shared))

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        prior=self.img_step(prev_state,prev_action,sample=sample,is_first=is_first)
        logit=self.posterior_stats(embed)
        post=dict(prior,logit=logit,stoch=self.sample(logit,sample))
        return post,prior

    def img_step(self, prev_state, prev_action, embed=None, sample=True, is_first=None):
        if is_first is None: is_first=prev_action.new_zeros(prev_action.shape[:-1])
        x=torch.cat((prev_state['stoch'].flatten(-2),prev_action),dim=-1)*(1.0-is_first[...,None])
        spikes,re,im=self.filter(x,prev_state,is_first,parallel=False)
        logit=self.prior_stats(spikes)
        return dict(logit=logit,stoch=self.sample(logit,sample),deter=spikes.unsqueeze(0),ssm_real=re,ssm_imag=im)

    def imagine(self, action, state=None, sample=True):
        if state is None: state=self.initial(action.shape[0])
        values=[]
        for t in range(action.shape[1]):
            state=self.img_step(state,action[:,t],sample=sample)
            values.append(state)
        return {k:torch.stack([s[k] for s in values],dim=2 if k=='deter' else 1) for k in values[0]}

    def diagnostics(self, post):
        taus=torch.cat([l.log_tau.detach().exp() for l in self.layers])
        return dict(ssm_spike_rate=post['deter'].float().mean(),
                    ssm_state_rms=(post['ssm_real'].square()+post['ssm_imag'].square()).mean().sqrt(),
                    ssm_tau_min=taus.min(),ssm_tau_median=taus.median(),ssm_tau_max=taus.max())
