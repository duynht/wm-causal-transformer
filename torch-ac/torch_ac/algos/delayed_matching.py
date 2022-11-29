from abc import ABC, abstractmethod

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv

import numpy
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch import Tensor
from icecream import ic

def get_std_opt(model):
    return NoamOpt(model.d_model, 2, 500,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
            
    def load_state_dict(self, *args, **kwargs):
        self.optimizer.load_state_dict(*args, **kwargs)
        
    @property
    def optimizer(self):
        return self.optimizer
        


class DMTSAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(
        self,
        envs, 
        model, 
        device=None, 
        num_frames_per_proc=None, 
        lr=0.001, 
        adam_eps=1e-8, 
        preprocess_obss=None,
        # recurrence=4,
    ):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        model : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        lr : float
            the learning rate for optimizers
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        """

        # Store parameters

        self.env = ParallelEnv(envs)
        self.model = model
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.preprocess_obss = preprocess_obss or default_preprocess_obss

        # Configure model
        self.model.to(self.device)
        self.model.train()

        # Store helpers values
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values
        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None] * (shape[0])
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, eps=adam_eps)
        self.act_loss = CrossEntropyLoss(ignore_index=self.model.act_dim - 1)
        self.state_loss = CrossEntropyLoss()
        
        
    def update(self):
        self.obs = self.env.reset()
        acc = 0
        
        timesteps = torch.arange(self.num_frames_per_proc).expand(self.num_procs, self.num_frames_per_proc).to(self.device)
        states = torch.full([self.num_procs, 1], 0).long().to(self.device)
        goals = torch.full([self.num_procs, 1], self.model.act_dim-1).to(self.device)
        memory = None
        
        for i in range(self.num_frames_per_proc):
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
                        
            act_logits, state_logits, memory = self.model(preprocessed_obs.image, memory, timesteps[:, :i+1], states, goals)
            
            states = torch.cat([states, preprocessed_obs.asked.long().unsqueeze(1)], dim=1)
            goals = torch.cat([goals, preprocessed_obs.goal.long().unsqueeze(1)], dim=1)
            
            action = torch.argmax(act_logits[:, i], dim=-1)
                
            obs, *_ = self.env.step(action.cpu().numpy())

            self.obss[i] = self.obs
            self.obs = obs
            
        actions = torch.argmax(act_logits, dim=-1)
        # pred_states = torch.argmax(state_logits, dim=-1)
        # ic(actions[-1])
        # ic(pred_states)
        
        acc = torch.sum(actions == goals[:, 1:]).item() / (self.num_procs * self.num_frames_per_proc)
        self.optimizer.zero_grad()
        loss = self.act_loss(act_logits.transpose(-1,-2), goals[:, 1:]) + self.state_loss(state_logits.transpose(-1,-2), states[:, 1:])
        loss.backward()
        self.optimizer.step()
        
        logs = {
            "loss": loss.item(),
            "acc": acc,
            "num_frames": self.num_frames,
            # "final_input": (preprocessed_obs.image, torch.tensor(i), self.attn_mask, torch.tensor(True), torch.tensor(False)),
            # "img": preprocessed_obs.image,
            # "labels": preprocessed_obs.goal,
            # "embed": embed[-1],
        }

        return logs