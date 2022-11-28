from abc import ABC, abstractmethod

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv

import numpy
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch import Tensor
from icecream import ic
    
def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

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
        
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr, eps=adam_eps)
        self.optimizer = get_std_opt(self.model)
        self.batch_num = 0

        # self.loss_fn = CrossEntropyLoss(label_smoothing=0.2)
        # self.loss_fn = CrossEntropyLoss(ignore_index=self.model.naction - 1, label_smoothing=0.3, reduction='none')
        
        weight = torch.ones(self.model.naction).to(self.device)
        weight[-1] = (self.num_procs * self.num_frames_per_proc) ** (-1)
        self.loss_fn = CrossEntropyLoss(weight=weight, label_smoothing=0.1)
       
        # self.loss_fn = CrossEntropyLoss(reduction='none')
        self.attn_mask = generate_square_subsequent_mask(self.num_frames_per_proc).to(self.device)
        # self.attn_mask = None
        
        
    def update(self):
        self.obs = self.env.reset()
        acc = 0
        batch_loss = 0
        goal = torch.full([self.num_procs, self.num_frames_per_proc], self.model.naction - 1).to(self.device)
        self.model.reset_memory_()
        start = [None for _ in range(self.num_procs)]
        for i in range(self.num_frames_per_proc):
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            if i > 0:
                # goal = torch.cat([goal, preprocessed_obs.goal.unsqueeze(1)],  dim=1).to(self.device)
                goal[:, i] = preprocessed_obs.goal
            # logits, embed = self.model(preprocessed_obs.image, i, self.attn_mask, return_embed=True, return_dist=False)
            logits, embed = self.model(preprocessed_obs.image, goal.clone(), i, self.attn_mask, return_embed=True, return_dist=False)
                       
            logits = logits.transpose(-1, -2)
            
            action = torch.argmax(logits[..., i], dim=1)
            

            # if start is None and torch.any(preprocessed_obs.asked):
            #     start = i
            
            for p in range(self.num_procs):
                if start[p] is None and preprocessed_obs.asked[p]:
                    start[p] = i
            
            # self.optimizer.zero_grad()
            # loss = self.loss_fn(logits[..., i], preprocessed_obs.goal)
            # batch_loss += loss.item()
            # loss.backward(retain_graph=True)
            # self.optimizer.step() 
            
            # loss = self.loss_fn(logits[..., :i+1], goal)
            
            # loss = self.loss_fn(logits[..., i], preprocessed_obs.goal)
            
            # if not torch.isnan(loss):
            #     batch_loss += loss.item()
            #     self.optimizer.zero_grad()
            #     loss.backward(retain_graph=True)
            #     self.optimizer.step()
                
            # batch_loss += self.loss_fn(action, preprocessed_obs.goal)
            # if torch.sum(preprocessed_obs.asked):
            # if preprocessed_obs[0].asked:
            #     ic(i)
            #     # ic(preprocessed_obs.asked)
            #     ic(preprocessed_obs.goal[0])
            #     ic(action[0])
                # ic(embed)
                
            # acc += torch.sum(action == preprocessed_obs.goal).item()
            # acc += torch.sum((action == preprocessed_obs.goal)[preprocessed_obs.goal != 16]).item() + torch.sum(preprocessed_obs.goal == 16)

            # if torch.all(preprocessed_obs.asked):
            #     loss = self.loss_fn(logits[..., i], preprocessed_obs.goal)
            #     self.optimizer.zero_grad()
            #     if i == self.num_frames_per_proc - 1:
            #         loss.backward()
            #     else:
            #         loss.backward(retain_graph=True)
            #     self.optimizer.step()           
                
            obs, _, terminated, truncated, _ = self.env.step(action.cpu().numpy())

            self.obss[i] = self.obs
            self.obs = obs
            
        # acc /= (self.num_procs * self.num_frames_per_proc)
        # batch_loss /= self.num_frames_per_proc
        
        self.optimizer.zero_grad()
        # for p in range(self.num_procs):
        #     logits[p,...,:start[p]] = logits[p,...,start[p]-1].clone().unsqueeze(1)
        #     logits[p,...,start[p]:] = logits[p,...,start[p]].clone().unsqueeze(1)
        # batch_loss = self.loss_fn(logits[..., start-2:], goal[:, start-2:])
        batch_loss = self.loss_fn(logits, goal)
        batch_loss.backward()
        self.optimizer.step()
        action = torch.argmax(logits, dim=1)
        acc = torch.sum(action == goal).item() / (self.num_procs * self.num_frames_per_proc)

        ic(action == goal)
        
        logs = {
            "loss": batch_loss.item(),
            "acc": acc,
            "num_frames": self.num_frames,
            "final_input": (preprocessed_obs.image, torch.tensor(i), self.attn_mask, torch.tensor(True), torch.tensor(False)),
            "img": preprocessed_obs.image,
            "labels": preprocessed_obs.goal,
            "embed": embed[-1],
        }

        return logs