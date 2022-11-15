from abc import ABC, abstractmethod

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv

import numpy
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch import Tensor
from icecream import ic

ic.disable()

    
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
        
def get_std_opt(model):
    return NoamOpt(model.d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class DMTSAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(
        self,
        envs, 
        acmodel, 
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
        acmodel : torch.Module
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
        self.acmodel = acmodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        # self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        # Control parameters

        # assert self.acmodel.recurrent or self.recurrence == 1
        # assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel

        self.acmodel.to(self.device)
        self.acmodel.train()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None] * (shape[0])
        # if self.acmodel.recurrent:
            # self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            # self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        # self.mask = torch.ones(shape[1], device=self.device)
        # self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        
        # self.values = torch.zeros(*shape, device=self.device)
        # self.advantages = torch.zeros(*shape, device=self.device)

        # self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.optimizer = get_std_opt(self.acmodel)
        self.batch_num = 0

        # self.loss_fn = CrossEntropyLoss()
        self.loss_fn = CrossEntropyLoss(ignore_index=self.acmodel.naction - 1, reduction='none')
        # weight = torch.ones(self.acmodel.naction).to(self.device)
        # weight = weight * 1e2
        # weight[-1] = self.acmodel.naction**(-2)
        # self.loss_fn = CrossEntropyLoss(weight=weight)
        # self.loss_fn = CrossEntropyLoss(reduction='none')
        self.attn_mask = generate_square_subsequent_mask(self.num_frames_per_proc).to(self.device)
        # self.attn_mask = None

    def update(self,):
        self.obs = self.env.reset()
        acc = 0
        batch_loss = 0
        goal = torch.full([self.num_procs], self.acmodel.naction - 1).unsqueeze(1).to(self.device)
        buffer = None
        

        # if self.acmodel.recurrent:
            # self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            # self.memories = torch.zeros(self.num_frames_per_proc, self.num_procs, self.acmodel.memory_size, device=self.device, requires_grad=True)
            # self.drop_mask = torch.zeros(self.num_frames_per_proc, self.num_procs, self.acmodel.d_model, device=self.device))
        self.acmodel.reset_memory()
        # goal = F.one_hot(torch.full((self.num_frames_per_proc, self.num_procs), self.acmodel.naction - 1)).float().to(self.device)
        for i in range(self.num_frames_per_proc):
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            
            if buffer is None:
                buffer = torch.zeros([self.num_frames_per_proc, *preprocessed_obs.image.shape], device=self.device)
            
            buffer[i] = preprocessed_obs.image
            
            logits, embed = self.acmodel(buffer, i, self.attn_mask, return_embed=True, return_dist=False)
            # logits, embed = self.acmodel(preprocessed_obs.image, i, self.attn_mask, return_embed=True, return_dist=False)
            
            action = torch.argmax(logits[...,i], dim=1)
            
            # breakpoint()
            if i > 0:
                goal = torch.cat([goal, preprocessed_obs.goal.unsqueeze(1)],  dim=1).to(self.device)
            # goal[i] = F.one_hot(preprocessed_obs.goal).float().to(self.device)
            # action = dist.sample()
            # logits = dist.logits

            # batch_loss += self.loss_fn(action, preprocessed_obs.goal)
            if torch.sum(preprocessed_obs.asked):
                ic(i)
                ic(preprocessed_obs.asked)
                ic(preprocessed_obs.goal)
                ic(action)
                # ic(embed)
            
            loss = self.loss_fn(logits[..., :i+1], goal)[preprocessed_obs.asked == True].mean()
            # batch_loss += loss
            
            if not torch.isnan(loss):
                batch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # loss = self.loss_fn(logits[..., i], preprocessed_obs.goal)[preprocessed_obs != 16].mean()
            # if not torch.isnan(loss):
            #     batch_loss += loss.item()
            # batch_loss += self.loss_fn(logits[...,i], preprocessed_obs.goal)[preprocessed_obs.goal != 16]

            # goal = F.one_hot(preprocessed_obs.goal).float().to(self.device)
            
            # loss = self.loss_fn(logits[i], preprocessed_obs.goal)
            # batch_loss += loss
            # batch_loss += loss[self.mask.bool()].mean()
            
            # acc += torch.sum((action == preprocessed_obs.goal)).item()
            acc += torch.sum((action == preprocessed_obs.goal)[preprocessed_obs.goal != 16]).item() + torch.sum(preprocessed_obs.goal == 16)

            obs, _, terminated, truncated, _ = self.env.step(action.cpu().numpy())


            # done = tuple(a | b for a, b in zip(terminated, truncated))
            # self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.obss[i] = self.obs
            self.obs = obs
            # self.memories[i].data = self.memories[i].data * self.mask.unsqueeze(1)

        # breakpoint()
        # batch_loss = self.loss_fn(logits, goal)

        # self.optimizer.zero_grad()
        # batch_loss.backward()
        # # breakpoint()
        # self.optimizer.step()

        acc = acc / (self.num_procs * self.num_frames_per_proc)

        logs = {
            "loss": batch_loss,
            "acc": acc,
            "num_frames": self.num_frames,
            "final_input": (preprocessed_obs.image, torch.tensor(i), self.attn_mask, torch.tensor(True), torch.tensor(False)),
            "img": preprocessed_obs.image,
            "labels": preprocessed_obs.goal,
            "embed": embed[-1],
        }

        return logs
