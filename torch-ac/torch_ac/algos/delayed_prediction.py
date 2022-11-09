from abc import ABC, abstractmethod

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv

import numpy
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch import Tensor

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
        if self.acmodel.recurrent:
            # self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        
        # self.values = torch.zeros(*shape, device=self.device)
        # self.advantages = torch.zeros(*shape, device=self.device)

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.batch_num = 0

        # self.loss_fn = CrossEntropyLoss(ignore_index=self.acmodel.naction - 1)
        # weight = torch.ones(self.acmodel.naction).to(self.device)
        # weight[-1] = 1/self.acmodel.naction
        # self.loss_fn = CrossEntropyLoss(weight=weight)
        self.loss_fn = CrossEntropyLoss(reduction='none')
        self.attn_mask = generate_square_subsequent_mask(self.num_frames_per_proc).to(self.device)

    def update(self):
        self.env.reset()
        acc = 0
        batch_loss = 0

        if self.acmodel.recurrent:
            # self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(self.num_frames_per_proc, self.num_procs, self.acmodel.memory_size, device=self.device)
            # self.drop_mask = torch.zeros(self.num_frames_per_proc, self.num_procs, self.acmodel.d_model, device=self.device))
        
        # goal = F.one_hot(torch.full((self.num_frames_per_proc, self.num_procs), self.acmodel.naction - 1)).float().to(self.device)
        for i in range(self.num_frames_per_proc):
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            
            logits, self.memories = self.acmodel(preprocessed_obs, self.memories, i, self.attn_mask, return_dist=False)
            
            action = torch.argmax(logits[i], dim=1)
            # goal[i] = F.one_hot(preprocessed_obs.goal).float().to(self.device)
            # action = dist.sample()
            # logits = dist.logits

            # batch_loss += self.loss_fn(action, preprocessed_obs.goal)
            # breakpoint()
            # batch_loss += self.loss_fn(logits[:i+1], goal[:i+1].clone())

            # goal = F.one_hot(preprocessed_obs.goal).float().to(self.device)
            loss = self.loss_fn(logits[i], preprocessed_obs.goal)
            batch_loss += loss[self.mask.bool()].mean()
            
            acc += torch.sum(action == preprocessed_obs.goal).item()

            obs, _, terminated, truncated, _ = self.env.step(action.cpu().numpy())


            done = tuple(a | b for a, b in zip(terminated, truncated))
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.obss[i] = self.obs
            self.obs = obs
            self.memories[i] = self.memories[i] * self.mask.unsqueeze(1)

        # breakpoint()
        # batch_loss = self.loss_fn(logits, goal)

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        acc /= (self.num_procs * self.num_frames_per_proc)

        logs = {
            "loss": batch_loss.item(),
            "acc": acc,
            "num_frames": self.num_frames,
        }

        return logs
