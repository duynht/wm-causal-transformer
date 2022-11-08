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
        discount=0.99, 
        lr=0.001, 
        gae_lambda=0.95,
        entropy_coef=0.01, 
        value_loss_coef=0.5, 
        max_grad_norm=0.5, 
        recurrence=4,
        adam_eps=1e-8, 
        clip_eps=0.2, 
        epochs=4, 
        batch_size=256, 
        preprocess_obss=None,
        reshape_reward=None
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
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        # Store parameters

        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        # self.discount = discount
        # self.lr = lr
        # self.gae_lambda = gae_lambda
        # self.entropy_coef = entropy_coef
        # self.value_loss_coef = value_loss_coef
        # self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward

        # Control parameters

        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

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
            self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)
        
        # self.values = torch.zeros(*shape, device=self.device)
        # self.advantages = torch.zeros(*shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

        
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.batch_num = 0

        self.loss_fn = CrossEntropyLoss()
        self.attn_mask = generate_square_subsequent_mask(self.num_frames_per_proc).to(self.device)

    def update(self):
        self.env.reset()
        acc = 0
        batch_loss = 0

        if self.acmodel.recurrent:
            # self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(self.num_frames_per_proc, self.num_procs, self.acmodel.memory_size, device=self.device)
            # self.drop_mask = torch.zeros(self.num_frames_per_proc, self.num_procs, self.acmodel.d_model, device=self.device))
        
        goal = F.one_hot(torch.full((self.num_frames_per_proc, self.num_procs), self.acmodel.naction - 1)).float().to(self.device)
        for i in range(self.num_frames_per_proc):
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)

            # logits, self.memories = self.acmodel(preprocessed_obs, self.memories, i, self.attn_mask, return_dist=False)
            
            # action = torch.argmax(logits[i], dim=1)
            # goal[i] = F.one_hot(preprocessed_obs.goal).float().to(self.device)

            # batch_loss += self.loss_fn(logits[:i+1], goal[:i+1].clone())

            dist, self.memories = self.acmodel(preprocessed_obs, self.memories, i, self.attn_mask)
            action = dist.sample()
            logits = dist.logits

            batch_loss += self.loss_fn(logits, preprocessed_obs.goal)
            # breakpoint()

            
            acc += torch.sum(action == preprocessed_obs.goal).item()

            obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())


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

    # def collect_experiences(self):
    #     """Collects rollouts and computes advantages.

    #     Runs several environments concurrently. The next actions are computed
    #     in a batch mode for all environments at the same time. The rollouts
    #     and advantages from all environments are concatenated together.

    #     Returns
    #     -------
    #     exps : DictList
    #         Contains actions, rewards, advantages etc as attributes.
    #         Each attribute, e.g. `exps.reward` has a shape
    #         (self.num_frames_per_proc * num_envs, ...). k-th block
    #         of consecutive `self.num_frames_per_proc` frames contains
    #         data obtained from the k-th environment. Be careful not to mix
    #         data from different environments!
    #     logs : dict
    #         Useful stats about the training process, including the average
    #         reward, policy loss, value loss, etc.
    #     """

    #     for i in range(self.num_frames_per_proc):
    #         # Do one agent-environment interaction

    #         preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
    #         # with torch.no_grad():
    #         #     if self.acmodel.recurrent:
    #         #         dist, memory = self.acmodel(preprocessed_obs, self.memories)
    #         #         # dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
    #         #     else:
    #         #         dist, value = self.acmodel(preprocessed_obs)
    #         # action = dist.sample()

    #         dist, memory = self.acmodel(preprocessed_obs, self.memories, i)
    #         action = torch.argmax(dist, dim=1)

    #         batch_loss += self.loss_fn(action, obs.goal)

    #         obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())

    #         done = tuple(a | b for a, b in zip(terminated, truncated))

    #         # Update experiences values

    #         self.obss[i] = self.obs
    #         self.obs = obs
    #         # if self.acmodel.recurrent:
    #         #     self.memories[i] = self.memory
    #         #     self.memory = memory
    #         # self.masks[i] = self.mask
    #         self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
    #         # self.actions[i] = action
    #         # # self.values[i] = value
    #         # if self.reshape_reward is not None:
    #         #     self.rewards[i] = torch.tensor([
    #         #         self.reshape_reward(obs_, action_, reward_, done_)
    #         #         for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
    #         #     ], device=self.device)
    #         # else:
    #         #     self.rewards[i] = torch.tensor(reward, device=self.device)
    #         # self.log_probs[i] = dist.log_prob(action)

    #         # # Update log values

    #         # self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
    #         # self.log_episode_reshaped_return += self.rewards[i]
    #         # self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

    #         # for i, done_ in enumerate(done):
    #         #     if done_:
    #         #         self.log_done_counter += 1
    #         #         self.log_return.append(self.log_episode_return[i].item())
    #         #         self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
    #         #         self.log_num_frames.append(self.log_episode_num_frames[i].item())

    #         # self.log_episode_return *= self.mask
    #         # self.log_episode_reshaped_return *= self.mask
    #         # self.log_episode_num_frames *= self.mask

    #     # Add advantage and return to experiences

    #     # preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
    #     # with torch.no_grad():
    #     #     if self.acmodel.recurrent:
    #     #         _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
    #     #     else:
    #     #         _, next_value = self.acmodel(preprocessed_obs)

    #     for i in reversed(range(self.num_frames_per_proc)):
    #         next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
    #         # next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
    #         # next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

    #         # delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
    #         # self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask
        
    #     self.optimizer.zero_grad()
    #     batch_loss.backward()
    #     self.optimizer.step()

    #     # Define experiences:
    #     #   the whole experience is the concatenation of the experience
    #     #   of each process.
    #     # In comments below:
    #     #   - T is self.num_frames_per_proc,
    #     #   - P is self.num_procs,
    #     #   - D is the dimensionality.

    #     exps = DictList()
    #     exps.obs = [self.obss[i][j]
    #                 for j in range(self.num_procs)
    #                 for i in range(self.num_frames_per_proc)]

        
    #     if self.acmodel.recurrent:
    #         # T x P x D -> P x T x D -> (P * T) x D
    #         exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
    #         # exps.memory = self.memories.transpose(0, 1).reshape(-1)
    #         # T x P -> P x T -> (P * T) x 1
    #         exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
    #     # for all tensors below, T x P -> P x T -> P * T
    #     exps.action = self.actions.transpose(0, 1).reshape(-1)
    #     # exps.value = self.values.transpose(0, 1).reshape(-1)
    #     exps.reward = self.rewards.transpose(0, 1).reshape(-1)
    #     # exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
    #     # exps.returnn = exps.value + exps.advantage
    #     exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

    #     # Preprocess experiences

    #     exps.obs = self.preprocess_obss(exps.obs, device=self.device)

    #     # Log some values

    #     keep = max(self.log_done_counter, self.num_procs)

    #     logs = {
    #         "return_per_episode": self.log_return[-keep:],
    #         "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
    #         "num_frames_per_episode": self.log_num_frames[-keep:],
    #         "num_frames": self.num_frames
    #     }

    #     self.log_done_counter = 0
    #     self.log_return = self.log_return[-self.num_procs:]
    #     self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
    #     self.log_num_frames = self.log_num_frames[-self.num_procs:]

    #     return exps, logs

    # def update_parameters(self, exps):
    #     # Collect experiences

    #     for _ in range(self.epochs):
    #         # Initialize log values

    #         # log_entropies = []
    #         # log_values = []
    #         # log_policy_losses = []
    #         # log_value_losses = []
    #         # log_grad_norms = []

    #         for inds in self._get_batches_starting_indexes():
    #             # Initialize batch values

    #             batch_entropy = 0
    #             batch_value = 0
    #             batch_policy_loss = 0
    #             batch_value_loss = 0
    #             batch_loss = 0

    #             # Initialize memory

    #             if self.acmodel.recurrent:
    #                 memory = exps.memory[inds]

    #             for i in range(self.recurrence):
    #                 # Create a sub-batch of experience

    #                 sb = exps[inds + i]

    #                 # Compute loss

    #                 if self.acmodel.recurrent:
    #                     dist, value, memory = self.acmodel(sb.obs, memory * sb.mask)
    #                 else:
    #                     dist, value = self.acmodel(sb.obs)

    #                 # entropy = dist.entropy().mean()

    #                 # ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
    #                 # surr1 = ratio * sb.advantage
    #                 # surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
    #                 # policy_loss = -torch.min(surr1, surr2).mean()

    #                 # value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
    #                 # surr1 = (value - sb.returnn).pow(2)
    #                 # surr2 = (value_clipped - sb.returnn).pow(2)
    #                 # value_loss = torch.max(surr1, surr2).mean()

    #                 # loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

    #                 loss = self.loss_fn(logits, )

    #                 # Update batch values

    #                 # batch_entropy += entropy.item()
    #                 # batch_value += value.mean().item()
    #                 # batch_policy_loss += policy_loss.item()
    #                 # batch_value_loss += value_loss.item()

    #                 batch_loss += loss

    #                 # Update memories for next epoch

    #                 if self.acmodel.recurrent and i < self.recurrence - 1:
    #                     exps.memory[inds + i + 1] = memory.detach()

    #             # Update batch values

    #             # batch_entropy /= self.recurrence
    #             # batch_value /= self.recurrence
    #             # batch_policy_loss /= self.recurrence
    #             # batch_value_loss /= self.recurrence
    #             batch_loss /= self.recurrence

    #             # Update actor-critic

    #             self.optimizer.zero_grad()
    #             batch_loss.backward()
    #             # grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
    #             # torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
    #             self.optimizer.step()

    #             # Update log values

    #             # log_entropies.append(batch_entropy)
    #             # log_values.append(batch_value)
    #             # log_policy_losses.append(batch_policy_loss)
    #             # log_value_losses.append(batch_value_loss)
    #             # log_grad_norms.append(grad_norm)

    #     # Log some values

    #     logs = {
    #         "entropy": numpy.mean(log_entropies),
    #         "value": numpy.mean(log_values),
    #         "policy_loss": numpy.mean(log_policy_losses),
    #         "value_loss": numpy.mean(log_value_losses),
    #         "grad_norm": numpy.mean(log_grad_norms)
    #     }

    #     return logs

    # def _get_batches_starting_indexes(self):
    #     """Gives, for each batch, the indexes of the observations given to
    #     the model and the experiences used to compute the loss at first.

    #     First, the indexes are the integers from 0 to `self.num_frames` with a step of
    #     `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
    #     more diverse batches. Then, the indexes are splited into the different batches.

    #     Returns
    #     -------
    #     batches_starting_indexes : list of list of int
    #         the indexes of the experiences to be used at first for each batch
    #     """

    #     indexes = numpy.arange(0, self.num_frames, self.recurrence)
    #     indexes = numpy.random.permutation(indexes)

    #     # Shift starting indexes by self.recurrence//2 half the time
    #     if self.batch_num % 2 == 1:
    #         indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
    #         indexes += self.recurrence // 2
    #     self.batch_num += 1

    #     num_indexes = self.batch_size // self.recurrence
    #     batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

    #     return batches_starting_indexes