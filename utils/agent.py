import torch

import utils
from .other import device
from model import ACModel
from visual_transformer import CausalVisionTransformer
from torch import Tensor

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_dir, args, num_envs=1):
        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(obs_space)
        # self.acmodel = ACModel(obs_space, action_space, use_memory=use_memory, use_text=use_text)
        self.acmodel = CausalVisionTransformer(
            in_channels=obs_space["image"][2],
            out_channels=32,
            kernel_size=3,
            obs_space=obs_space,
            action_space=action_space,
            d_model=args.d_model,
            nhead=args.nhead,
            d_hid=args.d_model,
            nlayers=args.nlayers,
            max_len=args.max_delay_frames + 3
        )
        self.argmax = args.argmax
        self.num_envs = num_envs

        # if self.acmodel.recurrent:
            # self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=device)
            # self.memories = torch.zeros(self.acmodel.max_len, self.num_envs, self.acmodel.memory_size, device=device)

        self.acmodel.load_state_dict(utils.get_model_state(model_dir))
        self.acmodel.to(device)
        self.acmodel.eval()
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))

        self.attn_mask = generate_square_subsequent_mask(self.acmodel.max_len).to(device)
        self.goal = torch.full([1, self.acmodel.max_len], self.acmodel.naction - 1).to(device)

    def get_actions(self, obss, step):
        # breakpoint()
        preprocessed_obss = self.preprocess_obss(obss, device=device)

        with torch.no_grad():
            
            # if self.acmodel.recurrent:
            #     dist, self.memories, = self.acmodel(preprocessed_obss, self.memories, step, self.attn_mask, return_dist=(~self.argmax))
            # else:
            #     dist, _ = self.acmodel(preprocessed_obss, attn_mask=self.attn_mask, return_dist=(~self.argmax))

            dist, embed = self.acmodel(preprocessed_obss.image, self.goal, step, self.attn_mask, return_embed=True, return_dist=False)

        if self.argmax:
            # actions = dist.probs.max(1, keepdim=True)[1]
            actions = torch.argmax(dist[...,step], dim=1)
        else:
            actions = dist.sample()

        actions = actions.squeeze()

        if not torch.sum(preprocessed_obss.asked):
            actions = torch.tensor([16])
        # else:
        #     # breakpoint()
        #     print((self.acmodel.step - 1) % self.acmodel.max_len, ':', actions)

        return actions.cpu().numpy()

    def get_action(self, obs, step):
        return self.get_actions([obs], step)

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
