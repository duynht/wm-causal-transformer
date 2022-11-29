import torch

import utils
from .other import device
from model import ACModel
from visual_transformer import CausalVisionTransformer
from gpt import DecisionTransformer
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
        # self.acmodel = CausalVisionTransformer(
        #     in_channels=obs_space["image"][2],
        #     out_channels=32,
        #     kernel_size=3,
        #     obs_space=obs_space,
        #     action_space=action_space,
        #     d_model=args.d_model,
        #     nhead=args.nhead,
        #     d_hid=args.d_model,
        #     nlayers=args.nlayers,
        #     max_len=args.max_delay_frames + 3
        # )

        self.model = DecisionTransformer(
            state_dim=2,
            act_dim=action_space.n,
            n_blocks=args.nlayers,
            h_dim=args.d_model,
            n_heads=args.nhead,
            context_len=args.max_delay_frames+3,
            drop_p=0.5,
            obs_space=obs_space,
        )
        
        self.get_embed = False
        self.argmax = args.argmax
        self.num_envs = num_envs
        self.max_frames = args.max_delay_frames + 3
        self.model.load_state_dict(utils.get_model_state(model_dir))
        self.model.to(device)
        self.model.eval()
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))

        # self.attn_mask = generate_square_subsequent_mask(self.acmodel.max_len).to(device)
        # self.goal = torch.full([1, self.acmodel.max_len], self.acmodel.naction - 1).to(device)
        
        self.timesteps = torch.arange(self.max_frames).expand(self.num_envs, self.max_frames).to(device)
        
        self.reset_()

    def reset_(self):
        self.memory = None
        self.states = torch.full([self.num_envs, 1], 0).long().to(device)
        self.goals = torch.full([self.num_envs, 1], self.model.act_dim - 1).to(device)
        
    def get_actions(self, obss, step):
        preprocessed_obss = self.preprocess_obss(obss, device=device)

        with torch.no_grad():
            if self.get_embed:
                # logits, embed, _, _ = self.acmodel(preprocessed_obss.image, self.goal, step, self.attn_mask, return_embed=True, return_dist=False)
                act_logits, state_logits, self.memory, embed = self.model(preprocessed_obss.image, self.memory, self.timesteps[:, :step+1], self.states, self.goals, return_embed=True)
            else: 
                act_logits, state_logits, self.memory = self.model(preprocessed_obss.image, self.memory, self.timesteps[:, :step+1], self.states, self.goals)
            self.states = torch.cat([self.states, preprocessed_obss.asked.long().unsqueeze(1)], dim=1)
            self.goals = torch.cat([self.goals, preprocessed_obss.goal.long().unsqueeze(1)], dim=1)
            
        actions = torch.argmax(act_logits[:,step], dim=-1)
        actions = actions.squeeze()

        if self.get_embed:
            return actions.cpu().numpy(), embed.cpu().numpy()
        else:
            return actions.cpu().numpy()

    def get_action(self, obs, step):
        return self.get_actions([obs], step)

    def analyze_feedbacks(self, rewards, dones):
        masks = 1 - torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(1)
        acc = torch.sum(torch.tensor(rewards)) / torch.sum(torch.arange(0, self.max_frames) * masks)
        return acc.cpu().numpy()
        # if self.acmodel.recurrent:
        #     self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
