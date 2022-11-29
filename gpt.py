import math
import torch
from torch import nn
import torch.nn.functional as F

class MaskedCausalAttention(nn.Module):
    def __init__(
        self,
        h_dim,
        max_T,
        n_heads,
        drop_p
    ) -> None:
        super().__init__()
        
        self.n_heads = n_heads
        self.max_T = max_T
        
        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)
        
        self.proj_net = nn.Linear(h_dim, h_dim)
        
        self.attn_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)
        
        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)
        
        self.register_buffer('mask', mask)
        
    def forward(self, x):
        B, T, C = x.shape # batch_size, seq_len, h_dim * n_heads
        
        N, D = self.n_heads, C // self.n_heads # n_heads, attn_dim
        
        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)
        
        # weights (B, N, T, T)
        weights = q @ k.transpose(2, 3) / math.sqrt(D)
        
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[...,:T, :T] == 0, float('-inf'))
        
        # normalize weights, -inf -> 0
        normalized_weights = F.softmax(weights, dim = -1)
        
        # attention (B, N, T, D)
        attention = self.attn_drop(normalized_weights @ v)
        
        attention = attention.transpose(1, 2).contiguous().view(B, T, N*D)
        
        out = self.proj_drop(self.proj_net(attention))
        
        return out
    
class Block(nn.Module):
    def __init__(
        self,
        h_dim,
        max_T,
        n_heads,
        drop_p
    ) -> None:
        super().__init__()
        
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 4 * h_dim),
            nn.GELU(),
            nn.Linear(4 * h_dim, h_dim),
            nn.Dropout(drop_p)
        )
        
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)
        
    def forward(self, x):
        x = x + self.attention(x)
        x = self.ln1(x)
        x = x + self.mlp(x)
        x = self.ln2(x)
        
        return x
    
class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        n_blocks,
        h_dim,
        context_len,
        n_heads,
        drop_p,
        obs_space,
        max_timestep=4096,
        out_channels: int = 32,
        kernel_size: int = 3,
        n_conv_layers: int = 4,
        pooling_kernel_size = 3,
        pooling_stride = 2,
        pooling_padding = 1
    ) -> None:
        super().__init__()
        
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim
        
        input_seq_len = 3 * context_len
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)
        
        # project to embedding
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)

        
        n, m, in_channels = obs_space['image']
        for _ in range(n_conv_layers):
            n = (n - (kernel_size - 1))
            m = (m - (kernel_size - 1))
            n = math.floor((n + 2 * pooling_padding - pooling_kernel_size) / pooling_stride) + 1 
            m = math.floor((m + 2 * pooling_padding - pooling_kernel_size) / pooling_stride) + 1 
            
        flatten_dim = out_channels * n * m
        
        n_filter_list = [in_channels] + \
                [out_channels for _ in range(n_conv_layers - 1)] + \
                [out_channels]

        self.embed_frame = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(
                        n_filter_list[i], n_filter_list[i + 1],
                        kernel_size=(kernel_size, kernel_size)
                        # stride=(stride, stride),
                        # padding=(padding, padding), bias=conv_bias
                ),
                # nn.Identity() if activation is None else activation(),
                nn.MaxPool2d(
                        kernel_size=pooling_kernel_size,
                        stride=pooling_stride,
                        padding=pooling_padding,
                        ) # if max_pool else nn.Identity()
            )
                for i in range(n_conv_layers)
            ],
            nn.Flatten(),
            nn.Linear(flatten_dim, h_dim),
        )
        
        self.embed_state = nn.Embedding(2, h_dim)
        self.embed_action = nn.Embedding(act_dim, h_dim)
        
        use_action_tanh = True
        
        # prediction
        self.predict_state = nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] +
                ([nn.Tanh()] if use_action_tanh else [])
            )
        )
        
    @property
    def device(self):
        return next(self.parameters()).device
            
    def forward(
        self, 
        frame,
        memory,
        timesteps,
        states,
        actions, 
        # returns_to_go,
    ):
        B, T, *_ = states.shape
        time_embeddings = self.embed_timestep(timesteps)
        
        frame_embeddings = self.embed_frame(frame)
        
        if self.embed_frame[0][0].weight.grad is not None:
            breakpoint()
        
        if memory is None:
            memory = frame_embeddings.unsqueeze(1)
        else:
            memory = torch.cat([memory, frame_embeddings.unsqueeze(1)], dim=1) + time_embeddings
            
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        
        h = torch.stack([
            memory,
            state_embeddings,
            action_embeddings,
        ], dim=1).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)
        
        h = self.embed_ln(h)
        
        h = self.transformer(h)
        
        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)
        # return_preds = self.predict_rtg(h[:, 2])
        state_logits = self.predict_state(h[:, 1])
        action_logits = self.predict_action(h[:, 2])
        
        state_preds = torch.argmax(state_logits, dim=-1).unsqueeze(-1).expand(action_logits.shape)
        action_logits = action_logits * state_preds +\
                        F.one_hot(torch.full([*action_logits.shape[:2]], self.act_dim - 1)).to(self.device) * (1 - state_preds)
        
        return (
            action_logits, 
            state_logits, 
            memory,
            # return_preds,
        )