import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import (
    TransformerEncoderLayer,
    TransformerEncoder, 
    TransformerDecoderLayer,
    TransformerDecoder,
)
from torch.utils.data import dataset
from torch.distributions.categorical import Categorical

# class PositionalEncoding(nn.Module):
#     def __init__(
#         self,
#         d_hid,
#         n_position=200
#     ) -> None:
#         super().__init__()

#         self.register_buffer('pe', self._get_sinusoid_encoding_table(n_position, d_hid))

#     def _get_sinusoid_encoding_table(self, n_position, d_hid):

#         def get_position_angle_vec(position):
#             return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

#         sinusoid_table = torch.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
#         sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
#         sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])

#         return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    
#     def forward(self, x):
#         return x + self.pos_table[:, :x.size(1)].clone().detach()

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class CausalVisionTransformer(nn.Module):

    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        kernel_size:int,
        # naction: int,
        obs_space,
        action_space,
        d_model: int, # 10
        nhead: int,
        d_hid: int,
        nlayers: int, # 2
        max_len: int, # 5
        dropout: float = 0.5,
        n_conv_layers: int = 4,
        pooling_kernel_size = 3,
        pooling_stride = 2,
        pooling_padding = 1
        
    ):
        super().__init__()
        # assert d_model % max_len == 0
        # d_model = d_model // max_len
        self.naction = action_space.n
        self.d_model = d_model

        self.model_type = 'Transformer'
        
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        for _ in range(n_conv_layers):
            n = (n - (kernel_size - 1))
            m = (m - (kernel_size - 1))
            n = math.floor((n + 2 * pooling_padding - pooling_kernel_size) / pooling_stride) + 1 
            m = math.floor((m + 2 * pooling_padding - pooling_kernel_size) / pooling_stride) + 1 
            
        flatten_dim = out_channels * n * m
        
        n_filter_list = [in_channels] + \
                        [out_channels for _ in range(n_conv_layers - 1)] + \
                        [out_channels]

        self.conv_encoder = nn.Sequential(
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
            nn.Linear(flatten_dim, d_model),
        )

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        self.tgt_emb = nn.Embedding(self.naction, self.d_model)
        
        # transf_encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        # self.transf_encoder = TransformerEncoder(transf_encoder_layers, nlayers)
        
        transf_decoder_layers = TransformerDecoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transf_decoder = TransformerDecoder(
            transf_decoder_layers,
            num_layers=nlayers,
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_model, self.naction),
            # nn.Softmax(dim=1)
        )

        self.recurrent = True
        self.max_len = max_len
        self.memory = None
        self.transf_memory = None

        self.init_weights()

    def init_weights(self) -> None:
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.conv_encoder.apply(weights_init)
        # self.transf_encoder.apply(weights_init)
        self.transf_decoder.apply(weights_init)
        self.decoder.apply(weights_init)

    def reset_memory_(self):
        del self.memory
        self.memory = None
        
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        img,
        goal,
        step,
        attn_mask: Tensor = None,
        return_embed: bool = False,
        return_dist: bool = True
    ) -> Tensor:
        """
        Args:
            memory: Tensor, shape [seq_len, batch_size, d_model]

            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, naction]
        """
        # assert not (straight_through and return_embed)
        
        x = img
        x = self.conv_encoder(x) 
        # * math.sqrt(self.d_model)
        
        if self.memory is None:
            self.memory = torch.cat(
                [
                    x.unsqueeze(1),
                    torch.zeros((self.max_len - step - 1, *x.shape), device=self.device).transpose(0, 1)
                ],
                dim=1
            )
        else:
            self.memory = self.memory + torch.cat(
                [
                    torch.zeros((step, *x.shape), device=self.device).transpose(0, 1),
                    x.unsqueeze(1), 
                    torch.zeros((self.max_len - step - 1, *x.shape), device=self.device).transpose(0, 1)
                ],
                dim=1
            )
        
        embed = self.pos_encoder(self.memory)
        
        if attn_mask is not None:
            # embed = self.transf_encoder(embed, attn_mask)
            embed = self.transf_decoder(self.tgt_emb(goal).squeeze(), embed, tgt_mask=attn_mask, memory_mask=attn_mask)
        else:
            # embed = self.transf_encoder(embed)
            embed = self.transf_decoder(self.tgt_emb(goal).squeeze(), embed)
        
        output = self.decoder(embed)

        if return_dist:
            output = Categorical(logits=output)

        # self.memory = self.memory.reshape(-1, self.max_len, -1)
        mem_mask = torch.zeros_like(self.memory, requires_grad=False)
        mem_mask[:, :step+1] = 1
        
        embed_mask = torch.zeros_like(embed, requires_grad=False)
        if step + 1 < self.max_len:
            embed_mask[:, step+1] = 1
        
        embed_cummean = (torch.cumsum(embed, dim=1)/torch.arange(1, self.max_len + 1, device=self.device).expand(embed.shape[0], embed.shape[2], embed.shape[1]).transpose(-1,-2))
        self.memory = self.memory * mem_mask + embed_cummean * embed_mask
        
        if return_embed:
            return output, embed
        
        return output, 