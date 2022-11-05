import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


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
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
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
    ):
        super().__init__()
        # breakpoint()
        # assert d_model % max_len == 0
        # self.mem_size = d_model // max_len
        self.memory_size = max_len
        self.mem_size = d_model
        self.naction = action_space.n
        self.d_model = d_model

        self.model_type = 'Transformer'

        n = obs_space["image"][0]
        m = obs_space["image"][1]
        flatten_dim = out_channels * (n - (kernel_size - 1))**2 * (m - (kernel_size - 1))**2

        # self.conv_encoder = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(flatten_dim, self.mem_size),
        #     nn.Tanh()
        # )
        
        self.conv_encoder = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.lin = nn.Linear(flatten_dim, self.mem_size)
        self.flatten_dim = flatten_dim
        self.tanh = nn.Tanh()

        # self.encoder = nn.Embedding(ntoken, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        transf_encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transf_encoder = TransformerEncoder(transf_encoder_layers, nlayers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, self.naction),
            nn.Softmax()
        )

 
        # self.conv_memory = torch.empty((0, d_model))
        self.recurrent = True

        self.init_weights()

    def init_weights(self) -> None:
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

        initrange = 0.1

        # self.conv_encoder.bias.data.zero_()
        # self.conv_encoder.weight.data.uniform_(-initrange, initrange)
        self.conv_encoder.apply(weights_init)

        # self.transf_encoder.weight.data.uniform_(-initrange, initrange)
        self.transf_encoder.apply(weights_init)

        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

        self.decoder.apply(weights_init)

    def forward(self, obs, memory, src_mask: Tensor = None, straight_through: bool = False, return_embed: bool = False) -> Tensor:
        """
        Args:
            memory: Tensor, shape [seq_len, batch_size, d_model]

            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, naction]
        """

        # assert not (straight_through and return_embed)
        breakpoint()
        x = obs.image.permute(0, 3, 1, 2) # batch, channel, width, height
        # x = self.conv_encoder(x) * math.sqrt(self.d_model) # batch, d_model

        x = self.conv_encoder(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.lin(x)
        x = self.tanh(x)
        x = x * math.sqrt(self.d_model)

        # self.conv_memory = torch.cat(self.conv_memory, embed)
        memory[:, self.mem_size * obs.step : self.mem_size * (obs.step + 1)] += x # batch, seq_len * d_model
        x = memory[:-self.max_len].view(memory.shape[0], self.max_len, -1).transpose(0, 1) # seq_len, batch, d_model
        # memory = torch.cat(memory, embed)

        # if not straight_through:

            # src = self.encoder(src) * math.sqrt(self.d_model)
            
        embed = self.pos_encoder(x)
        if src_mask:
            embed = self.transf_encoder(embed, src_mask)
        else:
            embed = self.transf_encoder(embed)
        output = self.decoder(embed)

        output = output.transpose(0, 1)

        def_action = F.one_hot(torch.ones_like(output[:, 0]), num_classes=self.naction)

        pick_mask = obs.asked
        drop_mask = 1 - pick_mask

        output = output  - (output * drop_mask).detach() + def_action * pick_mask
        
        # self.conv_memory = torch.empty((0, self.d_model))

        # probs = F.softmax(output, dim=1)
        # t = []
        # for prob, asked in zip(probs, obs.asked):
        #     if asked == True:
        #         t.append(prob)
        #     else:
        #         t.append(F.one_hot(torch.tensor([probs.shape[1] - 1])).squeeze())
        
        # probs = torch.stack(t).float().to(probs.device)
        
        if return_embed:
            return output, memory, embed
        
        return output, memory,

        # return output,


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)