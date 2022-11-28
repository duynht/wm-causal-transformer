import torch
from torch import (
    nn,
    Tensor,
)

from torch.distributions.categorical import Categorical

from x_transformers import (
    Decoder,
    ContinuousTransformerWrapper
)

import math

class VisualCausalTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        obs_space,
        action_space, 
        d_model: int = 10,
        nhead: int = 1,
        d_hid: int = 10,
        num_layers: int = 2,
        max_len: int = 8,
        dropout: float = 0.5,
        n_conv_layer: int = 1,
        pooling_kernel_size: int = 3,
        pooling_stride: int = 2,
        pooling_padding: int = 1,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.naction = action_space.n

        n = obs_space['image'][0]
        m = obs_space['image'][1]
        for _ in range(n_conv_layer):
            n = (n - (kernel_size - 1))
            m = (m - (kernel_size - 1))
            n = math.floor((n + 2 * pooling_padding - pooling_kernel_size) / pooling_stride) + 1
            m = math.floor((m + 2 * pooling_padding - pooling_kernel_size) / pooling_stride) + 1

        flatten_dim = out_channels * n * m

        n_filter_list = [in_channels] + \
                        [out_channels for _ in range(n_conv_layer - 1)] + \
                        [out_channels]

        self.conv_encoder = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        n_filter_list[i], n_filter_list[i+1],
                        kernel_size=(kernel_size, kernel_size),
                    ),
                    nn.MaxPool2d(
                        kernel_size=pooling_kernel_size,
                        stride=pooling_stride,
                        padding=pooling_padding,
                    )
                )
            ],
            nn.Flatten(),
            nn.Linear(flatten_dim, self.d_model),
        )
        
        self.transf_decoder = ContinuousTransformerWrapper(
            num_tokens=self.naction,
            max_seq_len=max_len,
            attn_layers=Decoder(
                dim=d_model,
                depth=num_layers,
                heads=nhead,
                rel_pos_bias=True
            )
        )
        
        self.memory = None
        
    def reset_memory_(self):
        del self.memory
        self.memory = None
        
    @property
    def device(self):
        return next(self.parameters()).device
    
    def init_(self) -> None:
        def weight_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight)
            
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                

        self.conv_encoder.apply(weight_init)
        
    
    def forward(
        self,
        x,
        attn_mask: Tensor = None,
        return_embed: bool = False,
        return_dist: bool = False,
    ) -> Tensor:
        
        x = self.conv_encoder(x)
        
        if self.memory is None:
            self.memory = x.unsqueeze(1)
        else:
            self.memory = torch.cat(self.memory, x, dim=1)
            
        x = self.transf_decoder(self.memory)
        
        x = x.permute(0, 2, 1)
        
        if return_dist:
            x = Categorical(logist=x)

        return x