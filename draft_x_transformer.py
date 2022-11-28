from x_transformers.x_transformers import (
    # TransformerWrapper, 
    # ContinuousTransformerWrapper,
    Decoder, 
    AttentionLayers, 
    TokenEmbedding, 
    AbsolutePositionalEmbedding,
    always,
    exists,
    default,
    repeat,
)

class ContinuousTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len,
        attn_layers,
        num_tokens = None,
        emb_dim = None,
        max_mem_len = 0.,
        shift_mem_down = 0,
        emb_dropout = 0.,
        post_emb_norm = False,
        num_memory_tokens = None,
        tie_embedding = False,
        use_abs_pos_emb = True,
        l2norm_embed = False,
        emb_frac_gradient = 1., # GLM-130B and Cogview successfully used this, set at 0.1
        is_tokenized = False,
    ):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down

        self.l2norm_embed = l2norm_embed 
        
        self.is_tokenized = is_tokenized
        
        if not is_tokenized:
            self.token_emb = TokenEmbedding(emb_dim, num_tokens, l2norm_embed = l2norm_embed)
        else:
            self.token_emb = nn.Identity()
            
        self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len, l2norm_embed = l2norm_embed) if (use_abs_pos_emb and not attn_layers.has_pos_emb) else always(0)

        self.emb_frac_gradient = emb_frac_gradient # fraction of the gradient that should go to the embedding, https://arxiv.org/abs/2105.13290

        self.post_emb_norm = nn.LayerNorm(dim) if post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        self.init_()

        self.to_logits = nn.Linear(dim, num_tokens) if not tie_embedding else lambda t: t @ self.token_emb.weight.t()

        # memory tokens (like [cls]) from Memory Transformers paper
        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

    def init_(self):
        if self.l2norm_embed:
            if not self.is_tokenized:
                nn.init.normal_(self.token_emb.emb.weight, std = 1e-5)
            if not isinstance(self.pos_emb, always):
                nn.init.normal_(self.pos_emb.emb.weight, std = 1e-5)
            return
        if not self.is_tokenized:
            nn.init.kaiming_normal_(self.token_emb.emb.weight)

    def forward(
        self,
        x,
        return_embeddings = False,
        mask = None,
        return_mems = False,
        return_attn = False,
        mems = None,
        pos = None,
        prepend_embeds = None,
        **kwargs
    ):
        b, n, device, num_mem, emb_frac_gradient = *x.shape, x.device, self.num_memory_tokens, self.emb_frac_gradient
        return_hiddens = return_mems | return_attn

        # absolute positional embedding

        external_pos_emb = exists(pos) and pos.dtype != torch.long
        pos_emb = self.pos_emb(x, pos = pos) if not external_pos_emb else pos
        x = self.token_emb(x) + pos_emb

        # post embedding norm, purportedly leads to greater stabilization

        x = self.post_emb_norm(x)

        # whether to append embeds, as in PaLI, for image embeddings

        if exists(prepend_embeds):
            prepend_seq, prepend_dim = prepend_embeds.shape[1:]
            assert prepend_dim == x.shape[-1], 'prepended embeddings need to have same dimensions as text model dimensions'

            x = torch.cat((prepend_embeds, x), dim = -2)

        # whether to reduce the gradient going to the embedding, from cogview paper, corroborated by GLM-130B model

        if emb_frac_gradient < 1:
            assert emb_frac_gradient > 0
            x = x * emb_frac_gradient + x.detach() * (1 - emb_frac_gradient)

        # embedding dropout

        x = self.emb_dropout(x)

        x = self.project_emb(x)

        if num_mem > 0:
            mem = repeat(self.memory_tokens, 'n d -> b n d', b = b)
            x = torch.cat((mem, x), dim = 1)

            # auto-handle masking after appending memory tokens
            if exists(mask):
                mask = F.pad(mask, (num_mem, 0), value = True)

        if self.shift_mem_down and exists(mems):
            mems_l, mems_r = mems[:self.shift_mem_down], mems[self.shift_mem_down:]
            mems = [*mems_r, *mems_l]

        if return_hiddens:
            x, intermediates = self.attn_layers(x, mask = mask, mems = mems, return_hiddens = True, **kwargs)
        else:
            x = self.attn_layers(x, mask = mask, mems = mems, **kwargs)

        x = self.norm(x)

        mem, x = x[:, :num_mem], x[:, num_mem:]

        out = self.to_logits(x) if not return_embeddings else x

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = list(map(lambda pair: torch.cat(pair, dim = -2), zip(mems, hiddens))) if exists(mems) else hiddens
            new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), new_mems))
            return out, new_mems

        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        return out
