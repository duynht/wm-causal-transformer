import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temparature, attn_dropout=0.1) -> None:
        super().__init__()
        self.temperature = temparature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1) -> None:
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.attention = ScaledDotProductAttention(temparature=d_k**0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b = q.size(0)
        len_q, len_k, len_v = q.size(1), k.size(1), v.size(1)

        residual = q

        # Linear projection
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        # Concatenate all heads
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        
        q = self.layer_norm(q)

        return q, attn

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1) -> None:
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in ,eps=1e-6)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1) -> None:
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedforward(d_model, d_inner, dropout=dropout)

    def forward(self, input, slf_attn_mask=None):
        output, slf_attn = self.slf_attn(input, input, input, mask=slf_attn_mask)
        output = self.pos_ffn(output)

        return output, slf_attn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1) -> None:
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedforward(d_model, d_inner, dropout=dropout)

    def forward(self, input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        output, dec_slf_attn = self.slf_attn(input, input, input, mask=slf_attn_mask)
        output, dec_enc_attn = self.enc_attn(output, output, output, mask=dec_enc_attn_mask)
        output = self.pos_ffn(output)
        
        return output, dec_slf_attn, dec_enc_attn

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_hid,
        n_position=200
    ) -> None:
        super().__init__()

        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    
    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_src_vocab,
        d_word_vec,
        n_layers,
        n_head,
        d_k,
        d_v,
        d_model,
        d_inner,
        pad_idx,
        dropout=0.1,
        n_position=200,
        scale_emb=False,
    ) -> None:
        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):
        slf_attn_list = []

        output = self.src_word_emb(src_seq)

        if self.scale_emb:
            output *= self.d_model ** 0.5

        output = self.dropout(self.position_enc(output))
        output = self.layer_norm(output)

        for enc_layer in self.layer_stack:
            output, slf_attn = enc_layer(output, slf_attn_mask=src_mask)
            slf_attn_list += [slf_attn] if return_attns else []

        if return_attns:
            return output, slf_attn_list

        return output, 

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        n_trg_vocab,
        d_word_vec,
        n_layers,
        n_head,
        d_k,
        d_v,
        d_model,
        d_inner,
        pad_idx,
        n_position=200,
        dropout=0.1,
        scale_emb=False
    ) -> None:
        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        slf_attn_list, dec_enc_attn_list = [], []

        output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            output *= self.d_model ** 0.5
        output = self.dropout(self.position_enc(output))
        output = self.layer_norm(output)

        for dec_layer in self.layer_stack:
            output, slf_attn, dec_enc_attn = dec_layer(output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            slf_attn_list += [slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return output, slf_attn_list, dec_enc_attn_list
        
        return output,

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    _, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1
    )).bool()

    return subsequent_mask

class Transformer(nn.Module):
    def __init__(
        self,
        n_src_vocab,
        n_trg_vocab,
        src_pad_idx,
        trg_pad_idx,
        d_word_vec=512,
        d_model=512,
        d_inner=2048,
        n_layers=6, 
        n_head=8,
        d_k=64,
        d_v=64,
        dropout=0.1,
        n_position=200,
        trg_emb_prj_weight_sharing=True,
        emb_src_trg_weight_sharing=True,
        scale_emb_or_prj='prj'
    ) -> None:
        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

        self.encoder = TransformerEncoder(
            n_src_vocab=n_src_vocab,
            n_position=n_position,
            d_word_vec=d_word_vec,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            pad_idx=src_pad_idx,
            dropout=dropout,
            scale_emb=scale_emb
        )

        self.decoder = TransformerDecoder(
            n_trg_vocab=n_trg_vocab,
            n_position=n_position,
            d_word_vec=d_word_vec,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            pad_idx=trg_pad_idx,
            dropout=dropout,
            scale_emb=scale_emb
        )

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if trg_emb_prj_weight_sharing:
            # Same weight for target word embedding and last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        if emb_src_trg_weight_sharing:
            # Same weight for source word embedding and the target word embedding
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight

    def forward(self, src_seq, trg_seq):
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output)

        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit.view(-1, seq_logit.size(2))

class TransformerVisualEncoder(nn.Module):
    def __init__(
        self,
        n_src_vocab,
        d_word_vec,
        n_layers,
        n_head,
        d_k,
        d_v,
        d_model,
        d_inner,
        pad_idx,
        dropout=0.1,
        n_position=200,
        scale_emb=False,
    ) -> None:
        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):
        slf_attn_list = []

        output = self.conv2d(src_seq)
        # output = self.src_word_emb(src_seq)

        if self.scale_emb:
            output *= self.d_model ** 0.5

        output = self.dropout(self.position_enc(output))
        output = self.layer_norm(output)

        for enc_layer in self.layer_stack:
            output, slf_attn = enc_layer(output, slf_attn_mask=src_mask)
            slf_attn_list += [slf_attn] if return_attns else []

        if return_attns:
            return output, slf_attn_list

        return output, 


class TransformerVisualDecoder(nn.Module):
    def __init__(
        self,
        n_trg_vocab,
        d_word_vec,
        n_layers,
        n_head,
        d_k,
        d_v,
        d_model,
        d_inner,
        pad_idx,
        n_position=200,
        dropout=0.1,
        scale_emb=False
    ) -> None:
        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        slf_attn_list, dec_enc_attn_list = [], []

        # output = self.trg_word_emb(trg_seq)
        

        if self.scale_emb:
            output *= self.d_model ** 0.5
        output = self.dropout(self.position_enc(output))
        output = self.layer_norm(output)

        for dec_layer in self.layer_stack:
            output, slf_attn, dec_enc_attn = dec_layer(output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            slf_attn_list += [slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return output, slf_attn_list, dec_enc_attn_list
        
        return output,



class CausalTransformer(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_dim,
        action_range,
        ordering=0,
        max_len=None,
        eval_context_len=None,
        max_ep_len=4096,
        action_tanh=True,
        stochastic_policy=False,
        init_temperature=0.1,
        target_entropy=None,
        **kwargs
    ) -> None:
        super().__init__()

        self.conv_encoder = nn.Conv2d(input_dim, hidden_dim, kernel_size=3)


        self.hidden_dim = hidden_dim
        
        self.transformer = Transformer(
            n_src_vocab=1,
            d_word_vec=hidden_dim,
            **kwargs
        )

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_dim)

        if ordering:
            self.embed_ordering = nn.Embedding(max_ep_len, hidden_dim)
        
        self.embed_return = nn.Linear(1, hidden_dim)
        self.embed_state = nn.Linear(self.state_dim, hidden_dim)
        self.embed_action = nn.Linear(self.act_dim, hidden_dim)

