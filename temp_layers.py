''' Define the temproal attention layer '''

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
#temporal attention layer     1

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))  #q(k.T, h, w)   k.T(k.T, w, h) both 3 dims
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class TemporalAttention(nn.Module):
    ''' Temporal Attention module '''
    def __init__(self, n_head, rnn_unit, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(rnn_unit, n_head * d_k)
        self.w_ks = nn.Linear(rnn_unit, n_head * d_k)
        self.w_vs = nn.Linear(rnn_unit, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (rnn_unit + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (rnn_unit + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (rnn_unit + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(rnn_unit)
        self.fc = nn.Linear(n_head * d_v, rnn_unit)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):  #q, k, v seen as rnn_output respectively  d_k:8  d_v:8 n_head:4 n_hid = 8
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head    #q, k, v (24256, 10, 32)
        sz_b, len_q, _ = q.size()   #24256, 10, 32
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)    #(24256, 10, 4, 8)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) #permute(4,24256,10,8) view(8,10, 8)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        output, attn = self.attention(q, k, v, mask=mask)   #mask 97024, 10, 10
        output = output.view(n_head, sz_b, len_q, d_v)   #out(4, 24256, 10, 8)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)#permute(24256,10,4,8)view(24256, 10,32
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        output = output[:, -1, :]
        return output, attn
