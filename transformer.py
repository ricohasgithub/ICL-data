import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import attention_map_vis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequenceEmbedder(nn.Module):
    """
    P = one-hot positional embedding dimension
    D = embedding dimension for items/labels
    N = sequence length
    """

    def __init__(self, P, D, N):
        self.P = P
        self.D = D
        self.N = N

    # examples and labels each has shape (batch_size, sequence_length, D)
    def forward(self, examples, labels, is_training=True):

        batch_size, N, D = examples.size()
        pos_encoding = torch.zeros(batch_size, 2 * N + 1, self.P)

        start_index = random.randint(0, self.P - (2 * N + 1))
        one_hot_indices = torch.arange(start_index, start_index + (2 * N + 1))
        pos_encoding.scatter_(2, one_hot_indices.unsqueeze(2), 1)

        interleaved = torch.empty((batch_size, 2 * N + 1, D), dtype=examples.dtype)
        interleaved[:, 0::2] = examples
        interleaved[:, 1::2] = labels[:, :-1]
        return torch.cat((pos_encoding, interleaved), dim=2)


class LayerNorm(nn.Module):
    "Code from: https://nlp.seas.harvard.edu/annotated-transformer/"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features)).to(device)
        self.b_2 = nn.Parameter(torch.zeros(features)).to(device)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Attention(nn.Module):

    "Implements Standard MultiHeadAttention"

    def __init__(self, n_heads=8, d_hidden=64, p_dropout=0.0, scaling=1.0, bias=True):

        super(Attention, self).__init__()

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
            # else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.n_heads = n_heads
        self.d_hidden = d_hidden
        self.p_dropout = p_dropout
        self.scaling = scaling
        self.bias = bias

        self.W_Q = nn.Linear(d_hidden, d_hidden).to(self.device)
        self.W_K = nn.Linear(d_hidden, d_hidden).to(self.device)
        self.W_V = nn.Linear(d_hidden, d_hidden).to(self.device)
        self.W_O = nn.Linear(d_hidden, d_hidden).to(self.device)

    def forward(
        self, x, y=None, mask=None, layer=-1, vis_mode=-1, epoch=-1, vis_path=""
    ):

        batch_size, seq_len = x.shape[0], x.shape[1]

        if y is None:
            y = x

        Q = (
            self.W_Q(x)
            .view(batch_size, -1, self.n_heads, self.d_hidden // self.n_heads)
            .transpose(1, 2)
        )
        K = (
            self.W_K(x)
            .view(batch_size, -1, self.n_heads, self.d_hidden // self.n_heads)
            .transpose(1, 2)
        )
        V = (
            self.W_V(x)
            .view(batch_size, -1, self.n_heads, self.d_hidden // self.n_heads)
            .transpose(1, 2)
        )

        x, att_dist = self.attention(Q, K, V, mask)

        # Save attention map
        if layer > -1:
            attention_map_vis(
                att_dist, vis_path, layer=layer, vis_mode=vis_mode, epoch=epoch
            )

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_hidden)
        return self.W_O(x)

    def attention(self, Q, K, V, mask=None):
        d_K = Q.size(-1)
        att_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_K)

        if mask is not None:
            att_scores = att_scores.masked_fill(mask == 0, -1e9)

        att_dist = att_scores.softmax(dim=-1)

        return torch.matmul(att_dist, V), att_dist


class CausalAttention(Attention):

    def __init__(self, n_heads=8, d_hidden=64, p_dropout=0.0, scaling=1.0, bias=True):
        super(CausalAttention, self).__init__(
            n_heads, d_hidden, p_dropout, scaling, bias
        )

    def forward(
        self, x, y=None, mask=None, layer=-1, vis_mode=-1, epoch=-1, vis_path=""
    ):
        batch_size, seq_len = x.shape[0], x.shape[1]
        t = torch.arange(seq_len).to(self.device)
        causal_mask = (t[:, None] >= t[None, :])[None, None, :, :]
        if mask is None:
            mask = torch.broadcast_to(causal_mask, (batch_size, 1, seq_len, seq_len))
        else:
            mask = mask * causal_mask
        return super(CausalAttention, self).forward(
            x=x,
            y=y,
            mask=mask,
            layer=layer,
            vis_mode=vis_mode,
            epoch=epoch,
            vis_path=vis_path,
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_heads=1,
        d_hidden=128,
        p_dropout=0.0,
        scaling=1.0,
        bias=True,
    ):
        super(TransformerBlock, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_heads = n_heads
        self.d_hidden = d_hidden
        self.p_dropout = p_dropout
        self.scaling = scaling
        self.bias = bias

        self.layer_norm = LayerNorm(self.d_hidden).to(self.device)
        self.causal_block = CausalAttention(
            self.n_heads, self.d_hidden, self.p_dropout, self.scaling, self.bias
        ).to(self.device)

    def forward(
        self, x, y=None, mask=None, layer=-1, vis_mode=-1, epoch=-1, vis_path=""
    ):
        x = x + self.causal_block(
            self.layer_norm(x),
            y,
            mask,
            layer=layer,
            vis_mode=vis_mode,
            epoch=epoch,
            vis_path=vis_path,
        )
        return x


class MLP(nn.Module):

    def __init__(self, n_classes, d_hidden=128):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(d_hidden, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_hidden)
        self.lin3 = nn.Linear(d_hidden, n_classes)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        query = x[:, -1, :]
        return self.lin3(query)


class Readout(nn.Module):

    def __init__(self, n_classes):
        super(Readout, self).__init__()
        self.n_classes = n_classes

    def forward(self, x):
        query = x[:, -1, :]
        return query[:, : self.n_classes]


class Transformer(nn.Module):

    def __init__(
        self, n_classes, n_layers=2, n_heads=1, p_dropout=0.0, d_hidden=128, mlp=None
    ):
        super(Transformer, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.d_hidden = d_hidden

        self.layer_norm = LayerNorm(self.d_hidden)
        for i in range(self.n_layers):
            setattr(
                self,
                f"transformer_block_{i}",
                TransformerBlock(
                    n_heads=self.n_heads,
                    d_hidden=self.d_hidden,
                    p_dropout=self.p_dropout,
                ).to(self.device),
            )

        if mlp is not None:
            print("NOT USING MLP")
            self.mlp = mlp
        else:
            print("USING BASE 3 LAYER MLP")
            self.mlp = MLP(n_classes, d_hidden)

    def forward(self, x, epoch=-1, vis_mode=-1, vis_path=""):

        for i in range(self.n_layers):
            if epoch % 100 == 0:
                x = getattr(self, f"transformer_block_{i}")(
                    x, layer=i, vis_mode=vis_mode, epoch=epoch, vis_path=vis_path
                )
            else:
                x = getattr(self, f"transformer_block_{i}")(x, layer=-1)

        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
