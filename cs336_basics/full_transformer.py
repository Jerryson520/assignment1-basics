from __future__ import annotations

from ast import Break, Module
import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, einsum
import torch.nn as nn
from collections import defaultdict
import regex as re
from einops import rearrange, einsum
import math

def _trunc_init(out_features: int, in_features: int, device=None, dtype=None) -> torch.Tensor:
    W = torch.empty(out_features, in_features, device=device, dtype=dtype)
    std = math.sqrt(2 / (in_features + out_features))
    nn.init.trunc_normal_(W, std=std, a=-3 * std, b=3 * std)
    return W


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    x = in_features - torch.max(in_features, dim=dim, keepdim=True).values
    new_x = torch.exp(x)
    denominator = torch.sum(new_x, dim=dim, keepdim=True)
    return new_x / denominator


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        W = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = math.sqrt(2/(in_features + out_features))
        nn.init.trunc_normal_(W, std=std, a=-3*std, b=3*std)

        self.weights = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights.T

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        sum_ = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x / torch.sqrt(sum_ + self.eps) 
        return (self.weights * x_norm).to(in_dtype)

class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()
        embeddings = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(embeddings, std=1, a=-3, b=3)
        self.embeddings = nn.Parameter(embeddings)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()
        self.weights1 = nn.Parameter(_trunc_init(d_ff, d_model, device, dtype))
        self.weights2 = nn.Parameter(_trunc_init(d_model, d_ff, device, dtype))
        self.weights3 = nn.Parameter(_trunc_init(d_ff, d_model, device, dtype))

    def forward(self, x: torch.Tensor):
        new_x = x @  self.weights1.T
        silu = new_x * torch.sigmoid(new_x)
        return ((silu * (x @ self.weights3.T)) @ self.weights2.T)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0
        j = torch.arange(0, d_k, 2, device=device)
        factor = 1 / theta ** (j/d_k)
        pos = torch.arange(0, max_seq_len, device=device)
        angles = einsum(pos, factor, "i, j -> i j")
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        sin_cached = self.sin_cached[token_positions]
        cos_cached = self.cos_cached[token_positions]

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        out = torch.empty_like(x)
        out[..., ::2] = x_even * cos_cached - x_odd * sin_cached
        out[..., 1::2] = x_even * sin_cached + x_odd * cos_cached
        return out

class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, num_heads, rope:RotaryPositionalEmbedding=None):
        super().__init__()
        assert d_model % num_heads == 0
        d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_k = d_k
        self.rope = rope
        
        self.weights_q = nn.Parameter(_trunc_init(num_heads*d_k, d_model))
        self.weights_k = nn.Parameter(_trunc_init(num_heads*d_k, d_model))
        self.weights_v = nn.Parameter(_trunc_init(num_heads*d_k, d_model))
        self.weights_o = nn.Parameter(_trunc_init(d_model, num_heads*d_k))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        Q = torch.einsum("...ld, md -> ...lm", x, self.weights_q)
        K = torch.einsum("...ld, md -> ...lm", x, self.weights_k)
        V = torch.einsum("...ld, md -> ...lm", x, self.weights_v)
        
        Q = rearrange(Q, "... l (h d) -> ... h l d", h=self.num_heads)
        K = rearrange(K, "... l (h d) -> ... h l d", h=self.num_heads)
        V = rearrange(V, "... l (h d) -> ... h l d", h=self.num_heads)
        if self.rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        
        scores = torch.einsum("...hid,...hjd -> ...hij", Q, K) / math.sqrt(self.d_k)

        seq_len = x.shape[-2]
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

        scores = run_softmax(scores, dim=-1)
        final_scores = torch.einsum("...hij, ...hjd -> ...hid", scores, V)
        final_scores = rearrange(final_scores, "... h l d -> ... l (h d)", h=self.num_heads)
        return torch.einsum("...le, de -> ...ld", final_scores, self.weights_o)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_k = d_k
        
        self.weights_q = nn.Parameter(_trunc_init(num_heads*d_k, d_model))
        self.weights_k = nn.Parameter(_trunc_init(num_heads*d_k, d_model))
        self.weights_v = nn.Parameter(_trunc_init(num_heads*d_k, d_model))
        self.weights_o = nn.Parameter(_trunc_init(d_model, num_heads*d_k))

    def forward(self, x: torch.Tensor):
        Q = torch.einsum("...ld, md -> ...lm", x, self.weights_q)
        K = torch.einsum("...ld, md -> ...lm", x, self.weights_k)
        V = torch.einsum("...ld, md -> ...lm", x, self.weights_v)
        
        Q = rearrange(Q, "... l (h d) -> ... h l d", h=self.num_heads)
        K = rearrange(K, "... l (h d) -> ... h l d", h=self.num_heads)
        V = rearrange(V, "... l (h d) -> ... h l d", h=self.num_heads)
        
        scores = torch.einsum("...hid,...hjd -> ...hij", Q, K) / math.sqrt(self.d_k)

        seq_len = x.shape[-2]
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

        scores = run_softmax(scores, dim=-1)
        final_scores = torch.einsum("...hij, ...hjd -> ...hid", scores, V)
        final_scores = rearrange(final_scores, "... h l d -> ... l (h d)", h=self.num_heads)
        return torch.einsum("...le, de -> ...ld", final_scores, self.weights_o)


class TransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model:int, 
        num_heads:int, 
        d_ff:int, 
        theta:int, 
        max_seq_len:int, 
        use_rmsnorm: bool = True,
        use_prenorm: bool = True,
        use_rope: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        d_k = d_model // num_heads
        self.ln1 = RMSNorm(d_model) if use_rmsnorm else nn.Identity()
        self.ln2 = RMSNorm(d_model) if use_rmsnorm else nn.Identity()
        self.rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len) if use_rope else None
        self.attn = MultiHeadAttentionWithRoPE(d_model, num_heads, self.rope)
        self.ffn = SwiGLU(d_model, d_ff)
        self.use_prenorm = use_prenorm
    
    def forward(self, x: torch.Tensor):
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len, device=x.device)
        y1 = x + self.attn(self.ln1(x), token_positions) if self.use_prenorm else self.ln1(x + self.attn(x, token_positions))
        y = y1 + self.ffn(self.ln2(y1)) if self.use_prenorm else self.ln2(y1 + self.ffn(y1))
        return y


class TransformerLM(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        context_length: int, 
        d_model:int, 
        num_layers: int, 
        num_heads:int, 
        d_ff:int, theta:int,
        use_rmsnorm: bool = True,
        use_prenorm: bool = True,
        use_rope: bool = True,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, theta, context_length, 
                use_rmsnorm=use_rmsnorm, use_prenorm=use_prenorm, use_rope=use_rope) 
            for i in range(num_layers)
        ])
        self.rmsnorm = RMSNorm(d_model) if use_rmsnorm else nn.Identity()
        self.Linear = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.rmsnorm(x)
        out = self.Linear(x)
        return out