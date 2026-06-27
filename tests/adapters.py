from __future__ import annotations

from ast import Break, Module
from email.policy import default
import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
from sympy.utilities.iterables import rotate_right
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, einsum
import torch.nn as nn
from collections import defaultdict
import regex as re
from cs336_basics.bpe_tokenizer import BPETokenizer
from einops import rearrange, einsum
import math
from collections.abc import Callable, Iterable
from typing import Optional
import numpy as np



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
        self.weights = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
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

class swiglu(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()
        self.weights1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.weights2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.weights3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor):
        new_x = x @  self.weights1.T
        silu = run_silu(new_x)
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
        
        self.weights_q = nn.Parameter(torch.empty(num_heads*d_k, d_model))
        self.weights_k = nn.Parameter(torch.empty(num_heads*d_k, d_model))
        self.weights_v = nn.Parameter(torch.empty(num_heads*d_k, d_model))
        self.weights_o = nn.Parameter(torch.empty(d_model, num_heads*d_k))

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

        scores = torch.softmax(scores, dim=-1)
        final_scores = torch.einsum("...hij, ...hjd -> ...hid", scores, V)
        final_scores = rearrange(final_scores, "... h l d -> ... l (h d)", h=self.num_heads)
        return torch.einsum("...le, de -> ...ld", final_scores, self.weights_o)

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    layer = Linear(d_in, d_out)
    layer.load_state_dict({"weights": weights})
    return layer(in_features)
       

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    embedding_layer = Embedding(vocab_size, d_model)
    embedding_layer.load_state_dict({"embeddings": weights})
    return embedding_layer(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    layer = swiglu(d_model, d_ff)
    layer.load_state_dict({"weights1": w1_weight, "weights2": w2_weight, "weights3": w3_weight})
    return layer(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... q d, ... k d -> ... q k") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = run_softmax(scores, dim=-1)
    return einsum(weights, V, "... q k, ... k v -> ... q v")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_k = d_k
        
        self.weights_q = nn.Parameter(torch.empty(num_heads*d_k, d_model))
        self.weights_k = nn.Parameter(torch.empty(num_heads*d_k, d_model))
        self.weights_v = nn.Parameter(torch.empty(num_heads*d_k, d_model))
        self.weights_o = nn.Parameter(torch.empty(d_model, num_heads*d_k))

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

        scores = torch.softmax(scores, dim=-1)
        final_scores = torch.einsum("...hij, ...hjd -> ...hid", scores, V)
        final_scores = rearrange(final_scores, "... h l d -> ... l (h d)", h=self.num_heads)
        return torch.einsum("...le, de -> ...ld", final_scores, self.weights_o)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    layer = MultiHeadAttention(d_model, num_heads)
    layer.load_state_dict({"weights_q": q_proj_weight, "weights_k": k_proj_weight, "weights_v": v_proj_weight, "weights_o": o_proj_weight})
    return layer(in_features)

def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
    mla_layer = MultiHeadAttentionWithRoPE(d_model, num_heads, rope)
    mla_layer.load_state_dict({"weights_q": q_proj_weight, "weights_k": k_proj_weight, "weights_v": v_proj_weight, "weights_o": o_proj_weight})
    return mla_layer(in_features, token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    
    layer = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
    return layer(in_query_or_key, token_positions)

class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, theta:int, max_seq_len:int):
        super().__init__()
        assert d_model % num_heads == 0
        d_k = d_model // num_heads
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
        self.attn = MultiHeadAttentionWithRoPE(d_model, num_heads, self.rope)
        self.ffn = swiglu(d_model, d_ff)
    
    def forward(self, x: torch.Tensor):
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len)
        y1 = x + self.attn(self.ln1(x), token_positions)
        y = y1 + self.ffn(self.ln2(y1))
        return y

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    block = TransformerBlock(d_model, num_heads, d_ff, theta, max_seq_len)
    state = {
    "ln1.weights": weights["ln1.weight"],
    "ln2.weights": weights["ln2.weight"],
    "attn.weights_q": weights["attn.q_proj.weight"],
    "attn.weights_k": weights["attn.k_proj.weight"],
    "attn.weights_v": weights["attn.v_proj.weight"],
    "attn.weights_o": weights["attn.output_proj.weight"],
    "ffn.weights1": weights["ffn.w1.weight"],
    "ffn.weights2": weights["ffn.w2.weight"],
    "ffn.weights3": weights["ffn.w3.weight"],
    }
    block.load_state_dict(state)
    return block(in_features)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model:int, num_layers: int, num_heads:int, d_ff:int, theta:int):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, theta, context_length) for i in range(num_layers)])
        self.rmsnorm = RMSNorm(d_model)
        self.Linear = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.rmsnorm(x)
        out = self.Linear(x)
        return out

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    llm = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)

    state = {
        "embedding.embeddings": weights["token_embeddings.weight"],
        "rmsnorm.weights": weights["ln_final.weight"],
        "Linear.weights": weights["lm_head.weight"]
    }
    
    for i in range(num_layers):
        state[f"blocks.{i}.ln1.weights"] = weights[f"layers.{i}.ln1.weight"]
        state[f"blocks.{i}.ln2.weights"] = weights[f"layers.{i}.ln2.weight"]
        state[f"blocks.{i}.attn.weights_q"] = weights[f"layers.{i}.attn.q_proj.weight"]
        state[f"blocks.{i}.attn.weights_k"] = weights[f"layers.{i}.attn.k_proj.weight"]
        state[f"blocks.{i}.attn.weights_v"] = weights[f"layers.{i}.attn.v_proj.weight"]
        state[f"blocks.{i}.attn.weights_o"] = weights[f"layers.{i}.attn.output_proj.weight"]
        state[f"blocks.{i}.ffn.weights1"] = weights[f"layers.{i}.ffn.w1.weight"]
        state[f"blocks.{i}.ffn.weights2"] = weights[f"layers.{i}.ffn.w2.weight"]
        state[f"blocks.{i}.ffn.weights3"] = weights[f"layers.{i}.ffn.w3.weight"]
    llm.load_state_dict(state)
    return llm(in_indices)
        

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    layer = RMSNorm(d_model, eps)
    layer.load_state_dict({"weights": weights})
    return layer(in_features)

def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    return in_features * torch.sigmoid(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    sample_size = batch_size * context_length
    start_points = np.random.randint(0, len(dataset)-context_length, batch_size)
    inputs = np.stack([dataset[i: i+context_length] for i in start_points])
    labels = np.stack([dataset[i+1: i+context_length+1] for i in start_points])
    return (torch.from_numpy(inputs).long().to(device), torch.from_numpy(labels).long().to(device))


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


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    max_val = torch.max(inputs, dim=-1, keepdim=True).values
    shifted = (inputs - max_val)
    logsumexp = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    correct = torch.gather(shifted, dim=-1, index=targets.unsqueeze(-1))
    return (logsumexp - correct).mean()


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    eps = 1e-6
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return
    l2_norm = torch.sqrt(sum([g.pow(2).sum() for g in grads]))
    
    factor = max_l2_norm / (l2_norm + eps)
    for g in grads:
        g.mul_(factor)
            

class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, 
        lr: float = 1e-3, 
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2):
            if lr < 0:
                return ValueError(f"{lr} cannnot be less than 0")
            defaults = {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            }
            super().__init__(params, defaults)

    def step(self, closure: Optional[Callable]=None):
        loss = closure if closure is None else closure

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
            
                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                
                m, v = state["m"], state["v"]      # 拿引用
                t = state["t"]

                alpha_t = lr * math.sqrt(1-beta2**t) / (1-beta1**t)
                p.data -= lr * weight_decay * p.data
                state["m"] = beta1 * m + (1-beta1) * grad
                state["v"] = beta2 * v + (1-beta2) * grad ** 2
                p.data -= alpha_t * state["m"] / (torch.sqrt(state["v"]) + eps)

                state["t"] += 1

        return loss

def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters
    elif warmup_iters <= it <= cosine_cycle_iters:
        factor = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos((factor * math.pi))) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    output = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iter": iteration}
    torch.save(output, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    d = torch.load(src)
    model.load_state_dict(d.get("model"))
    optimizer.load_state_dict(d.get("optimizer"))
    return d.get("iter")


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return BPETokenizer(vocab, merges, special_tokens)
    


def run_train_bpe(input_path, vocab_size, special_tokens, **kwargs):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    # ---- 预分词 ----
    logger.info("Reading corpus from %s ...", input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    logger.info("Corpus size: %.2f MB", len(text.encode("utf-8")) / 1e6)

    sep = "|".join(re.escape(i) for i in special_tokens)
    chunks = re.split(sep, text) if sep else [text]   # sep 为空时别 split,否则会按每个字符切

    word_freqs: dict[tuple[bytes, ...], int] = defaultdict(int)
    for chunk in tqdm(chunks, desc="Pre-tokenizing", unit="chunk"):
        for match in re.finditer(PAT, chunk):
            word_freqs[tuple(bytes([b]) for b in match.group().encode("utf-8"))] += 1
    logger.info("Unique pre-tokens: %d", len(word_freqs))

    # ---- 初始 pair 统计 + 倒排 index ----
    words = [list(w) for w in word_freqs]
    freqs = [word_freqs[w] for w in word_freqs]

    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for i, word in enumerate(words):
        f = freqs[i]
        for x, y in zip(word[:-1], word[1:]):
            pair_counts[(x, y)] += f
            pair_to_words[(x, y)].add(i)

    # ---- 合并循环 ----
    target_vocab = vocab_size - len(special_tokens)
    pbar = tqdm(total=max(0, target_vocab - len(vocab)), desc="BPE merges", unit="merge")
    while len(vocab) < target_vocab:
        if not pair_counts:
            break

        # 关键改动:用 max 取最大,O(P);不要用 sorted 全排序 O(P log P)
        great_merge = max(pair_counts, key=lambda p: (pair_counts[p], p))
        a, b = great_merge
        comb = a + b
        vocab[len(vocab)] = comb
        merges.append(great_merge)

        for idx in list(pair_to_words[great_merge]):
            word = words[idx]
            f = freqs[idx]
            for x, y in zip(word[:-1], word[1:]):
                pair_counts[(x, y)] -= f
                if pair_counts[(x, y)] <= 0:
                    del pair_counts[(x, y)]
                pair_to_words[(x, y)].discard(idx)

            new_word, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(comb)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            words[idx] = new_word

            for x, y in zip(new_word[:-1], new_word[1:]):
                pair_counts[(x, y)] += f
                pair_to_words[(x, y)].add(idx)

        pair_to_words.pop(great_merge, None)

        pbar.update(1)
        if len(merges) % 100 == 0:                       # 每 100 次刷一下当前状态,别每步都刷拖慢
            pbar.set_postfix(vocab=len(vocab), last=str(comb)[:16])
    pbar.close()

    logger.info("Done. merges=%d, vocab(pre-special)=%d", len(merges), len(vocab))

    for t in special_tokens:
        vocab[len(vocab)] = t.encode("utf-8")
    return vocab, merges
