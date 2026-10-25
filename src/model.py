"""Small, explicit encoder-decoder Transformer (Attention Is All You Need)."""
import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads, self.depth = heads, d_model // heads
        self.q, self.k, self.v = (nn.Linear(d_model, d_model) for _ in range(3))
        self.out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        b, qlen, _ = query.shape
        klen = key.size(1)
        split = lambda layer, x, n: layer(x).view(b, n, self.heads, self.depth).transpose(1, 2)
        q, k, v = split(self.q, query, qlen), split(self.k, key, klen), split(self.v, value, klen)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.depth)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = (weights @ v).transpose(1, 2).contiguous().view(b, qlen, -1)
        return self.out(context)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.attn(x, x, x, src_mask))
        return self.norm2(x + self.ff(x))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff):
        super().__init__()
        self.self_attn, self.cross_attn = MultiHeadAttention(d_model, heads), MultiHeadAttention(d_model, heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x, memory, tgt_mask, src_mask):
        x = self.norm1(x + self.self_attn(x, x, x, tgt_mask))
        x = self.norm2(x + self.cross_attn(x, memory, memory, src_mask))
        return self.norm3(x + self.ff(x))


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=64, heads=4, d_ff=128, layers=2, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.src_embed, self.tgt_embed = nn.Embedding(src_vocab, d_model), nn.Embedding(tgt_vocab, d_model)
        self.pos = PositionalEncoding(d_model)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, heads, d_ff) for _ in range(layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, heads, d_ff) for _ in range(layers)])
        self.projection = nn.Linear(d_model, tgt_vocab)

    def encode(self, src):
        src_mask = (src != self.pad_id)[:, None, None, :]
        x = self.pos(self.src_embed(src) * math.sqrt(self.src_embed.embedding_dim))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x, src_mask

    def decode(self, tgt, memory, src_mask):
        length = tgt.size(1)
        causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=tgt.device))[None, None, :, :]
        valid = (tgt != self.pad_id)[:, None, None, :]
        tgt_mask = causal & valid
        x = self.pos(self.tgt_embed(tgt) * math.sqrt(self.tgt_embed.embedding_dim))
        for layer in self.decoder:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.projection(x)

    def forward(self, src, tgt_input):
        memory, src_mask = self.encode(src)
        return self.decode(tgt_input, memory, src_mask)
