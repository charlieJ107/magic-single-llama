from typing import Optional
import torch
import math
from common import apply_rotary_emb, repeat_kv, ModelArgs


class Attention(torch.nn.Module):
    def __init__(self, args: ModelArgs, layer_id) -> None:
        super().__init__()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_head = args.n_heads
        self.n_rep = self.n_head // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.layer_id = layer_id

        self.wq = torch.nn.Linear(
            args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = torch.nn.Linear(
            args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = torch.nn.Linear(
            args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = torch.nn.Linear(
            args.n_heads * self.head_dim, args.dim, bias=False)
        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        # to: torch.device = x.device, dtype: torch.dtype = x.dtype
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos: start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos: start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        # (bs, cache_len + seqlen, n_local_heads, head_dim)
        keys = repeat_kv(keys, self.n_rep)
        # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        # (bs, n_local_heads, cache_len + seqlen, head_dim)
        keys = keys.transpose(1, 2)
        # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / \
            math.sqrt(self.head_dim)
        if mask is not None:
            # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = scores + mask
        scores = torch.nn.functional.softmax(
            scores.float(), dim=-1).type_as(xq)
        # (bs, n_local_heads, seqlen, head_dim)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)
