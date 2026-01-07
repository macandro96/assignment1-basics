import torch
import torch.nn as nn
import einops
from cs336_basics.utils import softmax


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        std = 2 / (in_features + out_features)
        data_tensor = torch.empty((out_features, in_features), dtype=dtype, device=device)
        data_tensor = torch.nn.init.trunc_normal_(data_tensor, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        self.W = nn.Parameter(
            data_tensor,
            requires_grad=True,
        )
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor):
        result = einops.einsum(x, self.W, "... dim_in, dim_out dim_in -> ... dim_out")
        return result


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        data_tensor = torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device)
        data_tensor = torch.nn.init.trunc_normal_(
            data_tensor,
            mean=0.0,
            std=1,
            a=-3.0,
            b=3.0,
        )
        self.W = nn.Parameter(data=data_tensor, requires_grad=True)

    def forward(self, x: torch.Tensor):
        x_flattened = einops.rearrange(x, "batch seq_length -> (batch seq_length)")
        embed_out = self.W[x_flattened]
        embed_out = einops.rearrange(embed_out, "(batch seq) dim -> batch seq dim", batch=x.shape[0], seq=x.shape[-1])
        return embed_out


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.dtype = dtype
        self.gain = nn.Parameter(
            torch.ones(d_model, dtype=dtype, device=device),
            requires_grad=True,
        )
        self.eps = eps

    def forward(self, x: torch.Tensor):
        # upcast x to float32
        x = x.to(torch.float32)

        # compute rms
        rms = torch.mean(x * x, dim=-1, keepdim=True)  # size batch_size, 1
        rms = torch.sqrt(rms + self.eps)  # size batch_size, 1

        out = x / rms
        out = einops.einsum(out, self.gain, "... dim, dim -> ... dim")
        return out.to(self.dtype)


class SwiGLU(nn.Module):
    """SwiGLU activation

    SwiGLU(x) = W2 * (SiLU(W1 * x) ⊙ W3 * x)
    where SiLU(x) = x * sigmoid(x)
    Shapes:
        - W1: dff, d_model
        - W2: d_model,  dff
        - W3: dff, d_model
    """

    def __init__(
        self,
        d_model: int,
        dff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.w1 = self._init_params(dff, d_model, device, dtype)
        self.w2 = self._init_params(d_model, dff, device, dtype)
        self.w3 = self._init_params(dff, d_model, device, dtype)

    def _init_params(
        self, in_feats: int, out_feats: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> nn.Parameter:
        std = 2 / (in_feats + out_feats)
        data_tensor = torch.nn.init.trunc_normal_(
            torch.empty((in_feats, out_feats), dtype=dtype, device=device), std=std, a=-3 * std, b=3 * std
        )
        param = nn.Parameter(
            data_tensor,
            requires_grad=True,
        )
        return param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = einops.einsum(x, self.w1, "... dm, dff dm -> ... dff")
        silu = w1_x * torch.sigmoid(w1_x)  # ... x dff

        w3_x = einops.einsum(x, self.w3, "... dm, dff dm -> ... dff")  # ... x dff

        out = silu * w3_x

        out = einops.einsum(out, self.w2, "... dff, dm dff -> ... dm")
        return out


def scaled_dot_product_attention(
    queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Scaled dot product attention with optional mask

    Args:
        - queries: query vector of shape (..., seq_len, d_k)
        - key: key vector of shape (..., seq_len, d_k)
        - value: value vector of shape (..., seq_len, d_k)
        - mask: optional mask value of shape (seq_len, seq_len)
    """
    d_k = queries.shape[-1]
    attention = einops.einsum(queries, keys, "... s1 d_k, ... s2 d_k -> ... s1 s2") / d_k ** (
        0.5
    )  # ... seq_len seq_len

    if mask is not None:
        mask_fill = torch.ones_like(mask) * float("-inf")
        mask_fill = mask_fill.masked_fill(mask, 0)
        attention = attention + mask_fill

    attention = softmax(attention, dim=-1)

    output = einops.einsum(attention, values, "... s1 s2, ... s2 d_k -> ... s1 d_k")
    return output


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        dim_level = torch.arange(0, d_k, 2) / d_k
        theta_dim = torch.pow(theta, dim_level)  # 1 x d_k / 2

        # now do across seq len
        seq_vec = torch.arange(0, max_seq_len).unsqueeze(1).repeat(1, d_k // 2)  # max_seq_len x d_k / 2
        thetas = seq_vec / theta_dim  # max_seq_len x d_k / 2

        cos_theta, sin_theta = torch.cos(thetas), torch.sin(thetas)
        rotation_matrix = torch.cat(
            [
                cos_theta.unsqueeze(-1),
                -sin_theta.unsqueeze(-1),
                sin_theta.unsqueeze(-1),
                cos_theta.unsqueeze(-1),
            ],
            dim=-1,
        )  # max_seq_len x d_k / 2 x 4
        self.register_buffer("rot", rotation_matrix, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        dk = x.shape[-1]
        data = einops.rearrange(x, "... seq_len (d1 d2) -> ... seq_len d1 d2", d1=dk // 2, d2=2)

        data = data.repeat_interleave(2, dim=-2)  # ... seq_len dk 2
        data = einops.rearrange(data, "... seq_len (d1 d2) d3 -> ... seq_len d1 (d2 d3)", d1=dk // 2, d3=2, d2=2)
        rot = self.rot
        if token_positions is not None:
            rot = self.rot[token_positions]
        else:
            rot = self.rot[: x.shape[-2]]  # seq_len x d_k / 2 x 4
        data = data * rot
        data = einops.rearrange(data, "... seq_len d1 (d2 d3) -> ... seq_len (d1 d2) d3", d1=dk // 2, d2=2, d3=2)

        data = torch.sum(data, dim=-1)  # ... seq_len dk
        return data


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RoPE | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = Linear(in_features=d_model, out_features=3 * d_model)
        self.w_o = Linear(d_model, d_model)
        self.rope = rope

    def forward(self, x: torch.Tensor, rope_token_positions: torch.Tensor | None = None):
        seq_len, d_model = x.shape[-2], x.shape[-1]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        Q, K, V = torch.chunk(self.qkv(x), chunks=3, dim=-1)  # N x d_model
        inner_dim = d_model // self.num_heads
        Q = Q.view(-1, seq_len, self.num_heads, inner_dim)
        Q = einops.rearrange(Q, "... seq_len h d -> ... h seq_len d")

        K = K.view(-1, seq_len, self.num_heads, inner_dim)
        K = einops.rearrange(K, "... seq_len h d -> ... h seq_len d")

        V = V.view(-1, seq_len, self.num_heads, inner_dim)
        V = einops.rearrange(V, "... seq_len h d -> ... h seq_len d")

        if self.rope:
            Q, K = self.rope(Q, rope_token_positions), self.rope(K, rope_token_positions)

        out = scaled_dot_product_attention(Q, K, V, causal_mask.bool())  # ...h, seq, d_model / h
        out = out.permute(0, 2, 1, 3).reshape(-1, seq_len, d_model)  # ... seq d_model
        return self.w_o(out)  # ... seq d_model


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RoPE | None = None):
        super().__init__()
        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, rope=rope)
        self.rmsnorm1 = RMSNorm(d_model=d_model)
        self.rmsnorm2 = RMSNorm(d_model=d_model)
        self.swiglu = SwiGLU(d_model=d_model, dff=d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        out = x + self.attn(self.rmsnorm1(x), rope_token_positions=token_positions)
        out = out + self.swiglu(self.rmsnorm2(out))
        return out
