"""iTransformer forecasting with exact multi-head attention over variables.

Input:  x [B, T, N], where T = seq_len and N may vary between calls.
Output: y [B, S, N], where S = pred_len.

Each variable's complete history is one token. Shared Linear(T, D) embedding,
post-norm encoder blocks, and a shared Linear(D, S) prediction head implement
an encoder-only, direct multi-step forecaster. No temporal position encoding,
variable identity embeddings, decoder, or additional final LayerNorm is used.

Implementation choices where the supplied analysis leaves details open:
  * Single linear layers for embedding and prediction, with projection biases.
  * GELU FFNs by default; activation='relu' is also supported.
  * Full, noncausal variable attention using PyTorch SDPA and its default
    1 / sqrt(head_dim) scaling. No attention matrices are returned or explicitly
    constructed in Python. Attention remains exact and quadratic in N.
  * Dropout on embeddings, attention probabilities, attention residual outputs,
    FFN hidden activations, and FFN residual outputs, only during training.
  * Optional historical-window normalization, enabled by default: independent
    per-sample, per-variable population variance over T, with norm_eps=1e-5.
    Both mean and standard deviation are detached by default. There is no
    learnable affine transform in this input normalization.
  * Separate LayerNorm over D in each residual sublayer, with
    layer_norm_eps=1e-5 and learnable affine parameters by default.

Architectural defaults match the supplied frozen CPU configuration:
T=96, S=96, D=32, heads=2, layers=2, d_ff=64, dropout=0.1.
These are implementation defaults, not a claim about a unique paper setting.

Subset training requires no architectural change: pass x[:, :, variable_ids]
to train on those variables and use the corresponding target columns. Pass
all variables at inference. Parameters and normalization do not depend on N;
attention uses all variables present in the current call. The caller selects
subsets; variables_per_train_window does not silently truncate model inputs.

Training/data configuration fields such as label_len, batch_size, epochs,
lr, seed, threads, and device do not change forward behavior. The caller
controls optimization, randomness, thread settings, and module placement.
Window denormalization restores the input's scale, which may itself already
be dataset-standardized. This module performs no data loading or evaluation.
"""

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["Forecaster"]


def _positive_int(config: dict, name: str, default: int) -> int:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _positive_float(config: dict, name: str, default: float) -> float:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    value = float(value)
    if not 0.0 < value < float("inf"):
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _bool_option(config: dict, name: str, default: bool) -> bool:
    value = config.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {value!r}")
    return value


class _VariableSelfAttention(nn.Module):
    """[B, N, D] -> [B, N, D]; attention token axis is always N."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch, variables, width = h.shape
        # One shared projection, then split into distinct attention heads.
        # [B, N, 3D] -> [3, B, heads, N, head_dim].
        qkv = self.qkv(h).reshape(
            batch, variables, 3, self.n_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        # Each head compares N variable tokens with N variable tokens.
        # SDPA applies softmax(Q K^T / sqrt(head_dim)) V. Its dropout must
        # explicitly be disabled at evaluation; it does not inspect training.
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        # [B, heads, N, head_dim] -> [B, N, D].
        attended = attended.transpose(1, 2).reshape(batch, variables, width)
        return self.out_proj(attended)


class _EncoderLayer(nn.Module):
    """Two post-norm residual sublayers, preserving [B, N, D]."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        activation: str,
        layer_norm_eps: float,
        layer_norm_affine: bool,
    ):
        super().__init__()
        self.attention = _VariableSelfAttention(d_model, n_heads, dropout)
        self.attention_output_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, elementwise_affine=layer_norm_affine
        )
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, elementwise_affine=layer_norm_affine
        )
        # Linear and activation act only on each token's final feature axis.
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # U = LN(H + SelfAttn(H)); H_next = LN(U + FFN(U)).
        u = self.norm1(h + self.attention_output_dropout(self.attention(h)))
        return self.norm2(u + self.ffn(u))


class Forecaster(nn.Module):
    """Variable-count-independent iTransformer.

    Required interface:
        model = Forecaster(config)
        predictions = model(x)  # [B, seq_len, N] -> [B, pred_len, N]

    Core configuration keys:
        seq_len, pred_len, d_model, n_heads, e_layers, d_ff, dropout.

    Optional implementation choices:
        activation: 'gelu' (default) or 'relu'.
        use_norm: True by default.
        detach_norm_stats: True by default; affects both mean and scale.
        norm_eps: 1e-5, added to historical population variance.
        layer_norm_eps: 1e-5, independent of norm_eps.
        layer_norm_affine: True by default.

    Input must be a real floating-point tensor with a dtype/device compatible
    with the module, or used under an appropriate PyTorch autocast context.
    Float16/bfloat16 window statistics are accumulated in float32. When window
    normalization is enabled, the returned forecast uses the input dtype.
    """

    def __init__(self, config: dict):
        super().__init__()
        if not isinstance(config, dict):
            raise TypeError("config must be a dict")
        self.config = dict(config)
        self.seq_len = _positive_int(config, "seq_len", 96)
        self.pred_len = _positive_int(config, "pred_len", 96)
        self.d_model = _positive_int(config, "d_model", 32)
        self.n_heads = _positive_int(config, "n_heads", 2)
        self.e_layers = _positive_int(config, "e_layers", 2)
        self.d_ff = _positive_int(config, "d_ff", 64)
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        dropout = config.get("dropout", 0.1)
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise ValueError("dropout must be a number in [0, 1]")
        self.dropout = float(dropout)
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be a finite number in [0, 1]")

        activation = config.get("activation", "gelu")
        if activation not in ("gelu", "relu"):
            raise ValueError("activation must be 'gelu' or 'relu'")
        self.use_norm = _bool_option(config, "use_norm", True)
        self.detach_norm_stats = _bool_option(config, "detach_norm_stats", True)
        self.norm_eps = _positive_float(config, "norm_eps", 1e-5)
        layer_norm_eps = _positive_float(config, "layer_norm_eps", 1e-5)
        layer_norm_affine = _bool_option(config, "layer_norm_affine", True)

        # The T ordered history positions become input features of one token.
        self.embedding = nn.Linear(self.seq_len, self.d_model)
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.layers = nn.ModuleList([
            _EncoderLayer(
                self.d_model,
                self.n_heads,
                self.d_ff,
                self.dropout,
                activation,
                layer_norm_eps,
                layer_norm_affine,
            )
            for _ in range(self.e_layers)
        ])
        self.projection = nn.Linear(self.d_model, self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError("x must have shape [B, seq_len, N]")
        if x.size(1) != self.seq_len:
            raise ValueError(
                f"Expected seq_len={self.seq_len}, received {x.size(1)}"
            )
        if x.size(2) < 1:
            raise ValueError("x must contain at least one variable")
        if not x.is_floating_point():
            raise TypeError("x must have a real floating-point dtype")

        if self.use_norm:
            # Statistics depend only on this input window, independently for
            # each sample and variable. Both have shape [B, 1, N].
            work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
            stats_source = work.detach() if self.detach_norm_stats else work
            variance, mean = torch.var_mean(
                stats_source, dim=1, unbiased=False, keepdim=True
            )
            scale = torch.sqrt(variance + self.norm_eps)
            normalized = ((work - mean) / scale).to(dtype=x.dtype)
        else:
            normalized = x

        # Invert axes: [B, T, N] -> [B, N, T] -> [B, N, D].
        h = self.embedding_dropout(self.embedding(normalized.transpose(1, 2)))
        for layer in self.layers:
            h = layer(h)

        # Shared direct forecast: [B, N, D] -> [B, N, S] -> [B, S, N].
        prediction = self.projection(h).transpose(1, 2)
        if self.use_norm:
            prediction = prediction.to(dtype=scale.dtype) * scale + mean
            prediction = prediction.to(dtype=x.dtype)
        return prediction.contiguous()
