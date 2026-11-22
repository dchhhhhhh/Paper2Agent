"""iTransformer forecasting with variable tokens and exact multi-head attention.

Dimensions:
    Input:              [B, T, N], where T = seq_len.
    Inverted input:     [B, N, T].
    Embedded tokens:    [B, N, D], where D = d_model.
    Attention Q/K/V:    [B, H, N, D/H], where H = n_heads.
    Projected forecast: [B, N, S], where S = pred_len.
    Output:             [B, S, N].

All learned mappings are shared across variables. No parameter depends on N,
so the same model can train on selected variables and evaluate on all variables.
For example, supplying x[:, :, indices] with 172 selected variables produces
172 forecasts; supplying all 862 variables produces 862 forecasts. The caller
selects training subsets. Attention always uses every variable supplied in the
current call, so subset and full-variable predictions need not be identical.

Implementation choices where the supplied paper analysis leaves details open:
    * A single Linear(T, D) embedding and Linear(D, S) forecasting head.
    * GELU by default, optionally ReLU.
    * Optional per-sample, per-variable history normalization, enabled by default.
      Statistics use only the input history: population variance over T, with
      epsilon inside the square root. Gradients flow through these statistics.
    * Affine LayerNorm over D, with configurable numerical epsilon.
    * Dropout on embeddings, attention probabilities, residual branches, and
      the activated intermediate FFN representation; training mode only.

Normalization restores predictions to the model input's scale. It does not
undo any dataset-level preprocessing performed outside this module.

Architecture defaults match the supplied frozen CPU configuration. Training
settings such as batch size, optimizer, seed, threads, and variable sampling
belong to the caller. label_len is unused by this encoder-only architecture.
"""

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["Forecaster"]


def _positive_int(config: dict, name: str, default: int) -> int:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return value


def _positive_float(config: dict, name: str, default: float) -> float:
    value = float(config.get(name, default))
    if not 0.0 < value < float("inf"):
        raise ValueError(f"{name} must be finite and positive; got {value!r}")
    return value


class _VariableSelfAttention(nn.Module):
    """Unmasked multi-head attention over N variable tokens, preserving [B,N,D]."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_dropout = dropout
        # One projection computes distinct Q, K, V for every head economically.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, n_variables, d_model = tokens.shape
        # [B,N,3D] -> [B,N,3,H,D/H] -> three tensors [B,H,N,D/H].
        packed = self.qkv(tokens).reshape(
            batch, n_variables, 3, self.n_heads, self.head_dim
        )
        query, key, value = packed.permute(2, 0, 3, 1, 4).unbind(dim=0)

        # SDPA attends over the penultimate dimension, which must remain N.
        # Each head computes softmax(Q K^T / sqrt(head_dim)) V over variables.
        # SDPA itself does not inspect module.training, so set dropout here.
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        # [B,H,N,D/H] -> [B,N,H,D/H] -> [B,N,D].
        attended = attended.transpose(1, 2).reshape(
            batch, n_variables, d_model
        )
        return self.output_projection(attended)


class _EncoderBlock(nn.Module):
    """Algorithm 1 post-norm block; both residual paths preserve [B,N,D]."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        activation: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.attention = _VariableSelfAttention(d_model, n_heads, dropout)
        self.attention_residual_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.ffn_residual_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # LayerNorm computes statistics over D separately for each [b,n].
        tokens = self.norm1(
            tokens + self.attention_residual_dropout(self.attention(tokens))
        )
        tokens = self.norm2(
            tokens + self.ffn_residual_dropout(self.ffn(tokens))
        )
        return tokens


class Forecaster(nn.Module):
    """Encoder-only iTransformer: forward([B,seq_len,N]) -> [B,pred_len,N].

    Configuration keys and defaults:
        seq_len=96, pred_len=96, d_model=32, n_heads=2,
        e_layers=2, d_ff=64, dropout=0.1,
        activation='gelu', use_norm=True,
        normalization_eps=1e-5, layer_norm_eps=1e-5.

    d_model must be divisible by n_heads. N can change between calls and need
    not equal any dataset channel-count metadata in config. Inputs must be
    floating-point tensors on the model's device with a compatible dtype.
    No future targets, decoder input, positional embeddings, or variable-ID
    embeddings are used.
    """

    def __init__(self, config: dict):
        super().__init__()
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")
        self.config = dict(config)
        self.seq_len = _positive_int(config, "seq_len", 96)
        self.pred_len = _positive_int(config, "pred_len", 96)
        self.d_model = _positive_int(config, "d_model", 32)
        self.n_heads = _positive_int(config, "n_heads", 2)
        self.e_layers = _positive_int(config, "e_layers", 2)
        self.d_ff = _positive_int(config, "d_ff", 64)
        self.dropout = float(config.get("dropout", 0.1))
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1 inclusive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.activation = str(config.get("activation", "gelu")).lower()
        if self.activation not in ("gelu", "relu"):
            raise ValueError("activation must be 'gelu' or 'relu'")
        self.use_norm = config.get("use_norm", True)
        if not isinstance(self.use_norm, bool):
            raise TypeError("use_norm must be a boolean")
        self.normalization_eps = _positive_float(
            config, "normalization_eps", 1e-5
        )
        self.layer_norm_eps = _positive_float(config, "layer_norm_eps", 1e-5)

        # Historical coordinates are ordered input features of each variable.
        self.embedding = nn.Linear(self.seq_len, self.d_model)
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.encoder = nn.ModuleList([
            _EncoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                activation=self.activation,
                layer_norm_eps=self.layer_norm_eps,
            )
            for _ in range(self.e_layers)
        ])
        self.projection = nn.Linear(self.d_model, self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B,T,N]; got {tuple(x.shape)}")
        if x.shape[1] != self.seq_len:
            raise ValueError(
                f"Expected history length {self.seq_len}; got {x.shape[1]}"
            )
        if x.shape[2] < 1:
            raise ValueError("x must contain at least one variable")
        if not x.is_floating_point():
            raise TypeError("x must have a floating-point dtype")

        if self.use_norm:
            # [B,T,N] -> statistics [B,1,N], exclusively over input history.
            # Accumulate reduced-precision input statistics in float32.
            stats_x = (
                x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
            )
            variance, mean = torch.var_mean(
                stats_x, dim=1, unbiased=False, keepdim=True
            )
            scale = torch.sqrt(variance + self.normalization_eps)
            normalized = ((stats_x - mean) / scale).to(dtype=x.dtype)
        else:
            normalized = x

        # [B,T,N] -> [B,N,T] -> [B,N,D]. No dependence on a fixed N.
        tokens = self.embedding_dropout(
            self.embedding(normalized.transpose(1, 2))
        )
        for block in self.encoder:
            tokens = block(tokens)

        # Shared regression head predicts all S future coordinates at once.
        prediction = self.projection(tokens).transpose(1, 2)  # [B,S,N]
        if self.use_norm:
            # Broadcast [B,1,N] across S; preserve the projection output dtype.
            output_dtype = prediction.dtype
            prediction = (
                prediction.to(dtype=mean.dtype) * scale + mean
            ).to(dtype=output_dtype)
        return prediction.contiguous()
