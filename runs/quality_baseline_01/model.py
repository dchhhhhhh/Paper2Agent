"""iTransformer: history [B, T, N] -> direct forecasts [B, S, N].

Each variable is one token. A shared Linear(T, D) embeds its complete
ordered history, and attention mixes variable tokens, never time tokens.
A shared Linear(D, S) predicts all future steps simultaneously. Parameters
are independent of N, so the same model supports training on variable
subsets and inference on all variables.

Implementation choices where the supplied analysis leaves details open:
* Single affine embedding and prediction head, both with bias.
* GELU by default; config['activation'] may alternatively specify 'relu'.
* Equal-width attention heads, requiring d_model % n_heads == 0.
* History-only normalization uses population variance and eps=1e-5.
  The history mean is detached; the standard deviation remains
  differentiable. Half/bfloat16 input statistics accumulate in float32.
* Post-LN blocks use affine LayerNorm over D with eps=1e-5. There is no
  additional final LayerNorm, positional embedding, or decoder.
* Configured dropout applies to embeddings, attention probabilities,
  attention residual branches, FFN activations, and FFN residual branches.
  Every dropout operation is disabled in eval mode.

Defaults match the supplied model configuration: T=S=96, D=32, two heads,
two encoder layers, d_ff=64, and dropout=0.1. Optional numerical settings
are 'norm_eps' and 'layer_norm_eps'.

Training/data fields such as label_len, variables_per_train_window,
batch_size, epochs, lr, seed, threads, and device are retained in config
but do not modify forward computation or global execution settings.
For subset training, the caller supplies x containing the selected
variables and uses targets with the same selection and ordering. Forward
always returns predictions for exactly the N variables supplied to it;
variable selection does not occur inside the model.

Internal denormalization restores the input scale. It does not undo any
external dataset preprocessing.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ['Forecaster']


def _positive_int(config: dict, name: str, default: int) -> int:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer; got {value!r}')
    return value


def _positive_float(config: dict, name: str, default: float) -> float:
    value = float(config.get(name, default))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f'{name} must be finite and positive; got {value!r}')
    return value


class _VariableSelfAttention(nn.Module):
    """Full, noncausal multi-head attention across N variable tokens."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout
        # A fused projection still supplies independent Q, K, and V weights.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_projection = nn.Linear(d_model, d_model, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, N, D]; each head has d_k = D / n_heads features.
        batch, n_variables, d_model = h.shape
        qkv = self.qkv(h).reshape(
            batch, n_variables, 3, self.n_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        # q, k, v: [B, heads, N, d_k]. SDPA computes
        # softmax(Q K^T / sqrt(d_k), dim=-1) V over the variable axis.
        # Delegate exact attention to PyTorch without constructing attention
        # scores in Python or requesting attention weights.
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(
            batch, n_variables, d_model
        )
        return self.out_projection(attended)


class _EncoderLayer(nn.Module):
    """Post-LN attention and token-wise FFN, preserving [B, N, D]."""

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
        self.attention_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, elementwise_affine=True
        )
        self.ffn_in = nn.Linear(d_model, d_ff, bias=True)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.activation_dropout = nn.Dropout(dropout)
        self.ffn_out = nn.Linear(d_ff, d_model, bias=True)
        self.ffn_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, elementwise_affine=True
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # U = LN_1(H + SelfAttention(H)), with residual-branch dropout.
        u = self.norm1(h + self.attention_dropout(self.attention(h)))
        # The shared FFN transforms only the final feature dimension:
        # [B, N, D] -> [B, N, d_ff] -> [B, N, D].
        ffn = self.ffn_in(u)
        ffn = self.activation_dropout(self.activation(ffn))
        ffn = self.ffn_out(ffn)
        # H_next = LN_2(U + FFN(U)).
        return self.norm2(u + self.ffn_dropout(ffn))


class Forecaster(nn.Module):
    """Encoder-only iTransformer with a runtime-sized variable axis.

    Args:
        config: Dictionary specifying seq_len, pred_len, d_model, n_heads,
            e_layers, d_ff, dropout, and optional activation/numerical eps.
            No configured variable count constrains the input.

    Input:
        Floating-point tensor x of shape [B, seq_len, N], N >= 1.
        Its dtype/device must be compatible with the module, as with
        ordinary PyTorch layers; autocast may be used by the caller.

    Output:
        Tensor of shape [B, pred_len, N], in the input variable order,
        input scale, and input dtype.
    """

    def __init__(self, config: dict):
        super().__init__()
        if not isinstance(config, dict):
            raise TypeError('config must be a dict')
        self.config = dict(config)
        self.seq_len = _positive_int(config, 'seq_len', 96)
        self.pred_len = _positive_int(config, 'pred_len', 96)
        self.d_model = _positive_int(config, 'd_model', 32)
        self.n_heads = _positive_int(config, 'n_heads', 2)
        self.e_layers = _positive_int(config, 'e_layers', 2)
        self.d_ff = _positive_int(config, 'd_ff', 64)
        self.dropout = float(config.get('dropout', 0.1))
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout <= 1.0:
            raise ValueError('dropout must be finite and between 0 and 1')
        if self.d_model % self.n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')
        self.activation_name = config.get('activation', 'gelu')
        if self.activation_name not in ('gelu', 'relu'):
            raise ValueError("activation must be 'gelu' or 'relu'")
        self.norm_eps = _positive_float(config, 'norm_eps', 1e-5)
        self.layer_norm_eps = _positive_float(config, 'layer_norm_eps', 1e-5)

        self.embedding = nn.Linear(self.seq_len, self.d_model, bias=True)
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.layers = nn.ModuleList([
            _EncoderLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                activation=self.activation_name,
                layer_norm_eps=self.layer_norm_eps,
            )
            for _ in range(self.e_layers)
        ])
        self.projection = nn.Linear(self.d_model, self.pred_len, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError('x must be a torch.Tensor')
        if x.ndim != 3:
            raise ValueError('x must have shape [B, seq_len, N]')
        if x.shape[1] != self.seq_len:
            raise ValueError(
                f'Expected history length {self.seq_len}, got {x.shape[1]}'
            )
        if x.shape[2] < 1:
            raise ValueError('x must contain at least one variable')
        if not x.is_floating_point():
            raise TypeError('x must have a floating-point dtype')

        # Compute one history-only mean and scale per sample/variable.
        # Statistics are [B, 1, N], with no pooling across variables.
        stats_x = (
            x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        )
        mean = stats_x.mean(dim=1, keepdim=True).detach()
        centered = stats_x - mean
        variance = centered.var(dim=1, keepdim=True, unbiased=False)
        scale = torch.sqrt(variance + self.norm_eps)
        normalized = (centered / scale).to(dtype=x.dtype)

        # Invert axes: [B, T, N] -> [B, N, T] -> [B, N, D].
        # History order is encoded by the embedding's input coordinates.
        h = self.embedding(normalized.transpose(1, 2))
        h = self.embedding_dropout(h)
        for layer in self.layers:
            h = layer(h)

        # Shared direct forecast: [B, N, D] -> [B, N, S] -> [B, S, N].
        forecast = self.projection(h).transpose(1, 2)
        # Broadcasting reuses only the corresponding variable's history
        # statistics. No targets or persistent normalization state are used.
        forecast = forecast.to(dtype=scale.dtype) * scale + mean
        return forecast.to(dtype=x.dtype).contiguous()
