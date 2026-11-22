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

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_projection = nn.Linear(d_model, d_model, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch, n_variables, d_model = h.shape
        qkv = self.qkv(h).reshape(batch, n_variables, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout_p if self.training else 0.0, is_causal=False)
        attended = attended.transpose(1, 2).reshape(batch, n_variables, d_model)
        return self.out_projection(attended)

class _EncoderLayer(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, activation: str, layer_norm_eps: float):
        super().__init__()
        self.attention = _VariableSelfAttention(d_model, n_heads, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=True)
        self.ffn_in = nn.Linear(d_model, d_ff, bias=True)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.activation_dropout = nn.Dropout(dropout)
        self.ffn_out = nn.Linear(d_ff, d_model, bias=True)
        self.ffn_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        u = self.norm1(h + self.attention_dropout(self.attention(h)))
        ffn = self.ffn_in(u)
        ffn = self.activation_dropout(self.activation(ffn))
        ffn = self.ffn_out(ffn)
        return self.norm2(u + self.ffn_dropout(ffn))

class Forecaster(nn.Module):

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
        self.norm_eps = _positive_float(config, 'norm_eps', 1e-05)
        self.layer_norm_eps = _positive_float(config, 'layer_norm_eps', 1e-05)
        self.embedding = nn.Linear(self.seq_len, self.d_model, bias=True)
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.layers = nn.ModuleList([_EncoderLayer(d_model=self.d_model, n_heads=self.n_heads, d_ff=self.d_ff, dropout=self.dropout, activation=self.activation_name, layer_norm_eps=self.layer_norm_eps) for _ in range(self.e_layers)])
        self.projection = nn.Linear(self.d_model, self.pred_len, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError('x must be a torch.Tensor')
        if x.ndim != 3:
            raise ValueError('x must have shape [B, seq_len, N]')
        if x.shape[1] != self.seq_len:
            raise ValueError(f'Expected history length {self.seq_len}, got {x.shape[1]}')
        if x.shape[2] < 1:
            raise ValueError('x must contain at least one variable')
        if not x.is_floating_point():
            raise TypeError('x must have a floating-point dtype')
        stats_x = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        mean = stats_x.mean(dim=1, keepdim=True).detach()
        centered = stats_x - mean
        variance = centered.var(dim=1, keepdim=True, unbiased=False)
        scale = torch.sqrt(variance + self.norm_eps)
        normalized = (centered / scale).to(dtype=x.dtype)
        h = self.embedding(normalized.transpose(1, 2))
        h = self.embedding_dropout(h)
        for layer in self.layers:
            h = layer(h)
        forecast = self.projection(h).transpose(1, 2)
        forecast = forecast.to(dtype=scale.dtype) * scale + mean
        return forecast.to(dtype=x.dtype).contiguous()
