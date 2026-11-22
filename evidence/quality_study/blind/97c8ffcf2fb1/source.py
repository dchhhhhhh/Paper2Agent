import torch
from torch import nn
from torch.nn import functional as F
__all__ = ['Forecaster']

def _integer_option(config: dict, name: str, default: int, minimum: int=1) -> int:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'config[{name!r}] must be an integer')
    if value < minimum:
        raise ValueError(f'config[{name!r}] must be >= {minimum}, got {value}')
    return value

class _VariableSelfAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.output_projection = nn.Linear(d_model, d_model, bias=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, n_variables, d_model = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, n_variables, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attention_dropout if self.training else 0.0, is_causal=False)
        attended = attended.transpose(1, 2).contiguous().view(batch, n_variables, d_model)
        return self.output_projection(attended)

class _EncoderBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, layer_norm_eps: float):
        super().__init__()
        self.attention = _VariableSelfAttention(d_model, n_heads, dropout)
        self.attention_output_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff, bias=True), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model, bias=True), nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.norm1(tokens + self.attention_output_dropout(self.attention(tokens)))
        return self.norm2(tokens + self.ffn(tokens))

class Forecaster(nn.Module):

    def __init__(self, config: dict):
        super().__init__()
        if not isinstance(config, dict):
            raise TypeError('config must be a dict')
        self.config = dict(config)
        self.seq_len = _integer_option(config, 'seq_len', 96)
        self.pred_len = _integer_option(config, 'pred_len', 96)
        self.label_len = _integer_option(config, 'label_len', 48, minimum=0)
        self.d_model = _integer_option(config, 'd_model', 32)
        self.n_heads = _integer_option(config, 'n_heads', 2)
        self.e_layers = _integer_option(config, 'e_layers', 2)
        self.d_ff = _integer_option(config, 'd_ff', 64)
        self.dropout = float(config.get('dropout', 0.1))
        if self.d_model % self.n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError('dropout must be a finite number in [0, 1]')
        self.normalization_eps = 1e-05
        self.layer_norm_eps = 1e-05
        self.embedding = nn.Linear(self.seq_len, self.d_model, bias=True)
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.encoder = nn.ModuleList([_EncoderBlock(d_model=self.d_model, n_heads=self.n_heads, d_ff=self.d_ff, dropout=self.dropout, layer_norm_eps=self.layer_norm_eps) for _ in range(self.e_layers)])
        self.projection = nn.Linear(self.d_model, self.pred_len, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError('x must be a torch.Tensor')
        if x.ndim != 3:
            raise ValueError(f'Expected x with shape [B, T, N], got {tuple(x.shape)}')
        if x.size(1) != self.seq_len:
            raise ValueError(f'Expected history length {self.seq_len}, got {x.size(1)}')
        if x.size(2) < 1:
            raise ValueError('x must contain at least one variable')
        if not x.is_floating_point():
            raise TypeError('x must have a floating-point dtype')
        variance, mean = torch.var_mean(x.detach(), dim=1, unbiased=False, keepdim=True)
        scale = torch.sqrt(variance + self.normalization_eps)
        normalized = (x - mean) / scale
        tokens = self.embedding(normalized.transpose(1, 2))
        tokens = self.embedding_dropout(tokens)
        for block in self.encoder:
            tokens = block(tokens)
        forecast = self.projection(tokens).transpose(1, 2)
        return (forecast * scale + mean).contiguous()
