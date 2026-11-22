import torch
from torch import nn
from torch.nn import functional as F
__all__ = ['Forecaster']

def _positive_int(config: dict, name: str, default: int) -> int:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer; got {value!r}')
    return value

class _VariableSelfAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch, n_variables, d_model = h.shape
        qkv = self.qkv(h).reshape(batch, n_variables, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attention_dropout if self.training else 0.0, is_causal=False)
        attended = attended.transpose(1, 2).contiguous().view(batch, n_variables, d_model)
        return self.out_proj(attended)

class _EncoderBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attention = _VariableSelfAttention(d_model, n_heads, dropout)
        self.attention_output_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-05, elementwise_affine=True)
        self.ffn_in = nn.Linear(d_model, d_ff, bias=True)
        self.ffn_hidden_dropout = nn.Dropout(dropout)
        self.ffn_out = nn.Linear(d_ff, d_model, bias=True)
        self.ffn_output_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-05, elementwise_affine=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        u = self.norm1(h + self.attention_output_dropout(self.attention(h)))
        ffn = self.ffn_in(u)
        ffn = self.ffn_hidden_dropout(F.gelu(ffn))
        ffn = self.ffn_out(ffn)
        return self.norm2(u + self.ffn_output_dropout(ffn))

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
        self.variables_per_train_window = _positive_int(config, 'variables_per_train_window', 172)
        self.label_len = config.get('label_len', 48)
        self.dropout = float(config.get('dropout', 0.1))
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError('dropout must be finite and satisfy 0 <= p < 1')
        if self.d_model % self.n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')
        self.input_norm_eps = 1e-05
        self.embedding = nn.Linear(self.seq_len, self.d_model, bias=True)
        self.encoder = nn.ModuleList([_EncoderBlock(self.d_model, self.n_heads, self.d_ff, self.dropout) for _ in range(self.e_layers)])
        self.projection = nn.Linear(self.d_model, self.pred_len, bias=True)

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError('x must be a torch.Tensor')
        if x.ndim != 3:
            raise ValueError('x must have shape [B, seq_len, N]')
        if x.shape[1] != self.seq_len:
            raise ValueError(f'Expected history length {self.seq_len}; got {x.shape[1]}')
        if x.shape[0] < 1 or x.shape[2] < 1:
            raise ValueError('Batch size and variable count must be positive')
        if not x.is_floating_point():
            raise TypeError('x must have a floating-point dtype')

    def sample_training_variables(self, x: torch.Tensor, *, generator=None) -> tuple:
        if not self.training:
            raise RuntimeError('Variable sampling is a training-only helper')
        self._validate_input(x)
        n_variables = x.shape[2]
        n_keep = min(self.variables_per_train_window, n_variables)
        if n_keep == n_variables:
            indices = torch.arange(n_variables, device=x.device)
            return (x, indices)
        indices = torch.randperm(n_variables, device=x.device, generator=generator)[:n_keep]
        return (x.index_select(dim=2, index=indices), indices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        stats_x = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        with torch.no_grad():
            mean = stats_x.mean(dim=1, keepdim=True)
            variance = stats_x.var(dim=1, keepdim=True, unbiased=False)
            scale = torch.sqrt(variance + self.input_norm_eps)
        normalized = ((stats_x - mean) / scale).to(dtype=x.dtype)
        h = self.embedding(normalized.transpose(1, 2))
        for block in self.encoder:
            h = block(h)
        prediction = self.projection(h).transpose(1, 2)
        prediction = prediction * scale + mean
        return prediction.to(dtype=x.dtype).contiguous()
