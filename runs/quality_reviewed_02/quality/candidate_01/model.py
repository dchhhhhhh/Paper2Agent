"""iTransformer forecasting with exact attention over variable tokens.

Tensor contract:
    Input:  [B, T, N], where T = config['seq_len'].
    Tokens: [B, N, D], where D = config['d_model'].
    Output: [B, S, N], where S = config['pred_len'].
N is determined independently on every call; parameters do not depend on N.

Implementation choices where the supplied paper analysis is not definitive:
    * Shared affine temporal embedding T -> D and forecast projection D -> S.
    * Full, noncausal multi-head attention, with biased Q/K/V/output projections.
    * Post-norm encoder blocks; no additional final encoder LayerNorm.
    * GELU FFNs, with dropout on attention probabilities, attention outputs,
      FFN hidden activations, and FFN outputs. No embedding dropout.
    * LayerNorm over D, with epsilon 1e-5 and learnable affine parameters.
    * Per-example, per-variable history normalization over T, using population
      variance and epsilon 1e-5. Both mean and scale are detached from autograd.
      Low-precision inputs use float32 for normalization statistics.

Defaults follow the supplied frozen architecture: T=S=96, D=32, two heads,
FFN width 64, two encoder blocks, dropout 0.1. These are local experiment
choices, not a claim about the paper's selected Traffic configuration.

Variable sampling is separate from forward so forward always preserves N.
During training, pass a variable subset directly or use
sample_training_variables(), then select the SAME indices from the future
training targets. The helper selects 172 variables when at least 172 are
available, uniformly without replacement, sharing indices within a batch.
At evaluation, pass the complete variable set directly to forward.

Outputs are restored to the scale of the input window. Any dataset-level
standardization or inverse transformation belongs to the caller. label_len is
retained as interface metadata; the model uses no decoder inputs.

Device placement, thread count, random seeding, batches, optimization, and
checkpoint selection belong to the caller. Construction uses ordinary PyTorch
initialization and does not alter global runtime settings.
"""

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
    """Exact multi-head self-attention with both attention axes indexing N."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_dropout = dropout
        # One combined projection is equivalent to separate Q/K/V projections.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, N, D]. Each head has d_head = D / n_heads features.
        batch, n_variables, d_model = h.shape
        qkv = self.qkv(h).reshape(
            batch, n_variables, 3, self.n_heads, self.head_dim
        )
        # [3, B, heads, N, d_head], with contiguous final feature dimension.
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)

        # Computes softmax(Q K^T / sqrt(d_head)) V exactly, with conceptual
        # scores [B, heads, N, N]. No causal mask, approximation, or returned
        # attention matrices; PyTorch can select its efficient CPU backend.
        # SDPA requires explicit zero dropout during evaluation.
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch, n_variables, d_model
        )
        return self.out_proj(attended)


class _EncoderBlock(nn.Module):
    """Two post-norm residual sublayers operating on [B, N, D]."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float
    ):
        super().__init__()
        self.attention = _VariableSelfAttention(d_model, n_heads, dropout)
        self.attention_output_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model, eps=1e-5, elementwise_affine=True
        )
        self.ffn_in = nn.Linear(d_model, d_ff, bias=True)
        self.ffn_hidden_dropout = nn.Dropout(dropout)
        self.ffn_out = nn.Linear(d_ff, d_model, bias=True)
        self.ffn_output_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model, eps=1e-5, elementwise_affine=True
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # U = LayerNorm_D(H + Dropout(SelfAttention(H))).
        u = self.norm1(h + self.attention_output_dropout(self.attention(h)))
        # Shared token-wise FFN: [B, N, D] -> [B, N, d_ff] -> [B, N, D].
        ffn = self.ffn_in(u)
        ffn = self.ffn_hidden_dropout(F.gelu(ffn))
        ffn = self.ffn_out(ffn)
        # H_next = LayerNorm_D(U + Dropout(FFN(U))).
        return self.norm2(u + self.ffn_output_dropout(ffn))


class Forecaster(nn.Module):
    """Encoder-only iTransformer supporting arbitrary nonempty variable sets.

    Recognized architecture settings and frozen defaults:
        seq_len=96, pred_len=96, d_model=32, n_heads=2,
        e_layers=2, d_ff=64, dropout=0.1.

    Additional settings:
        label_len=48: retained as metadata only.
        variables_per_train_window=172: used by the sampling helper.

    Other experiment keys are retained in config without changing the model
    computation. No variable identifiers or fixed-N parameters are used, so
    the same weights work on training subsets and the complete variable set.
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
        self.variables_per_train_window = _positive_int(
            config, 'variables_per_train_window', 172
        )
        self.label_len = config.get('label_len', 48)
        self.dropout = float(config.get('dropout', 0.1))
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError('dropout must be finite and satisfy 0 <= p < 1')
        if self.d_model % self.n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')

        self.input_norm_eps = 1e-5
        # Each variable's ordered T-step history becomes one D-feature token.
        self.embedding = nn.Linear(self.seq_len, self.d_model, bias=True)
        self.encoder = nn.ModuleList([
            _EncoderBlock(
                self.d_model, self.n_heads, self.d_ff, self.dropout
            )
            for _ in range(self.e_layers)
        ])
        # Shared across variables; generates all future steps simultaneously.
        self.projection = nn.Linear(self.d_model, self.pred_len, bias=True)

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError('x must be a torch.Tensor')
        if x.ndim != 3:
            raise ValueError('x must have shape [B, seq_len, N]')
        if x.shape[1] != self.seq_len:
            raise ValueError(
                f'Expected history length {self.seq_len}; got {x.shape[1]}'
            )
        if x.shape[0] < 1 or x.shape[2] < 1:
            raise ValueError('Batch size and variable count must be positive')
        if not x.is_floating_point():
            raise TypeError('x must have a floating-point dtype')

    def sample_training_variables(
        self, x: torch.Tensor, *, generator=None
    ) -> tuple:
        """Return (selected_history, indices) for externally paired training.

        selected_history: [B, seq_len, K].
        indices: [K], torch.long, on x.device.
        K = min(variables_per_train_window, N).

        For full Traffic input, K=172. Sampling is uniform without replacement
        and uses one shared index set per batch, an explicit local choice.
        Already-small subsets are returned in their existing order. Use these
        indices along the last axis of the corresponding future-only targets.
        A supplied torch.Generator must be compatible with x.device.
        """
        if not self.training:
            raise RuntimeError('Variable sampling is a training-only helper')
        self._validate_input(x)
        n_variables = x.shape[2]
        n_keep = min(self.variables_per_train_window, n_variables)
        if n_keep == n_variables:
            indices = torch.arange(n_variables, device=x.device)
            return x, indices
        indices = torch.randperm(
            n_variables, device=x.device, generator=generator
        )[:n_keep]
        return x.index_select(dim=2, index=indices), indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map [B, seq_len, N] to [B, pred_len, N], preserving the supplied N."""
        self._validate_input(x)

        # Statistics use only this input's history, independently for each
        # (sample, variable); neither batches nor variables are pooled.
        stats_x = (
            x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        )
        with torch.no_grad():
            mean = stats_x.mean(dim=1, keepdim=True)  # [B, 1, N]
            variance = stats_x.var(dim=1, keepdim=True, unbiased=False)
            scale = torch.sqrt(variance + self.input_norm_eps)  # [B, 1, N]
        normalized = ((stats_x - mean) / scale).to(dtype=x.dtype)

        # Time positions are encoded by the shared affine map's T inputs.
        # The attention sequence dimension is N, never T.
        h = self.embedding(normalized.transpose(1, 2))  # [B, N, D]
        for block in self.encoder:
            h = block(h)
        prediction = self.projection(h).transpose(1, 2)  # [B, S, N]

        # Broadcast the historical statistics across the S future steps.
        prediction = prediction * scale + mean
        return prediction.to(dtype=x.dtype).contiguous()
