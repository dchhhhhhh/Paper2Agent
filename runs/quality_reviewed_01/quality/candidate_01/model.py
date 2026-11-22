"""iTransformer: an encoder-only forecaster with attention across variables.

Tensor conventions:
    Input:             [B, T, N], historical observations in time order.
    Variable tokens:   [B, N, T], one complete history per variable.
    Embedded tokens:   [B, N, D].
    Attention Q/K/V:   [B, H, N, D // H].
    Projected tokens:  [B, N, S].
    Output:            [B, S, N], the entire future window.

Defaults match the supplied local model configuration: T=S=96, D=32,
H=2, L=2, d_ff=64, and dropout=0.1. These are local configuration choices.

Implementation choices for details not uniquely specified in the supplied
paper analysis:
    * Shared, biased linear history embedding and prediction projection.
    * GELU activation in each shared D -> d_ff -> D feed-forward network.
    * Historical normalization independently for each sample and variable,
      using population variance and sqrt(variance + 1e-5). Both the mean
      and scale are detached from autograd, then reused to restore outputs.
    * Two post-residual LayerNorm operations per block, each over D only,
      with epsilon=1e-5 and learnable affine parameters. No final LayerNorm.
    * Dropout on embeddings, attention probabilities, attention residual
      updates, FFN hidden activations, and FFN residual updates.
    * Full, unmasked, noncausal attention over the supplied variable tokens.
      PyTorch SDPA computes standard dense attention; this module does not
      explicitly allocate attention scores or substitute an approximation.

No parameter depends on N and no positional or variable-ID embedding is used.
The caller can supply [B, 96, 172] during subset training and [B, 96, 862]
during full-variable evaluation using the same model. The caller selects
variable subsets and matching targets; forward preserves their input order
and predicts every supplied variable. Subsets may differ between samples.

label_len is retained as interface metadata and does not enter computation.
Batch sizes, sampling policy, epochs, optimizer, learning-rate schedule, seed,
thread count, and dataset preprocessing belong to the caller. Construction
neither configures global runtime state nor performs training or evaluation.
Outputs have the same scale as the input to this module, including any
external dataset-level preprocessing already applied to that input.
"""

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["Forecaster"]


def _integer_option(config: dict, name: str, default: int, minimum: int = 1) -> int:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"config[{name!r}] must be an integer")
    if value < minimum:
        raise ValueError(f"config[{name!r}] must be >= {minimum}, got {value}")
    return value


class _VariableSelfAttention(nn.Module):
    """Multi-head self-attention over N variables, preserving [B, N, D]."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_dropout = dropout
        # A packed affine map provides distinct learned Q, K, and V maps.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.output_projection = nn.Linear(d_model, d_model, bias=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, n_variables, d_model = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch, n_variables, 3, self.n_heads, self.head_dim
        )
        # [B, N, 3, H, d_k] -> [3, B, H, N, d_k].
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)

        # Softmax is over the N keys for each variable query. SDPA uses
        # the standard 1/sqrt(d_k) scale. Evaluation must explicitly pass
        # zero dropout because SDPA does not inspect module.training.
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        # Concatenate heads: [B, H, N, d_k] -> [B, N, D].
        attended = attended.transpose(1, 2).contiguous().view(
            batch, n_variables, d_model
        )
        return self.output_projection(attended)


class _EncoderBlock(nn.Module):
    """Two post-normalized residual updates; all representations are [B,N,D]."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.attention = _VariableSelfAttention(d_model, n_heads, dropout)
        self.attention_output_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, elementwise_affine=True
        )
        # Linear maps act on the last dimension and share parameters
        # across both samples and variable tokens.
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, elementwise_affine=True
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # U = LayerNorm_1(H + SelfAttention(H)), including local dropout.
        tokens = self.norm1(
            tokens + self.attention_output_dropout(self.attention(tokens))
        )
        # H_next = LayerNorm_2(U + FFN(U)), including local dropout.
        return self.norm2(tokens + self.ffn(tokens))


class Forecaster(nn.Module):
    """Shared-variable iTransformer mapping [B, seq_len, N] to [B, pred_len, N].

    Recognized architectural config keys are seq_len, pred_len, d_model,
    n_heads, e_layers, d_ff, and dropout. label_len is stored but unused.
    Additional experiment keys are retained in config without changing the
    architecture or global PyTorch state. d_model must divide by n_heads.
    """

    def __init__(self, config: dict):
        super().__init__()
        if not isinstance(config, dict):
            raise TypeError("config must be a dict")
        self.config = dict(config)

        self.seq_len = _integer_option(config, "seq_len", 96)
        self.pred_len = _integer_option(config, "pred_len", 96)
        self.label_len = _integer_option(config, "label_len", 48, minimum=0)
        self.d_model = _integer_option(config, "d_model", 32)
        self.n_heads = _integer_option(config, "n_heads", 2)
        self.e_layers = _integer_option(config, "e_layers", 2)
        self.d_ff = _integer_option(config, "d_ff", 64)
        self.dropout = float(config.get("dropout", 0.1))

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError("dropout must be a finite number in [0, 1]")

        self.normalization_eps = 1e-5
        self.layer_norm_eps = 1e-5
        self.embedding = nn.Linear(self.seq_len, self.d_model, bias=True)
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.encoder = nn.ModuleList(
            [
                _EncoderBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_ff=self.d_ff,
                    dropout=self.dropout,
                    layer_norm_eps=self.layer_norm_eps,
                )
                for _ in range(self.e_layers)
            ]
        )
        self.projection = nn.Linear(self.d_model, self.pred_len, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict future values from floating-point historical input.

        Args:
            x: [B, seq_len, N], with N >= 1. Time coordinates must remain
               in chronological order. N can change on every forward call.

        Returns:
            A contiguous [B, pred_len, N] tensor in input variable order,
            with historical-window normalization inverted. No future
            observations, decoder prefix, or variable identifiers are used.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B, T, N], got {tuple(x.shape)}")
        if x.size(1) != self.seq_len:
            raise ValueError(
                f"Expected history length {self.seq_len}, got {x.size(1)}"
            )
        if x.size(2) < 1:
            raise ValueError("x must contain at least one variable")
        if not x.is_floating_point():
            raise TypeError("x must have a floating-point dtype")

        # Both statistics have shape [B, 1, N]; the only reduction is over
        # the historical time axis. Detachment is an explicit local choice.
        variance, mean = torch.var_mean(
            x.detach(), dim=1, unbiased=False, keepdim=True
        )
        scale = torch.sqrt(variance + self.normalization_eps)
        normalized = (x - mean) / scale

        # [B, T, N] -> [B, N, T] -> [B, N, D]. The shared embedding
        # preserves ordered time coordinates without positional embeddings.
        tokens = self.embedding(normalized.transpose(1, 2))
        tokens = self.embedding_dropout(tokens)
        for block in self.encoder:
            tokens = block(tokens)

        # [B, N, D] -> [B, N, S] -> [B, S, N]. Broadcast the same historical
        # mean and scale across all S future steps.
        forecast = self.projection(tokens).transpose(1, 2)
        return (forecast * scale + mean).contiguous()
