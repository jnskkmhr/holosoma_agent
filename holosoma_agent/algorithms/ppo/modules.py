from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import List

import torch
from torch import nn


# ── Local config dataclasses (replaced holosoma.config_types.algo imports) ────


@dataclass
class LayerConfig:
    """Configuration for a neural-network layer."""

    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "ELU"
    dropout_prob: float = 0.0
    use_layer_norm: bool = False

    # Encoder-specific (MLPEncoder / CNNEncoder)
    encoder_activation: str = "ELU"
    encoder_output_dim: int | None = None
    encoder_hidden_dims: List[int] | None = None
    encoder_input_name: str = ""

    # CNN-specific
    input_channels: int = 1
    input_height: int = 1
    input_width: int = 1
    hidden_channels: tuple[int, ...] | None = None
    kernel_size: int | tuple[int, ...] = 3
    stride: int | tuple[int, ...] = 1
    padding: str | int | tuple = "same"

    module_input_name: tuple[str, ...] = ()


@dataclass
class ModuleConfig:
    """Configuration for a network module."""

    type: str = "MLP"
    input_dim: List[str] = field(default_factory=list)
    output_dim: List[str | int] = field(default_factory=list)
    layer_config: LayerConfig = field(default_factory=LayerConfig)
    min_noise_std: float | None = None
    min_mean_noise_std: float | None = None


# ── Network builders ──────────────────────────────────────────────────────────


class ImgChLayerNorm(nn.Module):
    """Image channel-wise layer normalization."""

    def __init__(self, num_channels, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class CNNWrapper(nn.Module):
    """Wrapper that reshapes flat inputs into (B, C, H, W) for CNN layers."""

    def __init__(
        self, cnn_layers, input_channels, input_height, input_width, flatten_output=True
    ):
        super().__init__()
        self.cnn_layers = cnn_layers
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.expected_input_size = input_channels * input_height * input_width
        self.flatten_output = flatten_output

    @property
    def output_size(self):
        with torch.no_grad():
            dummy = torch.zeros(
                1, self.input_channels * self.input_height * self.input_width
            )
            return self.forward(dummy).shape[-1]

    def forward(self, x):
        batch_size = x.shape[0]
        if x.shape[1] != self.expected_input_size:
            raise ValueError(
                f"Input size mismatch: expected {self.expected_input_size}, got {x.shape[1]}"
            )
        x = x.view(batch_size, self.input_channels, self.input_height, self.input_width)
        x = self.cnn_layers(x)
        if self.flatten_output:
            x = x.view(batch_size, -1)
        else:
            x = x.view(batch_size, x.shape[1], -1).permute(0, 2, 1)
        return x


def build_mlp_layer(input_dim, hidden_dims, output_dim, layer_config: LayerConfig):
    if hidden_dims is None:
        return None
    layers = []
    activation = getattr(nn, layer_config.activation)()
    dropout = layer_config.dropout_prob
    if len(hidden_dims) == 0:
        layers.append(nn.Linear(input_dim, output_dim))
    else:
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[i], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(activation)
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def build_cnn_layer(
    input_channels: int,
    input_height: int,
    input_width: int,
    hidden_channels: tuple[int, ...] | None,
    kernel_size,
    stride,
    padding,
    layer_config: LayerConfig,
    flatten_output: bool = True,
):
    if hidden_channels is None:
        return None
    assert len(hidden_channels) > 0
    layers: list[nn.Module] = []
    activation = getattr(nn, layer_config.encoder_activation)()
    dropout = layer_config.dropout_prob
    use_ln = layer_config.use_layer_norm
    n = len(hidden_channels)

    def _expand(v):
        return (v,) * n if isinstance(v, (int, str)) else v

    kernel_sizes = _expand(kernel_size)
    strides = _expand(stride)
    paddings = _expand(padding)

    def _pad(p, k):
        if p == "same":
            return k // 2
        if p == "valid":
            return 0
        return p

    cur_in = input_channels
    for i, out_ch in enumerate(hidden_channels):
        layers.append(
            nn.Conv2d(
                cur_in,
                out_ch,
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=_pad(paddings[i], kernel_sizes[i]),
            )
        )
        if i < n - 1:
            if use_ln:
                layers.append(ImgChLayerNorm(out_ch))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout2d(p=dropout))
        cur_in = out_ch
    return CNNWrapper(
        nn.Sequential(*layers),
        input_channels,
        input_height,
        input_width,
        flatten_output,
    )


# ── BaseModule ─────────────────────────────────────────────────────────────────


class BaseModule(nn.Module):
    """General-purpose network module (MLP / MLPEncoder / CNNEncoder).

    Parameters
    ----------
    obs_dim_dict : dict[str, int]
        Flat dimension of each observation group (history already included).
    module_config : ModuleConfig
    history_length : dict[str, int]
        History factor per observation group.
    """

    def __init__(
        self,
        obs_dim_dict: dict[str, int],
        module_config: ModuleConfig,
        history_length: dict[str, int],
    ):
        super().__init__()
        self.obs_dim_dict = obs_dim_dict
        self.module_config_dict = module_config
        self.history_length = history_length
        self._calculate_input_dim()
        self._calculate_output_dim()
        self._build_network_layer(module_config)

    def _calculate_input_dim(self):
        self.input_dim = 0
        self.input_dim_dict: dict = {}
        self.input_indices_dict: dict = {}
        cur = 0
        for each in self.module_config_dict.input_dim:
            if each in self.obs_dim_dict:
                dim = self.obs_dim_dict[each]
                self.input_dim += dim
                self.input_dim_dict[each] = dim
                self.input_indices_dict[each] = slice(cur, cur + dim)
                cur += dim
            elif isinstance(each, (int, float)):
                dim = int(each)
                self.input_dim += dim
                self.input_dim_dict[each] = dim
                self.input_indices_dict[each] = slice(cur, cur + dim)
                cur += dim
            else:
                fn = inspect.currentframe().f_code.co_name
                raise ValueError(f"{fn} - Unknown input type: {each}")

    def _calculate_output_dim(self):
        self.output_dim = 0
        for each in self.module_config_dict.output_dim:
            if isinstance(each, (int, float)):
                self.output_dim += int(each)
            else:
                fn = inspect.currentframe().f_code.co_name
                raise ValueError(f"{fn} - Unknown output type: {each}")

    def _build_network_layer(self, cfg: ModuleConfig):
        ltype = cfg.type
        lc = cfg.layer_config
        if ltype == "MLP":
            self.module = build_mlp_layer(
                self.input_dim, lc.hidden_dims, self.output_dim, lc
            )
        elif ltype == "CNNEncoder":
            self.encoder = build_cnn_layer(
                lc.input_channels,
                lc.input_height,
                lc.input_width,
                lc.hidden_channels,
                lc.kernel_size,
                lc.stride,
                lc.padding,
                lc,
            )
            enc_out = self.encoder.output_size
            mlp_in = sum(self.input_dim_dict[k] for k in lc.module_input_name)
            self.module = build_mlp_layer(
                mlp_in + enc_out, lc.hidden_dims, self.output_dim, lc
            )
        elif ltype == "MLPEncoder":
            enc_out = (
                lc.encoder_output_dim
                if lc.encoder_hidden_dims is not None
                else self.input_dim_dict[lc.encoder_input_name]
            )
            self.encoder = build_mlp_layer(
                self.input_dim_dict[lc.encoder_input_name],
                lc.encoder_hidden_dims,
                enc_out,
                lc,
            )
            mlp_in = sum(self.input_dim_dict[k] for k in lc.module_input_name)
            self.module = build_mlp_layer(
                mlp_in + enc_out, lc.hidden_dims, self.output_dim, lc
            )
        else:
            raise NotImplementedError(f"Unsupported layer type: {ltype}")

    def forward(self, policy_input):
        return self.module(policy_input)
