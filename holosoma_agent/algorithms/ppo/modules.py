from __future__ import annotations

import inspect

import torch
from torch import nn

from holosoma_agent.configs.ppo_config import LayerConfig, ModuleConfig


class ImgChLayerNorm(nn.Module):
    """Image channel-wise layer normalization."""

    def __init__(self, num_channels, eps: float = 1e-5):
        """Initialize ImgChLayerNorm module.

        Parameters
        ----------
        num_channels: int
            Number of channels in the input tensor
        eps: float, optional
            Small value to prevent division by zero, by default 1e-5
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        """Forward pass for image channel-wise layer normalization.

        Normalizes each channel of the input tensor independently.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape [B, C, H, W]

        Returns
        -------
        torch.Tensor
            Output tensor of shape [B, C, H, W]
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class CNNWrapper(nn.Module):
    """Wrapper module that handles reshaping for CNN layers when working with flattened inputs."""

    def __init__(
        self, cnn_layers, input_channels, input_height, input_width, flatten_output=True
    ):
        """Initialize CNNWrapper module.

        Wraps CNN layers to handle reshaping for CNN layers when working with flattened inputs.
        For instance, this is useful when giving the flattened output of a CNN layer to an MLP layer.

        Parameters
        ----------
        cnn_layers: nn.Module
            CNN layers to wrap
        input_channels: int
            Number of input channels
        input_height: int
            Height of input feature maps
        input_width: int
            Width of input feature maps
        flatten_output: bool, optional
            Whether to flatten the output, by default True
        """
        super().__init__()
        self.cnn_layers = cnn_layers
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.expected_input_size = input_channels * input_height * input_width
        self.flatten_output = flatten_output

    @property
    def output_size(self):
        """Computes the output size of the CNN layers by doing a forward pass with dummy data."""
        with torch.no_grad():
            dummy_input = torch.zeros(
                1, self.input_channels * self.input_height * self.input_width
            )
            dummy_output = self.forward(dummy_input)
            return dummy_output.shape[-1]

    def forward(self, x):
        """Forward pass for CNNWrapper module.

        Reshapes the input tensor to (batch_size, channels, height, width) and applies the CNN layers.
        If flatten_output is True, flattens the output back to (batch_size, -1).

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape [B, C, H, W]

        Returns
        -------
        torch.Tensor
            Output tensor of shape [B, -1] if flatten_output is True, otherwise [B, C, H, W]
        """
        # Validate input size
        batch_size = x.shape[0]
        if x.shape[1] != self.expected_input_size:
            raise ValueError(
                f"Input size mismatch: expected {self.expected_input_size} "
                f"(channels={self.input_channels}, height={self.input_height}, width={self.input_width}), "
                f"but got {x.shape[1]}"
            )

        # Reshape from flattened input to (batch_size, channels, height, width)
        x = x.view(batch_size, self.input_channels, self.input_height, self.input_width)

        # Apply CNN layers
        x = self.cnn_layers(x)

        if self.flatten_output:
            # Flatten back to (batch_size, -1)
            x = x.view(batch_size, -1)
        else:
            # x is currently [batch_size, channels, height, width]
            # Reshape to [batch_size, height * width, channels]
            x = x.view(batch_size, x.shape[1], -1)
            x = x.permute(0, 2, 1)

        return x


def build_mlp_layer(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    layer_config: LayerConfig,
):
    """Builds a multi-layer perceptron (MLP) layer.

    Parameters
    ----------
    input_dim: int
        Number of input dimensions
    hidden_dims: tuple[int, ...]
        Tuple of hidden dimensions
    output_dim: int
        Number of output dimensions
    layer_config: dict
        Dictionary containing:
        - activation: Activation function name (e.g., "ReLU")
        - dropout_prob: Dropout probability (default: 0)

    Returns
    -------
    nn.Sequential
        The constructed MLP layer
    """
    if hidden_dims is None:
        return None

    layers = []
    activation = getattr(nn, layer_config.activation)()
    dropout = layer_config.dropout_prob
    use_layer_norm = layer_config.use_layer_norm

    if len(hidden_dims) == 0:
        # No hidden layer, just one linear layer
        layers.append(nn.Linear(input_dim, output_dim))
    else:
        # First hidden layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dims[0]))
        layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        # Additional hidden layers
        for layer_idx in range(len(hidden_dims)):
            if layer_idx == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[layer_idx], output_dim))
            else:
                layers.append(
                    nn.Linear(hidden_dims[layer_idx], hidden_dims[layer_idx + 1])
                )
                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dims[layer_idx + 1]))
                layers.append(activation)
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))

    return nn.Sequential(*layers)


def build_cnn_layer(
    input_channels: int,
    input_height: int,
    input_width: int,
    hidden_channels: tuple[int, ...] | None,
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...],
    padding: str | int | tuple[str | int, ...],
    layer_config: LayerConfig,
    flatten_output: bool = True,
):
    """Builds a convolutional neural network layer that works with flattened inputs.

    Parameters
    ----------
        input_channels: int
            Number of input channels
        input_height: int
            Height of input feature maps
        input_width: int
            Width of input feature maps
        hidden_channels: tuple[int, ...]
            Tuple of channel dimensions (last value becomes output channels)
        kernel_size: int or tuple[int, ...]
            Kernel size for convolutions (int or tuple for per-layer values)
        stride: int or tuple[int, ...]
            Stride for convolutions (int or tuple for per-layer values)
        padding: str | int | tuple[str | int, ...]
            Padding mode (int, "same", "valid", or tuple for per-layer values)
        layer_config: dict
            Dictionary containing:
            - activation: Activation function name (e.g., "ReLU")
            - dropout_prob: Dropout probability (default: 0)
            - use_layer_norm: Whether to use layer normalization (default: False)

    Returns
    -------
        CNNWrapper
            The constructed CNN layer wrapped to handle flattened inputs/outputs
    """
    if hidden_channels is None:
        return None
    assert len(hidden_channels) > 0, "hidden_channels must be a non-empty tuple"

    layers: list[nn.Module] = []
    activation = getattr(nn, layer_config.encoder_activation)()
    dropout = layer_config.dropout_prob
    use_layer_norm = layer_config.use_layer_norm

    num_layers = len(hidden_channels)
    # Convert single values to tuples if needed
    if isinstance(kernel_size, int):
        kernel_sizes = (kernel_size,) * num_layers
    else:
        kernel_sizes = kernel_size
        if len(kernel_sizes) != num_layers:
            raise ValueError(
                f"kernel_size tuple length ({len(kernel_sizes)}) must match number of layers ({num_layers})"
            )

    if isinstance(stride, int):
        strides = (stride,) * num_layers
    else:
        strides = stride
        if len(strides) != num_layers:
            raise ValueError(
                f"stride tuple length ({len(strides)}) must match number of layers ({num_layers})"
            )

    if isinstance(padding, (str, int)):
        paddings = (padding,) * num_layers
    else:
        paddings = padding
        if len(paddings) != num_layers:
            raise ValueError(
                f"padding tuple length ({len(paddings)}) must match number of layers ({num_layers})"
            )

    # Helper function to get padding value
    def get_padding_value(padding_spec, kernel_size_val):
        if padding_spec == "same":
            return kernel_size_val // 2
        if padding_spec == "valid":
            return 0
        return padding_spec

    # Build layers
    current_in_channels = input_channels
    for layer_idx in range(num_layers):
        current_out_channels = hidden_channels[layer_idx]
        current_kernel_size = kernel_sizes[layer_idx]
        current_stride = strides[layer_idx]
        current_padding = get_padding_value(paddings[layer_idx], current_kernel_size)

        # Add convolution layer
        layers.append(
            nn.Conv2d(
                current_in_channels,
                current_out_channels,
                kernel_size=current_kernel_size,
                stride=current_stride,
                padding=current_padding,
            )
        )

        # Add layer norm, activation and dropout for all layers except the last one
        if layer_idx < num_layers - 1:
            if use_layer_norm:
                layers.append(ImgChLayerNorm(current_out_channels))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout2d(p=dropout))

        current_in_channels = current_out_channels

    cnn_sequential = nn.Sequential(*layers)

    # Wrap with CNNWrapper to handle flattened inputs/outputs
    return CNNWrapper(
        cnn_sequential, input_channels, input_height, input_width, flatten_output
    )


class BaseModule(nn.Module):
    def __init__(
        self,
        obs_dim_dict: dict[str, int | tuple[int, ...]],
        output_dim: int,
        module_config_dict: ModuleConfig,
    ):
        """
        Initialize the BaseModule.

        Parameters
        ----------
        obs_dim_dict : dict[str, int | tuple[int, ...]]
            Dictionary containing observation dimensions for different types of observations.
            For example, {"mlp": 128, "cnn_encoder": (3, 64, 64)}
        output_dim : int
            Dimension of the output.
        module_config_dict : ModuleConfig
            Configuration for the module.
        """
        super().__init__()
        self.obs_dim_dict = obs_dim_dict
        self.output_dim = output_dim
        self._build_network_layer(module_config_dict)

    def _build_network_layer(self, module_config: ModuleConfig):
        layer_type = module_config.module_type
        layer_config = module_config.layer_config
        if layer_type == "MLP":
            mlp_obs_dim = self.obs_dim_dict["mlp"]
            self.module = build_mlp_layer(
                mlp_obs_dim,
                layer_config.hidden_dims,
                self.output_dim,
                layer_config,
            )
        elif layer_type == "CNNEncoder":
            # (channel, height, width)
            cnn_encoder_obs_dim = self.obs_dim_dict["encoder"]
            mlp_obs_dim = self.obs_dim_dict["mlp"]
            self.encoder = build_cnn_layer(
                cnn_encoder_obs_dim[0],
                cnn_encoder_obs_dim[1],
                cnn_encoder_obs_dim[2],
                layer_config.hidden_channels,
                layer_config.kernel_size,
                layer_config.stride,
                layer_config.padding,
                layer_config,
                flatten_output=True,
            )
            encoder_output_dim = self.encoder.output_size
            self.module = build_mlp_layer(
                mlp_obs_dim + encoder_output_dim,
                layer_config.hidden_dims,
                self.output_dim,
                layer_config,
            )
        elif layer_type == "MLPEncoder":
            mlp_encoder_obs_dim = self.obs_dim_dict["encoder"]
            mlp_obs_dim = self.obs_dim_dict["mlp"]
            assert layer_config.encoder_output_dim is not None, (
                "encoder_output_dim must be specified for MLPEncoder"
            )
            encoder_output_dim = layer_config.encoder_output_dim
            self.encoder = build_mlp_layer(
                mlp_encoder_obs_dim,
                layer_config.encoder_hidden_dims,
                encoder_output_dim,
                layer_config,
            )
            self.module = build_mlp_layer(
                mlp_obs_dim + encoder_output_dim,
                layer_config.hidden_dims,
                self.output_dim,
                layer_config,
            )
        else:
            raise NotImplementedError(f"Unsupported layer type: {layer_type}")

    def forward(self, policy_input: torch.Tensor):
        # Only forward the MLP layer
        return self.module(policy_input)
