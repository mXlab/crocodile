from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
from ..torch_utils import misc
from ..torch_utils import persistence
from ..torch_utils.ops import conv2d_resample
from ..torch_utils.ops import upfirdn2d
from ..torch_utils.ops import bias_act


# ----------------------------------------------------------------------------
@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


# ----------------------------------------------------------------------------
@dataclass
class MappingNetworkConfig:
    z_dim: int = 0
    num_layers: int = 8  # Number of mapping layers.
    embed_features: Optional[
        int
    ] = None  # Label embedding dimensionality, None = same as w_dim.
    layer_features: Optional[
        int
    ] = None  # Number of intermediate features in the mapping layers, None = same as w_dim.
    activation: str = "lrelu"  # Activation function: 'relu', 'lrelu', etc.
    lr_multiplier: float = 0.01  # Learning rate multiplier for the mapping layers.
    w_avg_beta: Optional[
        float
    ] = 0.995  # Decay for tracking the moving average of W during training, None = do not track.


class MappingNetwork(torch.nn.Module):
    def __init__(
        self,
        c_dim: int,  # Conditioning label (C) dimensionality, 0 = no label.
        w_dim: int,  # Intermediate latent (W) dimensionality.
        num_ws: int,  # Number of intermediate latents to output, None = do not broadcast.
        config: MappingNetworkConfig = MappingNetworkConfig(),
    ):
        super().__init__()
        self.z_dim = config.z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = config.num_layers
        self.w_avg_beta = config.w_avg_beta

        embed_features = config.embed_features
        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0

        layer_features = config.layer_features
        if layer_features is None:
            layer_features = w_dim
        features_list = (
            [config.z_dim + embed_features] + [layer_features] * (config.num_layers - 1) + [w_dim]
        )

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(config.num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(
                in_features,
                out_features,
                activation=config.activation,
                lr_multiplier=config.lr_multiplier,
            )
            setattr(self, f"fc{idx}", layer)

        if num_ws is not None and config.w_avg_beta is not None:
            self.register_buffer("w_avg", torch.zeros([w_dim]))

    def forward(
        self, z: torch.Tensor, c: torch.Tensor, truncation_psi: float = 1, truncation_cutoff: Optional[float] = None, skip_w_avg_update: bool = False
    ):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function("input"):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f"fc{idx}")
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function("update_w_avg"):
                self.w_avg.copy_(
                    x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta)
                )

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function("broadcast"):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function("truncate"):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(
                        x[:, :truncation_cutoff], truncation_psi
                    )
        return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class FullyConnectedLayer(nn.Module):
    def __init__(
        self,
        in_features: int,  # Number of input features.
        out_features: int,  # Number of output features.
        bias: bool = True,  # Apply additive bias before the activation function?
        activation: str = "linear",  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier: float = 1,  # Learning rate multiplier.
        bias_init: float = 0,  # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.randn([out_features, in_features]) / lr_multiplier
        )
        self.bias = (
            torch.nn.Parameter(torch.full([out_features], np.float32(bias_init)))
            if bias
            else None
        )
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x: torch.Tensor):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == "linear" and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,  # Number of input channels.
        out_channels: int,  # Number of output channels.
        kernel_size: int,  # Width and height of the convolution kernel.
        bias: bool = True,  # Apply additive bias before the activation function?
        activation: str = "linear",  # Activation function: 'relu', 'lrelu', etc.
        up: int = 1,  # Integer upsampling factor.
        down: int = 1,  # Integer downsampling factor.
        resample_filter: List[int] = [1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
        conv_clamp: Optional[int] = None,  # Clamp the output to +-X, None = disable clamping.
        channels_last: bool = False,  # Expect the input to have memory_format=channels_last?
        trainable: bool = True,  # Update the weights of this layer during training?
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = (
            torch.channels_last if channels_last else torch.contiguous_format
        )
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(
            memory_format=memory_format
        )
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer("weight", weight)
            if bias is not None:
                self.register_buffer("bias", bias)
            else:
                self.bias = None

    def forward(self, x: torch.Tensor, gain: float = 1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = self.up == 1  # slightly faster
        x = conv2d_resample.conv2d_resample(
            x=x,
            w=w.to(x.dtype),
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=self.padding,
            flip_weight=flip_weight,
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x



# ----------------------------------------------------------------------------
@dataclass
class DiscriminatorBlockConfig:
    activation: str = "lrelu"  # Activation function: 'relu', 'lrelu', etc.
    resample_filter: List[int] = field(
        default_factory=[1, 3, 3, 1].copy
    )  # Low-pass filter to apply when resampling activations.
    fp16_channels_last: bool = False  # Use channels-last memory format with FP16?
    freeze_layers: int = 0  # Freeze-D: Number of layers to freeze.


class DiscriminatorBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,  # Number of input channels, 0 = first block.
        tmp_channels: int,  # Number of intermediate channels.
        out_channels: int,  # Number of output channels.
        resolution: int,  # Resolution of this block.
        img_channels: int,  # Number of input color channels.
        first_layer_idx: int,  # Index of the first layer.
        use_fp16: bool = False,  # Use FP16 for this block?
        architecture: str = "resnet",  # Architecture: 'orig', 'skip', 'resnet'.
        conv_clamp: Optional[
            int
        ] = None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        config: DiscriminatorBlockConfig = DiscriminatorBlockConfig(),
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ["orig", "skip", "resnet"]
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = use_fp16 and config.fp16_channels_last
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(config.resample_filter))

        self.num_layers = 0

        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = layer_idx >= config.freeze_layers
                self.num_layers += 1
                yield trainable

        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == "skip":
            self.fromrgb = Conv2dLayer(
                img_channels,
                tmp_channels,
                kernel_size=1,
                activation=config.activation,
                trainable=next(trainable_iter),
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
            )

        self.conv0 = Conv2dLayer(
            tmp_channels,
            tmp_channels,
            kernel_size=3,
            activation=config.activation,
            trainable=next(trainable_iter),
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
        )

        self.conv1 = Conv2dLayer(
            tmp_channels,
            out_channels,
            kernel_size=3,
            activation=config.activation,
            down=2,
            trainable=next(trainable_iter),
            resample_filter=config.resample_filter,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
        )

        if architecture == "resnet":
            self.skip = Conv2dLayer(
                tmp_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                down=2,
                trainable=next(trainable_iter),
                resample_filter=config.resample_filter,
                channels_last=self.channels_last,
            )

    def forward(self, x: torch.Tensor, img: torch.Tensor, force_fp32: bool = True):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = (
            torch.channels_last
            if self.channels_last and not force_fp32
            else torch.contiguous_format
        )

        # Input.
        if x is not None:
            misc.assert_shape(
                x, [None, self.in_channels, self.resolution, self.resolution]
            )
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == "skip":
            misc.assert_shape(
                img, [None, self.img_channels, self.resolution, self.resolution]
            )
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = (
                upfirdn2d.downsample2d(img, self.resample_filter)
                if self.architecture == "skip"
                else None
            )

        # Main layers.
        if self.architecture == "resnet":
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img


# ----------------------------------------------------------------------------


@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size: int, num_channels: int = 1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings():  # as_tensor results are registered as constants
            G = (
                torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N))
                if self.group_size is not None
                else N
            )
        F = self.num_channels
        c = C // F

        y = x.reshape(
            G, -1, F, c, H, W
        )  # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)  # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)  # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()  # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])  # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)  # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)  # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)  # [NCHW]   Append to input as new channels.
        return x


# ----------------------------------------------------------------------------
@dataclass
class DiscriminatorEpilogueConfig:
    resolution: int = 4
    mbstd_group_size: int = 4  # Group size for the minibatch standard deviation layer, None = entire minibatch.
    mbstd_num_channels: int = (
        1  # Number of features for the minibatch standard deviation layer, 0 = disable.
    )
    activation: str = "lrelu"  # Activation function: 'relu', 'lrelu', etc.


class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,  # Number of input channels.
        cmap_dim: int,  # Dimensionality of mapped conditioning label, 0 = no label.
        img_channels: int,  # Number of input color channels.
        architecture: str = "resnet",  # Architecture: 'orig', 'skip', 'resnet'.
        conv_clamp: Optional[
            int
        ] = None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        config: DiscriminatorEpilogueConfig = DiscriminatorEpilogueConfig(),
    ):
        assert architecture in ["orig", "skip", "resnet"]
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = config.resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == "skip":
            self.fromrgb = Conv2dLayer(
                img_channels, in_channels, kernel_size=1, activation=config.activation
            )
        self.mbstd = (
            MinibatchStdLayer(
                group_size=config.mbstd_group_size,
                num_channels=config.mbstd_num_channels,
            )
            if config.mbstd_num_channels > 0
            else None
        )
        self.conv = Conv2dLayer(
            in_channels + config.mbstd_num_channels,
            in_channels,
            kernel_size=3,
            activation=config.activation,
            conv_clamp=conv_clamp,
        )
        self.fc = FullyConnectedLayer(
            in_channels * (config.resolution**2),
            in_channels,
            activation=config.activation,
        )
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(
        self,
        x: torch.Tensor,
        img: torch.Tensor,
        cmap: torch.Tensor,
        force_fp32: bool = False,
    ):
        misc.assert_shape(
            x, [None, self.in_channels, self.resolution, self.resolution]
        )  # [NCHW]
        _ = force_fp32  # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == "skip":
            misc.assert_shape(
                img, [None, self.img_channels, self.resolution, self.resolution]
            )
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x


# ----------------------------------------------------------------------------
@dataclass
class DiscriminatorConfig:
    architecture: str = "resnet"  # Architecture: 'orig', 'skip', 'resnet'.
    channel_base: int = 32768  # Overall multiplier for the number of channels.
    channel_max: int = 512  # Maximum number of channels in any layer.
    num_fp16_res: int = 0  # Use FP16 for the N highest resolutions.
    conv_clamp: Optional[
        int
    ] = None  # Clamp the output of convolution layers to +-X, None = disable clamping.
    cmap_dim: Optional[
        int
    ] = None  # Dimensionality of mapped conditioning label, None = default.
    block: DiscriminatorBlockConfig = (
        DiscriminatorBlockConfig()
    )  # Arguments for DiscriminatorBlock.
    mapping: MappingNetworkConfig = MappingNetworkConfig(
        w_avg_beta=None
    )  # Arguments for MappingNetwork.
    epilogue: DiscriminatorEpilogueConfig = (
        DiscriminatorEpilogueConfig()
    )  # Arguments for DiscriminatorEpilogue.


class Discriminator(torch.nn.Module):
    def __init__(
        self,
        c_dim,  # Conditioning label (C) dimensionality.
        img_resolution,  # Input resolution.
        img_channels,  # Number of input color channels.
        config: DiscriminatorConfig = DiscriminatorConfig(),
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [
            2**i for i in range(self.img_resolution_log2, 2, -1)
        ]
        channels_dict = {
            res: min(config.channel_base // res, config.channel_max)
            for res in self.block_resolutions + [4]
        }
        fp16_resolution = max(
            2 ** (self.img_resolution_log2 + 1 - config.num_fp16_res), 8
        )

        cmap_dim = config.cmap_dim
        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = res >= fp16_resolution
            block = DiscriminatorBlock(
                in_channels,
                tmp_channels,
                out_channels,
                resolution=res,
                img_channels=img_channels,
                first_layer_idx=cur_layer_idx,
                use_fp16=use_fp16,
                architecture=config.architecture,
                conv_clamp=config.conv_clamp,
                config=config.block,
            )
            setattr(self, f"b{res}", block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(
                c_dim=c_dim,
                w_dim=cmap_dim,
                num_ws=None,
                config=config.mapping,
            )
        self.b4 = DiscriminatorEpilogue(
            channels_dict[4],
            cmap_dim=cmap_dim,
            img_channels=img_channels,
            architecture=config.architecture,
            conv_clamp=config.conv_clamp,
            config=config.epilogue,
        )

    def forward(self, img: torch.Tensor, c: torch.Tensor, **block_kwargs):
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f"b{res}")
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x


# ----------------------------------------------------------------------------
