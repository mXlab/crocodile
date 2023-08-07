import math
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..torch_utils import misc
from ..torch_utils import persistence
from ..torch_utils.ops import upfirdn2d
from ..torch_utils.ops import bias_act


# ----------------------------------------------------------------------------
@misc.profiled_function
def normalize_2nd_moment(x: torch.Tensor, dim: int = 1, eps: float = 1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


# ----------------------------------------------------------------------------
@misc.profiled_function
def modulated_style_mlp(x: torch.Tensor, weight: torch.Tensor, styles: torch.Tensor):
    batch_size = x.shape[0]
    channel = x.shape[1]
    width = x.shape[2]
    height = x.shape[3]

    w = None
    dcoefs = None

    w = weight.unsqueeze(0)
    w = w * styles.reshape(batch_size, 1, -1)
    dcoefs = (w.square().sum(dim=[2]) + 1e-8).rsqrt()

    x = x.reshape(batch_size, channel, width * height).permute(0, 2, 1)
    x = x * styles.to(x.dtype).reshape(batch_size, 1, -1)
    x = torch.matmul(x, weight.t().to(x.dtype))
    x = x * dcoefs.to(x.dtype).reshape(batch_size, 1, -1)
    x = x.permute(0, 2, 1).reshape(batch_size, -1, width, height)

    return x


# ----------------------------------------------------------------------------
@misc.profiled_function
def modulated_channel_attention(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    w_weight: torch.Tensor,
    u_weight: torch.Tensor,
    proj_weight: Optional[torch.Tensor],
    styles: torch.Tensor,
    num_heads: int,
):
    batch_size = x.shape[0]
    seq_length = x.shape[1]
    hidden_dimension = x.shape[2]

    assert hidden_dimension % num_heads == 0

    depth = hidden_dimension // num_heads

    attention_scale = torch.tensor(depth**-0.5).to(x.dtype)

    layernorm = nn.InstanceNorm1d(seq_length, affine=False)

    styles1 = styles[:, :hidden_dimension]
    styles2 = styles[:, hidden_dimension:]

    x = x * styles1.to(x.dtype).reshape(batch_size, 1, -1)
    x = layernorm(x)

    q = q_weight.unsqueeze(0)
    q = q * styles1.reshape(batch_size, 1, -1)
    q_dcoefs = (q.square().sum(dim=[2]) + 1e-8).rsqrt()

    k = k_weight.unsqueeze(0)
    k = k * styles1.reshape(batch_size, 1, -1)
    k_dcoefs = (k.square().sum(dim=[2]) + 1e-8).rsqrt()

    v = v_weight.unsqueeze(0)
    v = v * styles1.reshape(batch_size, 1, -1)
    v_dcoefs = (v.square().sum(dim=[2]) + 1e-8).rsqrt()

    w = w_weight.unsqueeze(0)
    w = w * styles2.reshape(batch_size, 1, -1)
    w_dcoefs = (w.square().sum(dim=[2]) + 1e-8).rsqrt()

    q_value = torch.matmul(x, q_weight.t().to(x.dtype)) * q_dcoefs.to(x.dtype).reshape(
        batch_size, 1, -1
    )
    q_value = q_value.reshape(batch_size, seq_length, num_heads, depth).permute(
        0, 2, 1, 3
    )
    k_value = torch.matmul(x, k_weight.t().to(x.dtype)) * k_dcoefs.to(x.dtype).reshape(
        batch_size, 1, -1
    )
    k_value = k_value.reshape(batch_size, seq_length, num_heads, depth).permute(
        0, 2, 1, 3
    )
    if proj_weight is not None:
        k_value = torch.matmul(
            k_value.permute(0, 1, 3, 2), proj_weight.t().to(x.dtype)
        ).permute(0, 1, 3, 2)
    v_value = torch.matmul(x, v_weight.t().to(x.dtype))

    v_value = v_value * v_dcoefs.to(x.dtype).reshape(batch_size, 1, -1)

    v_value = v_value * styles2.to(x.dtype).reshape(batch_size, 1, -1)
    skip = v_value
    if proj_weight is not None:
        v_value = torch.matmul(
            v_value.permute(0, 2, 1), proj_weight.t().to(x.dtype)
        ).permute(0, 2, 1)
        v_value = v_value.reshape(batch_size, 256, num_heads, depth).permute(0, 2, 1, 3)

    else:
        v_value = v_value.reshape(batch_size, seq_length, num_heads, depth).permute(
            0, 2, 1, 3
        )

    attn = torch.matmul(q_value, k_value.permute(0, 1, 3, 2)) * attention_scale
    revised_attn = attn

    attn_score = revised_attn.softmax(dim=-1)

    x = (
        torch.matmul(attn_score, v_value)
        .permute(0, 2, 1, 3)
        .reshape(batch_size, seq_length, hidden_dimension)
    )

    x = torch.matmul(x, w_weight.t().to(x.dtype))

    x = x * w_dcoefs.to(x.dtype).reshape(batch_size, 1, -1)

    u = u_weight.unsqueeze(0)
    u = u * styles2.reshape(batch_size, 1, -1)
    u_dcoefs = (u.square().sum(dim=[2]) + 1e-8).rsqrt()

    skip = torch.matmul(skip, u_weight.t().to(x.dtype))
    skip = skip * u_dcoefs.to(x.dtype).reshape(batch_size, 1, -1)

    x = x + skip

    return x


# ----------------------------------------------------------------------------
@dataclass
class FullyConnectedLayerConfig:
    bias: bool = True  # Apply additive bias before the activation function?
    activation: str = "linear"  # Activation function: 'relu', 'lrelu', etc.
    lr_multiplier: float = 1  # Learning rate multiplier.
    bias_init: float = 0  # Initial value for the additive bias.


class FullyConnectedLayer(nn.Module):
    def __init__(
        self,
        in_features: int,  # Number of input features.
        out_features: int,  # Number of output features.
        config: FullyConnectedLayerConfig = FullyConnectedLayerConfig(),
    ):
        super().__init__()
        self.activation = config.activation
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.randn([out_features, in_features]) / config.lr_multiplier
        )
        self.bias = (
            torch.nn.Parameter(torch.full([out_features], np.float32(config.bias_init)))
            if config.bias
            else None
        )
        self.weight_gain = config.lr_multiplier / np.sqrt(in_features)
        self.bias_gain = config.lr_multiplier

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
@dataclass
class MappingNetworkConfig:
    num_layers: int = 8  # Number of mapping layers.
    embed_features: Optional[
        int
    ] = None  # Label embedding dimensionality, None = same as w_dim.
    layer_features: Optional[
        int
    ] = None  # Number of intermediate features in the mapping layers, None = same as w_dim.
    w_avg_beta: float = 0.995  # Decay for tracking the moving average of W during training, None = do not track.
    layer: FullyConnectedLayerConfig = FullyConnectedLayerConfig(
        activation="lrelu", lr_multiplier=0.01
    )


class MappingNetwork(torch.nn.Module):
    def __init__(
        self,
        z_dim: int,  # Input latent (Z) dimensionality, 0 = no latent.
        c_dim: int,  # Conditioning label (C) dimensionality, 0 = no label.
        w_dim: int,  # Intermediate latent (W) dimensionality.
        num_ws: int,  # Number of intermediate latents to output, None = do not broadcast.
        config: MappingNetworkConfig = MappingNetworkConfig(),
    ):
        super().__init__()
        self.z_dim = z_dim
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
            [z_dim + embed_features]
            + [layer_features] * (config.num_layers - 1)
            + [w_dim]
        )

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(config.num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, config=config.layer)
            setattr(self, f"fc{idx}", layer)

        if num_ws is not None and config.w_avg_beta is not None:
            self.register_buffer("w_avg", torch.zeros([w_dim]))

    def forward(
        self,
        z: torch.Tensor,
        c: torch.Tensor,
        truncation_psi: float = 1,
        truncation_cutoff: Optional[float] = None,
        skip_w_avg_update: bool = False,
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

        assert x is not None

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
@dataclass
class EncoderLayerConfig:
    depth: int = 32
    minimum_head: int = 1
    use_noise: bool = True
    channels_last: bool = False
    conv_clamp: Optional[int] = None


class Encoderlayer(nn.Module):
    def __init__(
        self,
        h_dim: int,
        w_dim: int,
        out_dim: int,
        seq_length: int,
        proj_weight: Optional[nn.Parameter] = None,
        config: EncoderLayerConfig = EncoderLayerConfig(),
    ):
        super().__init__()
        self.h_dim = h_dim
        self.num_heads = max(config.minimum_head, h_dim // config.depth)
        self.w_dim = w_dim
        self.out_dim = out_dim
        self.seq_length = seq_length
        self.use_noise = config.use_noise
        self.conv_clamp = config.conv_clamp
        self.affine1 = FullyConnectedLayer(w_dim, h_dim * 2, bias_init=1)

        memory_format = (
            torch.channels_last if config.channels_last else torch.contiguous_format
        )

        self.q_weight = torch.nn.Parameter(
            torch.FloatTensor(h_dim, h_dim)
            .uniform_(-1.0 / math.sqrt(h_dim), 1.0 / math.sqrt(h_dim))
            .to(memory_format=memory_format)
        )
        self.k_weight = torch.nn.Parameter(
            torch.FloatTensor(h_dim, h_dim)
            .uniform_(-1.0 / math.sqrt(h_dim), 1.0 / math.sqrt(h_dim))
            .to(memory_format=memory_format)
        )
        self.v_weight = torch.nn.Parameter(
            torch.FloatTensor(h_dim, h_dim)
            .uniform_(-1.0 / math.sqrt(h_dim), 1.0 / math.sqrt(h_dim))
            .to(memory_format=memory_format)
        )
        self.w_weight = torch.nn.Parameter(
            torch.FloatTensor(out_dim, h_dim)
            .uniform_(-1.0 / math.sqrt(h_dim), 1.0 / math.sqrt(h_dim))
            .to(memory_format=memory_format)
        )

        self.proj_weight = proj_weight

        self.u_weight = torch.nn.Parameter(
            torch.FloatTensor(out_dim, h_dim)
            .uniform_(-1.0 / math.sqrt(h_dim), 1.0 / math.sqrt(h_dim))
            .to(memory_format=memory_format)
        )
        if config.use_noise:
            self.register_buffer("noise_const", torch.randn([self.seq_length, 1]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_dim]))

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        noise_mode: str = "random",
        gain: float = 1,
    ):
        assert noise_mode in ["random", "const", "none"]
        misc.assert_shape(x, [None, self.seq_length, self.h_dim])
        styles1 = self.affine1(w)

        noise = None
        if self.use_noise and noise_mode == "random":
            noise = (
                torch.randn([x.shape[0], self.seq_length, 1], device=x.device)
                * self.noise_strength
            )
        if self.use_noise and noise_mode == "const":
            noise = self.noise_const * self.noise_strength

        x = modulated_channel_attention(
            x=x,
            q_weight=self.q_weight,
            k_weight=self.k_weight,
            v_weight=self.v_weight,
            w_weight=self.w_weight,
            u_weight=self.u_weight,
            proj_weight=self.proj_weight,
            styles=styles1,
            num_heads=self.num_heads,
        )

        if noise is not None:
            x = x.add_(noise)

        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = x + self.bias.to(x.dtype)
        x = F.leaky_relu(x, negative_slope=0.2)
        assert act_clamp is not None
        x = torch.clamp(x, max=act_clamp, min=-act_clamp)
        return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int,
        channels_last: bool = False,
    ):
        super().__init__()
        self.conv_clamp = None
        self.affine = FullyConnectedLayer(
            w_dim, in_channels, FullyConnectedLayerConfig(bias_init=1)
        )
        memory_format = (
            torch.channels_last if channels_last else torch.contiguous_format
        )
        self.weight = torch.nn.Parameter(
            torch.FloatTensor(out_channels, in_channels)
            .uniform_(-1.0 / math.sqrt(in_channels), 1.0 / math.sqrt(in_channels))
            .to(memory_format=memory_format)
        )
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        styles = self.affine(w)
        x = modulated_style_mlp(x=x, weight=self.weight, styles=styles)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x


# ----------------------------------------------------------------------------
@dataclass
class EncoderBlockConfig:
    architecture: str = "skip"
    linformer: bool = False
    fp16_channels_last: bool = False
    resample_filter: List[int] = field(default_factory=[1, 3, 3, 1].copy)
    scale_ratio: int = 2
    encoder: EncoderLayerConfig = EncoderLayerConfig(conv_clamp=256)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        h_dim: int,
        w_dim: int,
        out_dim: int,
        img_resolution: int,
        resolution: int,
        img_channels: int,
        is_first: bool,
        is_last: bool,
        use_fp16: bool = False,
        config: EncoderBlockConfig = EncoderBlockConfig(),
    ):
        super().__init__()
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.out_dim = out_dim
        self.img_resolution = img_resolution
        self.resolution = resolution
        self.img_channels = img_channels
        self.seq_length = resolution * resolution
        self.is_first = is_first
        self.is_last = is_last
        self.architecture = config.architecture
        self.use_fp16 = use_fp16
        self.register_buffer(
            "resample_filter", upfirdn2d.setup_filter(config.resample_filter)
        )
        self.channels_last = use_fp16 and config.fp16_channels_last
        self.num_attention = 0
        self.num_torgb = 0
        self.scale_ratio = config.scale_ratio
        self.proj_weight = None

        memory_format = torch.contiguous_format

        if self.resolution >= 32 and config.linformer:
            self.proj_weight = torch.nn.Parameter(
                torch.FloatTensor(256, self.seq_length)
                .uniform_(
                    -1.0 / math.sqrt(self.seq_length), 1.0 / math.sqrt(self.seq_length)
                )
                .to(memory_format=memory_format)
            )

        if self.is_first and self.resolution == 8:
            self.const = torch.nn.Parameter(torch.randn([self.seq_length, self.h_dim]))

        if self.is_first:
            self.pos_embedding = torch.nn.Parameter(
                torch.zeros(1, self.seq_length, self.h_dim)
            )

        if not self.is_last or out_dim is None:
            self.out_dim = h_dim

        self.enc = Encoderlayer(
            h_dim=self.h_dim,
            w_dim=self.w_dim,
            out_dim=self.out_dim,
            seq_length=self.seq_length,
            proj_weight=self.proj_weight,
        )
        self.num_attention += 1

        if self.is_last and self.architecture == "skip":
            self.torgb = ToRGBLayer(
                self.out_dim,
                self.img_channels,
                w_dim=w_dim,
                channels_last=self.channels_last,
            )
            self.num_torgb += 1

    def forward(
        self,
        x: torch.Tensor,
        img: torch.Tensor,
        ws: torch.Tensor,
        force_fp32: bool = True,
        fused_modconv: Optional[bool] = None,
    ):
        misc.assert_shape(ws, [None, self.num_attention + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = (
            torch.channels_last
            if self.channels_last and not force_fp32
            else torch.contiguous_format
        )
        if fused_modconv is None:
            with misc.suppress_tracer_warnings():  # this value will be treated as a constant
                fused_modconv = (not self.training) and (
                    dtype == torch.float32 or int(x.shape[0]) == 1
                )

        # Input
        if self.is_first and self.resolution == 8:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1])
        else:
            misc.assert_shape(x, [None, self.seq_length, self.h_dim])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers
        if self.is_first:
            x = x + self.pos_embedding

        if self.architecture == "resnet":
            y = self.skip(
                x.permute(0, 2, 1).reshape(
                    ws.shape[0], self.h_dim, self.resolution, self.resolution
                )
            )
            x = self.enc(x, next(w_iter))
            y = y.reshape(ws.shape[0], self.h_dim, self.seq_length)
            x = y.add_(x)
        else:
            x = self.enc(x, next(w_iter)).to(dtype=dtype, memory_format=memory_format)
        # ToRGB
        if self.is_last:
            if img is not None:
                misc.assert_shape(
                    img,
                    [
                        None,
                        self.img_channels,
                        self.resolution // 2,
                        self.resolution // 2,
                    ],
                )
                img = upfirdn2d.upsample2d(img, self.resample_filter)

            if self.architecture == "skip":
                y = self.torgb(
                    x.permute(0, 2, 1).reshape(
                        ws.shape[0], self.out_dim, self.resolution, self.resolution
                    ),
                    next(w_iter),
                    fused_modconv=fused_modconv,
                )
                y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
                img = img.add_(y) if img is not None else y
            # upsample
            if self.resolution != self.img_resolution:
                x = upfirdn2d.upsample2d(
                    x.permute(0, 2, 1).reshape(
                        ws.shape[0], self.out_dim, self.resolution, self.resolution
                    ),
                    self.resample_filter,
                )
                x = x.reshape(
                    ws.shape[0], self.out_dim, self.seq_length * self.scale_ratio**2
                ).permute(0, 2, 1)

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img


# ----------------------------------------------------------------------------
@dataclass
class SynthesisNetworkConfig:
    num_layers: List[int] = field(default_factory=[1, 3, 3].copy)
    num_channels: List[int] = field(default_factory=[256, 64, 16].copy)
    channel_base: int = 8192
    channel_max: int = 256
    num_fp16_res: int = 0
    encoder: EncoderBlockConfig = EncoderBlockConfig()


class SynthesisNetwork(nn.Module):
    def __init__(
        self,
        w_dim: int,
        img_resolution: int,
        img_channels: int,
        config: SynthesisNetworkConfig = SynthesisNetworkConfig(),
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_block = config.num_layers
        self.block_resolutions = [
            2**i for i in range(3, self.img_resolution_log2 + 1)
        ]
        assert len(self.block_resolutions) == len(self.num_block)
        channels_dict = dict(zip(*[self.block_resolutions, config.num_channels]))
        fp16_resolution = max(
            2 ** (self.img_resolution_log2 + 1 - config.num_fp16_res), 8
        )

        self.num_ws = 0
        for i, res in enumerate(self.block_resolutions):
            h_dim = channels_dict[res]
            out_dim = None
            if res != self.img_resolution:
                out_dim = channels_dict[res * 2]
            assert out_dim is not None
            use_fp16 = res >= fp16_resolution
            num_block_res = self.num_block[i]
            for j in range(num_block_res):
                is_first = j == 0
                is_last = j == num_block_res - 1
                block = EncoderBlock(
                    h_dim=h_dim,
                    w_dim=w_dim,
                    out_dim=out_dim,
                    img_resolution=img_resolution,
                    resolution=res,
                    img_channels=img_channels,
                    is_first=is_first,
                    is_last=is_last,
                    use_fp16=use_fp16,
                    config=config.encoder,
                )
                self.num_ws += block.num_attention
                if is_last:
                    self.num_ws += block.num_torgb
                setattr(self, f"b{res}_{j}", block)

    def forward(self, ws: Optional[torch.Tensor] = None):
        block_ws = []
        with torch.autograd.profiler.record_function("split_ws"):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            assert ws is not None
            ws = ws.to(torch.float32)
            w_idx = 0
            for i, res in enumerate(self.block_resolutions):
                num_block_res = self.num_block[i]
                res_ws = []
                for j in range(num_block_res):
                    block = getattr(self, f"b{res}_{j}")
                    res_ws.append(
                        ws.narrow(1, w_idx, block.num_attention + block.num_torgb)
                    )
                    w_idx += block.num_attention
                block_ws.append(res_ws)

        x = img = None
        for i, (res, cur_ws) in enumerate(zip(self.block_resolutions, block_ws)):
            num_block_res = self.num_block[i]
            for j in range(num_block_res):
                block = getattr(self, f"b{res}_{j}")
                x, img = block(x, img, cur_ws[j])

        return img


# ----------------------------------------------------------------------------
@dataclass
class GeneratorConfig:
    z_dim: int = 512
    w_dim: int = 512
    mapping: MappingNetworkConfig = MappingNetworkConfig()
    synthesis: SynthesisNetworkConfig = SynthesisNetworkConfig()


class Generator(nn.Module):
    def __init__(
        self,
        c_dim: int,
        img_resolution: int,
        img_channels: int,
        config: GeneratorConfig = GeneratorConfig(),
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.z_dim = config.z_dim
        self.w_dim = config.w_dim

        self.synthesis = SynthesisNetwork(
            w_dim=config.w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            config=config.synthesis,
        )
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(
            z_dim=config.z_dim,
            c_dim=c_dim,
            w_dim=config.w_dim,
            num_ws=self.num_ws,
            config=config.mapping,
        )

    def forward(
        self,
        z: torch.Tensor,
        c: torch.Tensor,
        truncation_psi: float = 1,
        truncation_cutoff: Optional[float] = None,
    ):
        ws = self.mapping(
            z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff
        )
        output = self.synthesis(ws)
        return output
