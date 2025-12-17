from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder
from ..modules.norm import LayerNorm32
from ..modules import sparse as sp
from ..modules.sparse.transformer import ModulatedSparseTransformerCross4DBlock
# from .structured_latent_flow import SparseResBlock3d
from .sparse_structure_flow import TimestepEmbedder
from .sparse_elastic_mixin import SparseTransformerElasticMixin



class SparseResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        
        assert not (downsample and upsample), "Cannot downsample and upsample at the same time"

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels, bias=True),
        )
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        self.updown = None
        if self.downsample:
            self.updown = sp.SparseDownsample(2)
        elif self.upsample:
            self.updown = sp.SparseUpsample(2)

    def _updown(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.updown is not None:
            x = self.updown(x)
        return x

    def forward(self, x: sp.SparseTensor, emb: torch.Tensor, num_frames: int) -> sp.SparseTensor:
        x = self._updown(x)
        
        if emb.shape[0] != x.shape[0]:
            emb = emb.view(x.shape[0], -1, emb.shape[-1]).mean(dim=1)
        emb_out = self.emb_layers(emb).type(x.dtype)
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h, num_frames)
        h = h.replace(self.norm2(h.feats)) * (1 + scale) + shift
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h, num_frames)
        h = h + self.skip_connection(x)
        return h
    
class SparseResBlock4d(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
        updown_type: Literal["temporal", "spatio-temporal"] = "temporal",
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        
        assert not (downsample and upsample), "Cannot downsample and upsample at the same time"

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv4d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv4d(self.out_channels, self.out_channels, 3))
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels, bias=True),
        )
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        self.updown = None
        self.updown_t = None
        if self.downsample:
            if updown_type == "spatio-temporal":
                self.updown = sp.SparseDownsample(2)
            self.updown_t = sp.SparseDownsample_t(2)
        elif self.upsample:
            if updown_type == "spatio-temporal":
                self.updown = sp.SparseUpsample(2)
            self.updown_t = sp.SparseUpsample_t(2)

    def _updown(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.downsample:
            if self.updown is not None:
                x = self.updown(x)
            if self.updown_t is not None:
                x = self.updown_t(x)
        elif self.upsample:
            if self.updown_t is not None:
                x = self.updown_t(x)
            if self.updown is not None:
                x = self.updown(x)
        return x

    def forward(self, x: sp.SparseTensor, emb: torch.Tensor, num_frames: int) -> sp.SparseTensor:
        if self.downsample:
            num_frames = num_frames // 2
        x = self._updown(x)
        if emb.shape[0] != x.shape[0]:
            emb = emb.view(x.shape[0], -1, emb.shape[-1]).mean(dim=1)
        emb_out = self.emb_layers(emb).type(x.dtype)
        
        scale, shift = torch.chunk(emb_out, 2, dim=1)

        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h, num_frames)
        h = h.replace(self.norm2(h.feats)) * (1 + scale) + shift
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h, num_frames)
        h = h + self.skip_connection(x)

        return h

class SparseResBlock1d(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        
        assert not (downsample and upsample), "Cannot downsample and upsample at the same time"

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv1d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv1d(self.out_channels, self.out_channels, 3))
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels, bias=True),
        )
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        self.updown_t = None
        if self.downsample:
            self.updown_t = sp.SparseDownsample_t(2)
        elif self.upsample:
            self.updown_t = sp.SparseUpsample_t(2)

    def _updown(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.updown_t is not None:
            x = self.updown_t(x)
        return x

    def forward(self, x: sp.SparseTensor, emb: torch.Tensor, num_frames: int) -> sp.SparseTensor:
        if self.downsample:
            num_frames = num_frames // 2
        x = self._updown(x)
        if emb.shape[0] != x.shape[0]:
            emb = emb.view(x.shape[0], -1, emb.shape[-1]).mean(dim=1)
        emb_out = self.emb_layers(emb).type(x.dtype)
        
        scale, shift = torch.chunk(emb_out, 2, dim=1)

        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h, num_frames)
        h = h.replace(self.norm2(h.feats)) * (1 + scale) + shift
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h, num_frames)
        h = h + self.skip_connection(x)

        return h

class SLatFlow4DModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        num_io_res_blocks: int = 2,
        num_io_res4d_blocks: int = 2,
        io_block_channels: List[int] = None,
        io_block_channels_t: List[int] = None,
        pe_mode: Literal["ape", "rope", "rope_t"] = "ape",
        rope_base: float = 10000.0,
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        use_skip_connection: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        expand_self_attn: bool = False,
        conv_type: Literal["3D", "4D", "3D_4D", "3D_1D", "non"] = "3D",
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.num_io_res_blocks = num_io_res_blocks
        self.num_io_res4d_blocks = num_io_res4d_blocks
        self.io_block_channels = io_block_channels
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.use_skip_connection = use_skip_connection
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.expand_self_attn = expand_self_attn
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.conv_type = conv_type

        if self.io_block_channels is not None:
            assert int(np.log2(patch_size)) == np.log2(patch_size), "Patch size must be a power of 2"
            assert np.log2(patch_size) == len(io_block_channels), "Number of IO ResBlocks must match the number of stages"

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape" or pe_mode == "rope_t":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(in_channels, model_channels if io_block_channels is None else io_block_channels[0])
        
        self.input_blocks = nn.ModuleList([])
        if io_block_channels is not None:
            for chs, next_chs in zip(io_block_channels, io_block_channels[1:] + [model_channels]):
                if self.conv_type == "4D":
                    self.input_blocks.extend([
                        SparseResBlock4d(
                            chs,
                            model_channels,
                            out_channels=chs,
                        )
                        for _ in range(num_io_res_blocks-1)
                    ])
                    self.input_blocks.append(
                        SparseResBlock4d(
                            chs,
                            model_channels,
                            out_channels=next_chs,
                            downsample=True,
                            updown_type='spatio-temporal',
                        )
                    )
                else:
                    self.input_blocks.extend([
                        SparseResBlock3d(
                            chs,
                            model_channels,
                            out_channels=chs,
                        )
                        for _ in range(num_io_res_blocks-1)
                    ])
                    self.input_blocks.append(
                        SparseResBlock3d(
                            chs,
                            model_channels,
                            out_channels=next_chs,
                            downsample=True,
                        )
                    )
            
        
        self.input_blocks_t = nn.ModuleList([])
        if io_block_channels_t is not None: 
            for chs, next_chs in zip(io_block_channels_t, io_block_channels_t[1:] + [model_channels]):
                if self.conv_type == "3D_4D":
                    self.input_blocks_t.extend([
                        SparseResBlock4d(
                            chs,
                            model_channels,
                            out_channels=chs,
                        )
                        for _ in range(num_io_res_blocks-1)
                    ])
                    self.input_blocks_t.append(
                        SparseResBlock4d(
                            chs,
                            model_channels,
                            out_channels=next_chs,
                            downsample=True,
                        )
                    )
                elif self.conv_type == "3D_1D":
                    self.input_blocks_t.extend([
                        SparseResBlock1d(
                            chs,
                            model_channels,
                            out_channels=chs,
                        )
                        for _ in range(num_io_res_blocks-1)
                    ])
                    self.input_blocks_t.append(
                        SparseResBlock1d(
                            chs,
                            model_channels,
                            out_channels=next_chs,
                            downsample=True,
                        )
                    )
                elif self.conv_type == 'non':
                    self.input_blocks_t.append(
                        sp.SparseDownsample_t(2)
                    )
            
        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCross4DBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                use_rope_t=(pe_mode == "rope_t"),
                rope_base=rope_base,
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
                expand_self_attn=expand_self_attn,
            )
            for _ in range(num_blocks)
        ])

        self.out_blocks = nn.ModuleList([])
        if io_block_channels is not None:
            for chs, prev_chs in zip(reversed(io_block_channels), [model_channels] + list(reversed(io_block_channels[1:]))):
                if self.conv_type == "4D":
                    self.out_blocks.append(
                        SparseResBlock4d(
                            prev_chs * 2 if self.use_skip_connection else prev_chs,
                            model_channels,
                            out_channels=chs,
                            upsample=True,
                            updown_type='spatio-temporal',
                        )
                    )
                    self.out_blocks.extend([
                        SparseResBlock4d(
                            chs * 2 if self.use_skip_connection else chs,
                            model_channels,
                            out_channels=chs,
                        )
                        for _ in range(num_io_res_blocks-1)
                    ])
                else:
                    self.out_blocks.append(
                        SparseResBlock3d(
                            prev_chs * 2 if self.use_skip_connection else prev_chs,
                            model_channels,
                            out_channels=chs,
                            upsample=True,
                        )
                    )
                    self.out_blocks.extend([
                        SparseResBlock3d(
                            chs * 2 if self.use_skip_connection else chs,
                            model_channels,
                            out_channels=chs,
                        )
                        for _ in range(num_io_res_blocks-1)
                    ])
                
        self.out_blocks_t = nn.ModuleList([])
        if io_block_channels_t is not None:
            for chs, prev_chs in zip(reversed(io_block_channels_t), [model_channels] + list(reversed(io_block_channels_t[1:]))):
                if self.conv_type == "3D_4D":
                    self.out_blocks_t.append(
                        SparseResBlock4d(
                            prev_chs * 2 if self.use_skip_connection else prev_chs,
                            model_channels,
                            out_channels=chs,
                            upsample=True,
                        )
                    )
                    self.out_blocks_t.extend([
                        SparseResBlock4d(
                            chs * 2 if self.use_skip_connection else chs,
                            model_channels,
                            out_channels=chs,
                        )
                        for _ in range(num_io_res4d_blocks-1)
                    ])
                elif self.conv_type == "3D_1D":
                    self.out_blocks_t.append(
                        SparseResBlock1d(
                            prev_chs * 2 if self.use_skip_connection else prev_chs,
                            model_channels,
                            out_channels=chs,
                            upsample=True,
                        )
                    )
                    self.out_blocks_t.extend([
                        SparseResBlock1d(
                            chs * 2 if self.use_skip_connection else chs,
                            model_channels,
                            out_channels=chs,
                        )
                        for _ in range(num_io_res4d_blocks-1)
                    ])
                elif self.conv_type == 'non':
                    self.out_blocks_t.append(
                        sp.SparseUpsample_t(2)
                    )

        self.out_layer = sp.SparseLinear(model_channels if io_block_channels is None else io_block_channels[0], out_channels)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.input_blocks_t.apply(convert_module_to_f16)
        self.blocks.apply(convert_module_to_f16)
        self.out_blocks.apply(convert_module_to_f16)
        self.out_blocks_t.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.input_blocks_t.apply(convert_module_to_f32)
        self.blocks.apply(convert_module_to_f32)
        self.out_blocks.apply(convert_module_to_f32)
        self.out_blocks_t.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: torch.Tensor, num_frames: int) -> sp.SparseTensor:
        h = self.input_layer(x).type(self.dtype)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        t_emb_t = t_emb.view(t_emb.shape[0]//2, 2, -1).mean(dim=1)
        cond = cond.type(self.dtype)

        skips = []
        # pack with input blocks
        for block in self.input_blocks:
            h = block(h, t_emb, num_frames=num_frames)
            skips.append(h.feats)
        
        skips_t = []
        # pack with input blocks
        for block in self.input_blocks_t:
            if self.conv_type == 'non':
                h = block(h)
            else:
                h = block(h, t_emb, num_frames=num_frames)
            skips_t.append(h.feats)
        
        if self.pe_mode == "ape" or self.pe_mode == "rope_t":
            h = h + self.pos_embedder(h.coords[:, 1:]).type(self.dtype)
        
        cond_t = cond.view(cond.shape[0]//2, -1, cond.shape[-1])
        num_frames_t = num_frames//2
        for block in self.blocks:
            h = block(h, t_emb_t, cond_t, num_frames=num_frames_t)

        # unpack with output blocks            
        for block, skip in zip(self.out_blocks_t, reversed(skips_t)):
            if self.conv_type == 'non':
                h = block(h)
            elif self.use_skip_connection:
                h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb, num_frames=num_frames)
            else:
                h = block(h, t_emb, num_frames=num_frames)

        # unpack with output blocks
        for block, skip in zip(self.out_blocks, reversed(skips)):
            if self.use_skip_connection:
                h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb, num_frames=num_frames)
            else:
                h = block(h, t_emb, num_frames=num_frames)

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h.type(x.dtype))
        return h
    

class ElasticSLatFlow4DModel(SparseTransformerElasticMixin, SLatFlow4DModel):
    """
    SLat Flow Model with elastic memory management.
    Used for training with low VRAM.
    """
    pass
