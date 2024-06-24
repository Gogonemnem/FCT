# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.box_head import ROI_BOX_HEAD_REGISTRY
from .pvt_v2 import PyramidVisionTransformerStage

from .pvt_v2 import get_norm
from functools import partial


# To get torchscript support, we make the head a subclass of `nn.Module`.
# Therefore, to add new layers in this head class, please make sure they are
# added in the order they will be used in forward().
@ROI_BOX_HEAD_REGISTRY.register()
class PVT4BoxHead(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        depths=(3, 4, 6, 3),
        embed_dims=(64, 128, 256, 512),
        num_heads=(1, 2, 4, 8),
        sr_ratios=(8, 4, 2, 1),
        mlp_ratios=(8., 8., 4., 4.),
        qkv_bias=True,
        linear=False,
        proj_drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer="LN",
    ):
        super().__init__()

        norm_layer = partial(get_norm, norm_layer)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
    
        i = -1 # because we take last stage in PVTv2
        self.stage = PyramidVisionTransformerStage(
                dim=input_shape.channels,
                dim_out=embed_dims[i],
                depth=depths[i],
                downsample=i != 0,
                num_heads=num_heads[i],
                sr_ratio=sr_ratios[i],
                mlp_ratio=mlp_ratios[i],
                linear_attn=linear,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
        self.feature_info = dict(num_chs=embed_dims[i], reduction=4 * 2**i, module=f'stages.{i}')

        self._output_size = ShapeSpec(channels=embed_dims[i], height=input_shape.height // 2, width=input_shape.width // 2)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "depths": cfg.MODEL.PVT.DEPTHS,
            "embed_dims": cfg.MODEL.PVT.EMBED_DIMS,
            "num_heads": cfg.MODEL.PVT.NUM_HEADS,
            "sr_ratios": cfg.MODEL.PVT.SR_RATIOS,
            "mlp_ratios": cfg.MODEL.PVT.MLP_RATIOS,
            "qkv_bias": cfg.MODEL.PVT.QKV_BIAS,
            "linear": cfg.MODEL.PVT.LINEAR,
            "proj_drop_rate": cfg.MODEL.PVT.PROJ_DROP_RATE,
            "attn_drop_rate": cfg.MODEL.PVT.ATTN_DROP_RATE,
            "drop_path_rate": cfg.MODEL.PVT.DROP_PATH_RATE,
            "norm_layer": cfg.MODEL.PVT.NORM_LAYER,
        }

    def forward(self, x):
        x = self.stage(x)
        return x

    @property
    @torch.jit.unused
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return o