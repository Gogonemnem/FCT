import torch

from detectron2.config import configurable
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.layers import ShapeSpec

from timm.models.pvt_v2 import PyramidVisionTransformerStage, PyramidVisionTransformerV2 as timmPyramidVisionTransformerV2
from timm.layers.norm import GroupNorm, GroupNorm1, LayerNorm, LayerNorm2d, LayerNormExp2d, RmsNorm
from functools import partial

def get_norm(norm_layer: str, out_channels: int, **kwargs):
    """
    Args:
        norm_layer (str): the name of the normalization layer
        out_channels (int): the number of channels of the input feature
        kwargs: other parameters for the normalization layers

    Returns:
        nn.Module: the normalization layer
    """
    if norm_layer == "LN":
        return LayerNorm(out_channels, **kwargs)
    else:
        raise NotImplementedError(f"Norm type {norm_layer} is not supported (in this code)")

    

@BACKBONE_REGISTRY.register()
class PyramidVisionTransformerV2(timmPyramidVisionTransformerV2, Backbone):
    
    @configurable
    def __init__(
            self,
            *,
            in_chans=3,
            num_classes=None,
            global_pool='avg',
            depths=(3, 4, 6, 3),
            embed_dims=(64, 128, 256, 512),
            num_heads=(1, 2, 4, 8),
            sr_ratios=(8, 4, 2, 1),
            mlp_ratios=(8., 8., 4., 4.),
            qkv_bias=True,
            linear=False,
            drop_rate=0.,
            proj_drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer="LayerNorm",
            out_features=None,
            freeze_at=0,
            only_train_norm=False,
            
    ):
        norm_layer = partial(get_norm, norm_layer)

        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes if num_classes is not None else 0,
            global_pool=global_pool,
            depths=depths,
            embed_dims=embed_dims,
            num_heads=num_heads,
            sr_ratios=sr_ratios,
            mlp_ratios=mlp_ratios,
            qkv_bias=qkv_bias,
            linear=linear,
            drop_rate=drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer
            )
        
        self._out_feature_strides = {"patch_embed": 4}
        self._out_feature_channels = {"patch_embed": embed_dims[0]}
        
        if out_features is not None:
            # Avoid keeping unused layers in this module. They consume extra memory
            # and may cause allreduce to fail
            num_stages = max(
                [{"pvt2": 1, "pvt3": 2, "pvt4": 3, "pvt5": 4}.get(f, 0) for f in out_features]
            )
            self.stages = self.stages[:num_stages]
        self.stage_names = tuple(["pvt" + str(i + 2) for i in range(num_stages)])
        self._out_feature_strides.update({name: 2 ** (i + 2) for i, name in enumerate(self.stage_names)})
        self._out_feature_channels.update({name: embed_dims[i] for i, name in enumerate(self.stage_names)})
        
        if out_features is None:
            if num_classes is not None:
                out_features = ["linear"]
            else:
                out_features = ["pvt" + str(num_stages + 1)]
        self._out_features = out_features
        assert len(self._out_features)
        
        # children = [x[0] for x in self.named_children()]
        # for out_feature in self._out_features:
        #     assert out_feature in children, "Available children: {}".format(", ".join(children))
        # assert len(self._out_features)
        self._freeze_stages(freeze_at, only_train_norm)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "in_chans": input_shape.channels,
            "global_pool": cfg.MODEL.PVT.GLOBAL_POOL,
            "depths": cfg.MODEL.PVT.DEPTHS,
            "embed_dims": cfg.MODEL.PVT.EMBED_DIMS,
            "num_heads": cfg.MODEL.PVT.NUM_HEADS,
            "sr_ratios": cfg.MODEL.PVT.SR_RATIOS,
            "mlp_ratios": cfg.MODEL.PVT.MLP_RATIOS,
            "qkv_bias": cfg.MODEL.PVT.QKV_BIAS,
            "linear": cfg.MODEL.PVT.LINEAR,
            "drop_rate": cfg.MODEL.PVT.DROP_RATE,
            "attn_drop_rate": cfg.MODEL.PVT.ATTN_DROP_RATE,
            "drop_path_rate": cfg.MODEL.PVT.DROP_PATH_RATE,
            "proj_drop_rate": cfg.MODEL.PVT.PROJ_DROP_RATE,
            "norm_layer": cfg.MODEL.PVT.NORM_LAYER,
            "out_features": cfg.MODEL.PVT.OUT_FEATURES,
            "only_train_norm": cfg.MODEL.BACKBONE.ONLY_TRAIN_NORM,
            "freeze_at": cfg.MODEL.BACKBONE.FREEZE_AT,
        }


    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"PVTv2 takes an input of shape (N, C, H, W). Got {x.shape} instead!"
    
        outputs = {}

        x = self.patch_embed(x)
        if "patch_embed" in self._out_features:
            outputs["patch_embed"] = x
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        
        if self.num_classes:
            if "linear" in self._out_features:
                outputs["linear"] = self.forward_head(x)
        return outputs
    
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
    
    def _freeze_stages(self, freeze_at=0, only_train_norm=False):
        if freeze_at >= 1:
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        
        module_names = ["downsample", "blocks", "norm"]
        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                for module_name in module_names:
                    module = getattr(stage, module_name)
                    if module is None:
                        continue

                    for param in module.parameters():
                        param.requires_grad = False

                    if not only_train_norm:
                        continue

                    for name, param in module.named_parameters():
                        if 'norm' in name:
                            param.requires_grad = True

