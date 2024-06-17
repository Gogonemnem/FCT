from torch.nn import functional as F

from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelP6P7

from .pvt_v2 import PyramidVisionTransformerV2
from .fsod_pvt_v2 import FsodPyramidVisionTransformerV2


class FsodFPN(FPN):
    def forward(self, x, y):
        x, y = self.bottom_up(x, y)
        x = self._fpn_forward(x)
        y = self._fpn_forward(y)
        return x, y

    def _fpn_forward(self, x):
        results = []
        prev_features = self.lateral_convs[0](x[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = x[features]
                ## Instead of scale factor x2, we use the size instead. Makes the fpn more robust to uneven input sizes
                top_down_features = F.interpolate(prev_features, size=features.shape[-2:], mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            if self.top_block.in_feature in x:
                top_block_in_feature = x[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}
    
@BACKBONE_REGISTRY.register()
def build_retinanet_pvtv2_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = PyramidVisionTransformerV2(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    last_in_feature = in_features[-1]
    in_channels_p6p7 = bottom_up.output_shape()[last_in_feature].channels
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels, in_feature=last_in_feature),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_retinanet_fsod_pvtv2_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = FsodPyramidVisionTransformerV2(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    last_in_feature = in_features[-1]
    in_channels_p6p7 = bottom_up.output_shape()[last_in_feature].channels
    backbone = FsodFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels, in_feature=last_in_feature),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone