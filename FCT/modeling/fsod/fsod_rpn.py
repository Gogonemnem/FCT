from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator.rpn import RPN
@PROPOSAL_GENERATOR_REGISTRY.register()
class FsodRPN(RPN):
    # def forward(
    #     self, images, query_features, support_features, gt_instances=None
    # ):
    #     # average pooling on support features
    #     support_features_pool = support_features.mean(0, True).mean(dim=[2, 3], keepdim=True) 
    #     rpn_features = {key: F.conv2d(query_features[key], support_features_pool.permute(1,0,2,3), groups=query_features[key].shape[1]) for key in query_features.keys()} # attention map for attention-style rpn
    #     return super().forward(images, rpn_features, gt_instances)
    pass