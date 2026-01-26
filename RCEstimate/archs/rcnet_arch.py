from torch import nn
import torch.nn.functional as F
import torch
from archs.rcnet_util import CNNEncoder, FeatureTransformer, BasicUpdateBlock
from archs.rcnet_util import upsample_corres_with_mask, global_correlation_softmax, local_correlation_with_corres
from archs.transformer_util import feature_add_position

"""
    This implementation is based on UniMatch (https://github.com/autonomousvision/unimatch)
"""
class RCNet(nn.Module):
    def __init__(self,
                 num_scales=1,
                 feature_channels=128,
                 upsample_factor=8,
                 num_head=1,
                 ffn_dim_expansion=4,
                 num_transformer_layers=6,
                 reg_refine=False,  # optional local regression refinement
                 refine_num=-1,
                 ):
        super(RCNet, self).__init__()

        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.upsample_factor = upsample_factor
        self.reg_refine = reg_refine
        self.refine_num = refine_num 

        # CNN
        self.backbone0 = CNNEncoder(input_dim=3, output_dim=feature_channels, num_output_scales=num_scales)
        self.backbone1 = CNNEncoder(input_dim=3, output_dim=feature_channels, num_output_scales=num_scales)  # background

        # Transformer
        self.transformer = FeatureTransformer(num_layers=num_transformer_layers,
                                              d_model=feature_channels,
                                              nhead=num_head,
                                              ffn_dim_expansion=ffn_dim_expansion,
                                              )

        # Convex upsampling similar to RAFT
        # Concat feature0 and low res flow as input
        self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))
            
        if self.reg_refine:
            self.refine_proj = nn.Conv2d(128, 256, 1)
            self.refine = BasicUpdateBlock(corr_channels=(2 * 4 + 1) ** 2,
                                           downsample_factor=upsample_factor,
                                           flow_dim=2,
                                           bilinear_up=False,
                                           )

    def extract_feature(self, img0, img1):
        feature0 = self.backbone0(img0)
        feature1 = self.backbone1(img1) # 

        # reverse: resolution from low to high
        feature0.reverse()
        feature1.reverse()

        return feature0, feature1

    
    def upsample_corres(self, corres, feature, bilinear=False, upsample_factor=8, is_depth=False):
        if bilinear:
            multiplier = 1 if is_depth else upsample_factor
            up_corres = F.interpolate(corres, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * multiplier
        else:
            concat = torch.cat((corres, feature), dim=1)
            mask = self.upsampler(concat)
            up_corres = upsample_corres_with_mask(corres, mask, upsample_factor=self.upsample_factor,
                                            multiple_value=True)
        return up_corres

    def forward(self, inputs,
                attn_type='swin',
                attn_splits_list=[2],
                corr_radius_list=[-1],
                prop_radius_list=[-1],
                ):

        results_dict = {}

        img0 = inputs['input_img']  # [B, 3, H, W]
        img1 = inputs['background']  # [B, 3, H, W]

        # list of features
        feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features

        corres = None

        assert len(attn_splits_list) == len(corr_radius_list) == len(prop_radius_list) == self.num_scales

        

        scale_idx = 0
        feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]
        feature0_ori, feature1_ori = feature0, feature1
        
        upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))

        # attention configs
        attn_splits = attn_splits_list[scale_idx]
        corr_radius = corr_radius_list[scale_idx]
        prop_radius = prop_radius_list[scale_idx]

        # add position to features
        feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)

        # Transformer
        feature0, feature1 = self.transformer(feature0, feature1,
                                                attn_type=attn_type,
                                                attn_num_splits=attn_splits,
                                                )

        corres_pred = global_correlation_softmax(feature0, feature1)[0]
        corres = corres_pred
        corres_list = []
        
        if self.training:
            bilinear_corres = upsample_factor*F.interpolate(corres, scale_factor=upsample_factor, mode='bilinear', align_corners=True)
            w,h = bilinear_corres.shape[-1], bilinear_corres.shape[-2]
            bilinear_corres = bilinear_corres / torch.tensor([w-1, h-1], device=bilinear_corres.device).view(1, 2, 1, 1)
            corres_list.append(bilinear_corres)

        if not self.reg_refine:
            # upsample to the original image resolution
            corres_up = self.upsample_corres(corres, feature0,upsample_factor=upsample_factor) # [B, 2, H, W]

            w,h = corres_up.shape[-1], corres_up.shape[-2]
            corres_up = corres_up / torch.tensor([w-1, h-1], device=corres_up.device).view(1, 2, 1, 1)
            
            corres_list.append(corres_up)
        else:
            assert self.refine_num > 0
            for refine_iter_idx in range(self.refine_num):
                corres = corres.detach()
                correlation = local_correlation_with_corres(
                            feature0_ori,
                            feature1_ori,
                            corres=corres,
                            local_radius=4,
                        )  # [B, (2R+1)^2, H, W]
                proj = self.refine_proj(feature0)

        
                """
                Here we follow the implementation of unimatch to add a local regression refinement. In the original paper, this local regression refinement should be like the recurrent refinement in RAFT, which pass the variable 'net' as the latent hidden state 'h' for GRU.
                However, in the implementation of unimatch, the 'net' seems be recreated in every iteration, instead of being updated.
                I'm not sure if this is a bug or a design choice. Just keep it as the code in unimatch.
                """
                net, inp = torch.chunk(proj, chunks=2, dim=1)

                net = torch.tanh(net)
                inp = torch.relu(inp)

                net, up_mask, residual_corres = self.refine(net, inp, correlation, corres.clone(),
                                                            )
                corres = corres + residual_corres
                if self.training or refine_iter_idx == self.refine_num - 1:
                    corres_up = upsample_corres_with_mask(corres, up_mask, upsample_factor=self.upsample_factor)
                    w,h = corres_up.shape[-1], corres_up.shape[-2]
                    corres_up = corres_up / torch.tensor([w-1, h-1], device=corres_up.device).view(1, 2, 1, 1)
                    corres_list.append(corres_up)
                
                
        results_dict.update({'correspondence': corres_list})

        return results_dict

