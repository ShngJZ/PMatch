import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange

from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .loftr_module import LocalFeatureTransformer

class LoFTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = build_backbone(config)
        self.pos_encoding = PositionEncodingSine(config['coarse']['d_model'])

        self.transformers = LocalFeatureTransformer(config['coarse'], layer_names=config['coarse']['transformers_layer_names'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'], layer_names=config['coarse']['layer_names'])

        self.temperature = 0.1
        self.INF = 1e9

    def format_output_feature(self, image0, feat_c0, feat_c0_fpn, feat_m0, feat_f0, image1, feat_c1, feat_c1_fpn, feat_m1, feat_f1):
        f_q_pyramid = {
            8: torch.cat([feat_c0, feat_c0_fpn], dim=1), 4: feat_m0, 2: feat_f0, 1: image0
        }
        f_s_pyramid = {
            8: torch.cat([feat_c1, feat_c1_fpn], dim=1), 4: feat_m1, 2: feat_f1, 1: image1
        }
        return f_q_pyramid, f_s_pyramid

    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['query'].size(0),
            'hw0_i': data['query'].shape[2:], 'hw1_i': data['support'].shape[2:]
        })

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_m, feats_f = self.backbone(
                torch.cat([data['query'], data['support']], dim=0)
            )
            (feat_c0, feat_c1), (feat_m0, feat_m1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_m.split(data['bs']), feats_f.split(data['bs'])
        else:  # handle different input shapes
            (feat_c0, feat_m0, feat_f0), (feat_c1, feat_m1, feat_f1) = self.backbone(data['query']), self.backbone(data['support'])

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # Preserve FPN Feature
        feat_c0_fpn = feat_c0
        feat_c1_fpn = feat_c1

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'query_mask' in data:
            mask_c0, mask_c1 = data['query_mask'].flatten(-2), data['support_mask'].flatten(-2)

        feat_c0, feat_c1 = self.transformers(feat_c0, feat_c1, mask_c0, mask_c1)
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        f_q_pyramid, f_s_pyramid = self.format_output_feature(data['query'],
                                                              rearrange(feat_c0, 'n (h w) c -> n c h w', h=data['hw0_c'][0], w=data['hw0_c'][1]),
                                                              feat_c0_fpn, feat_m0, feat_f0,
                                                              data['support'],
                                                              rearrange(feat_c1, 'n (h w) c -> n c h w', h=data['hw1_c'][0], w=data['hw1_c'][1]),
                                                              feat_c1_fpn, feat_m1, feat_f1)

        sim_matrix = self.coarse_alignment(feat_c0, feat_c1)
        conf_matrix_sprv = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        return f_q_pyramid, f_s_pyramid, sim_matrix, conf_matrix_sprv

    def coarse_alignment(self, feat_c0, feat_c1, mask_c0=None, mask_c1=None):
        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5, [feat_c0, feat_c1])

        sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
        if mask_c0 is not None:
            sim_matrix.masked_fill_(~(mask_c0[..., None] * mask_c1[:, None]).bool(), -self.INF)
        return sim_matrix

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
