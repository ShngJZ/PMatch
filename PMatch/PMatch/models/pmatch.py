import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from PMatch.PMatch.loftr import LoFTR, default_cfg
from tools.dataops import get_tuple_transform_ops

def local_correlation(
    feature0,
    feature1,
    local_radius,
    padding_mode="zeros",
    flow = None
):
    b, c, h, w = feature0.size()
    if flow is None:
        # If flow is None, assume feature0 and feature1 are aligned
        coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device="cuda"),
                    torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device="cuda"),
                ))
        coords = torch.stack((coords[1], coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
    else:
        coords = flow.permute(0,2,3,1) # If using flow, sample around flow target.
    r = local_radius
    local_window = torch.meshgrid(
                (
                    torch.linspace(-2*local_radius/h, 2*local_radius/h, 2*r+1, device="cuda"),
                    torch.linspace(-2*local_radius/w, 2*local_radius/w, 2*r+1, device="cuda"),
                ))
    local_window = torch.stack((local_window[1], local_window[0]), dim=-1)[
            None
        ].expand(b, 2*r+1, 2*r+1, 2).reshape(b, (2*r+1)**2, 2)
    coords = (coords[:,:,:,None]+local_window[:,None,None]).reshape(b,h,w*(2*r+1)**2,2)
    window_feature = F.grid_sample(
        feature1, coords, padding_mode=padding_mode, align_corners=False
    )[...,None].reshape(b,c,h,w,(2*r+1)**2)
    corr = torch.einsum("bchw, bchwk -> bkhw", feature0, window_feature)/(c**.5)
    return corr

class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        dw=False,
        kernel_size=5,
        hidden_blocks=3,
        displacement_emb = None,
        displacement_emb_dim = None,
        local_corr_radius = None,
        corr_in_other = None,
        no_support_fm = False,
    ):
        super().__init__()
        self.block1 = self.create_block(
            in_dim, hidden_dim, dw=dw, kernel_size=kernel_size
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        if displacement_emb:
            self.has_displacement_emb = True
            self.disp_emb = nn.Conv2d(2,displacement_emb_dim,1,1,0)
        else:
            self.has_displacement_emb = False
        self.local_corr_radius = local_corr_radius
        self.corr_in_other = corr_in_other
        self.no_support_fm = no_support_fm

    def create_block(
        self,
        in_dim,
        out_dim,
        dw=False,
        kernel_size=5,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert (
                out_dim % in_dim == 0
            ), "outdim must be divisible by indim for depthwise"
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
        )
        norm = nn.BatchNorm2d(out_dim)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)

    def forward(self, x, y, flow):
        """Computes the relative refining displacement in pixels for a given image x,y and a coarse flow-field between them

        Args:
            x ([type]): [description]
            y ([type]): [description]
            flow ([type]): [description]

        Returns:
            [type]: [description]
        """
        b,c,hs,ws = x.shape
        with torch.no_grad():
            x_hat = F.grid_sample(y, flow.permute(0, 2, 3, 1), align_corners=False)
        if self.has_displacement_emb:
            query_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device="cuda"),
                torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device="cuda"),
            )
            )
            query_coords = torch.stack((query_coords[1], query_coords[0]))
            query_coords = query_coords[None].expand(b, 2, hs, ws)
            in_displacement = flow-query_coords
            emb_in_displacement = self.disp_emb(in_displacement)
            if self.local_corr_radius:
                #TODO: should corr have gradient?
                if self.corr_in_other:
                    # Corr in other means take a kxk grid around the predicted coordinate in other image
                    local_corr = local_correlation(x, y, local_radius=self.local_corr_radius, flow=flow)
                else:
                    # Otherwise we use the warp to sample in the first image
                    # This is actually different operations, especially for large viewpoint changes
                    local_corr = local_correlation(x, x_hat, local_radius=self.local_corr_radius,)
                if self.no_support_fm:
                    x_hat = torch.zeros_like(x)
                d = torch.cat((x, x_hat, emb_in_displacement, local_corr), dim=1)
            else:
                d = torch.cat((x, x_hat, emb_in_displacement), dim=1)
        else:
            if self.no_support_fm:
                x_hat = torch.zeros_like(x)
            d = torch.cat((x, x_hat), dim=1)
        d = self.block1(d)
        d = self.hidden_blocks(d)
        d = self.out_conv(d)
        certainty, displacement = d[:, :-2], d[:, -2:]
        return certainty, displacement

class CosKernel(nn.Module):  # similar to softmax kernel
    def __init__(self, T, learn_temperature=False):
        super().__init__()
        self.learn_temperature = learn_temperature
        if self.learn_temperature:
            self.T = nn.Parameter(torch.tensor(T))
        else:
            self.T = T

    def __call__(self, x, y, eps=1e-6):
        c = torch.einsum("bnd,bmd->bnm", x, y) / (
            x.norm(dim=-1)[..., None] * y.norm(dim=-1)[:, None] + eps
        )
        if self.learn_temperature:
            T = self.T.abs() + 0.01
        else:
            T = torch.tensor(self.T, device=c.device)
        K = ((c - 1.0) / T).exp()
        return K

class CAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CAB, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = x  # high, low (old, new)
        x = torch.cat([x1, x2], dim=1)
        x = self.global_pooling(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmod(x)
        x2 = x * x2
        res = x2 + x1
        return res

class RRB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(RRB, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    def forward(self, x):
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        return self.relu(x + res)

class DFN(nn.Module):
    def __init__(
        self,
        internal_dim,
        feat_input_module,
        rrb_d,
        cab,
        rrb_u,
        terminal_module,
    ):
        super().__init__()
        self.internal_dim=internal_dim
        self.feat_input_module = feat_input_module
        self.rrb_d = rrb_d
        self.cab = cab
        self.rrb_u = rrb_u
        self.terminal_module = terminal_module

    def forward(self, embeddings, feats, context):
        feats = self.feat_input_module(feats)
        embeddings = torch.cat([feats, embeddings], dim=1)
        embeddings = self.rrb_d(embeddings)

        context = self.cab([context, embeddings])
        context = self.rrb_u(context)
        preds = self.terminal_module(context)

        pred_coord = preds[:, -2:]
        pred_certainty = preds[:, :-2]
        return pred_coord, pred_certainty

class GP(nn.Module):
    def __init__(
        self,
        gp_dim=64,
        basis="fourier",
    ):
        super().__init__()
        self.pos_conv = torch.nn.Conv2d(2, gp_dim, 1, 1)
        self.basis = basis

    def get_local_cov(self, cov):
        K = self.covar_size
        b, h, w, h, w = cov.shape
        hw = h * w
        cov = F.pad(cov, 4 * (K // 2,))  # pad v_q
        delta = torch.stack(
            torch.meshgrid(
                torch.arange(-(K // 2), K // 2 + 1), torch.arange(-(K // 2), K // 2 + 1)
            ),
            dim=-1,
        )
        positions = torch.stack(
            torch.meshgrid(
                torch.arange(K // 2, h + K // 2), torch.arange(K // 2, w + K // 2)
            ),
            dim=-1,
        )
        neighbours = positions[:, :, None, None, :] + delta[None, :, :]
        points = torch.arange(hw)[:, None].expand(hw, K**2)
        local_cov = cov.reshape(b, hw, h + K - 1, w + K - 1)[
            :,
            points.flatten(),
            neighbours[..., 0].flatten(),
            neighbours[..., 1].flatten(),
        ].reshape(b, h, w, K**2)
        return local_cov

    def reshape(self, x):
        return rearrange(x, "b d h w -> b (h w) d")

    def project_to_basis(self, x):
        if self.basis == "fourier":
            return torch.cos(8 * math.pi * self.pos_conv(x))
        elif self.basis == "linear":
            return self.pos_conv(x)
        else:
            raise ValueError(
                "No other bases other than fourier and linear currently supported in public release"
            )

    def get_pos_enc(self, b, c, h, w, device):
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            )
        )

        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        coarse_embedded_coords = self.project_to_basis(coarse_coords)
        return coarse_embedded_coords

    def forward(self, sim_matrix, b, h, w):
        f = self.get_pos_enc(b, None, h, w, device=sim_matrix.device)
        f = f.view([b, -1, h * w]).permute([0, 2, 1])

        conf_matrix = torch.softmax(sim_matrix, dim=2)
        gp_feats = conf_matrix @ f
        gp_feats = rearrange(gp_feats, "b (h w) d -> b d h w", h=h, w=w)

        return gp_feats


class Encoder(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.resnet = resnet

    def forward(self, x):
        x0 = x
        b, c, h, w = x.shape
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x1 = self.resnet.relu(x)

        x = self.resnet.maxpool(x1)
        x2 = self.resnet.layer1(x)

        x3 = self.resnet.layer2(x2)

        x4 = self.resnet.layer3(x3)

        x5 = self.resnet.layer4(x4)
        feats = {32: x5, 16: x4, 8: x3, 4: x2, 2: x1, 1: x0}
        return feats
    
    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            pass

class Decoder(nn.Module):
    def __init__(
        self, detach=True, scales=["8", "4", "2", "1"]
    ):
        super().__init__()

        # Init Embedding Decoder
        gp_dim, dfn_dim, feat_dim, concatenated_dim = 256, 384, 256, 640
        embedding_decoder = DFN(
            internal_dim=dfn_dim,
            feat_input_module=nn.Conv2d(concatenated_dim, feat_dim, 1, 1),
            rrb_d=RRB(gp_dim + feat_dim, dfn_dim),
            cab=CAB(2 * dfn_dim, dfn_dim),
            rrb_u=RRB(dfn_dim, dfn_dim),
            terminal_module=nn.Conv2d(dfn_dim, 3, 1, 1, 0),
        )

        # Init ConvRefiner
        dw, hidden_blocks, kernel_size, displacement_emb = True, 8, 5, "linear"
        conv_refiner = nn.ModuleDict(
            {
                "8": ConvRefiner(
                    2 * concatenated_dim+64+49,
                    2 * concatenated_dim+64+49,
                    3,
                    kernel_size=kernel_size,
                    dw=dw,
                    hidden_blocks=hidden_blocks,
                    displacement_emb=displacement_emb,
                    displacement_emb_dim=64,
                    local_corr_radius = 3,
                    corr_in_other = True,
                ),
                "4": ConvRefiner(
                    2 * 196+32+25,
                    2 * 196+32+25,
                    3,
                    kernel_size=kernel_size,
                    dw=dw,
                    hidden_blocks=hidden_blocks,
                    displacement_emb=displacement_emb,
                    displacement_emb_dim=32,
                    local_corr_radius = 2,
                    corr_in_other = True,
                ),
                "2": ConvRefiner(
                    2 * 128+16,
                    2 * 128+16,
                    3,
                    kernel_size=kernel_size,
                    dw=dw,
                    hidden_blocks=hidden_blocks,
                    displacement_emb=displacement_emb,
                    displacement_emb_dim=16,
                ),
                "1": ConvRefiner(
                    2 * 3+6,
                    24,
                    3,
                    kernel_size=kernel_size,
                    dw=dw,
                    hidden_blocks=hidden_blocks,
                    displacement_emb=displacement_emb,
                    displacement_emb_dim=6,
                ),
            }
        )
        gp = GP(
            gp_dim=gp_dim
        )

        # Initialize Class Elements
        self.embedding_decoder = embedding_decoder
        self.conv_refiner = conv_refiner
        self.gp = gp
        self.detach = detach
        self.scales = scales

    def upsample_preds(self, flow, certainty, query, support):
        b, c, h, w = query.shape
        flow = flow.permute(0, 3, 1, 2)
        certainty = F.interpolate(
            certainty, size=(h, w), align_corners=False, mode="bilinear"
        )
        flow = F.interpolate(
            flow, size=(h, w), align_corners=False, mode="bilinear"
        )
        delta_certainty, delta_flow = self.conv_refiner["1"](query, support, flow)
        flow = torch.stack(
                (
                    flow[:, 0] + delta_flow[:, 0] / (4 * w),
                    flow[:, 1] + delta_flow[:, 1] / (4 * h),
                ),
                dim=1,
            )
        flow = flow.permute(0, 2, 3, 1)
        certainty = certainty + delta_certainty
        return flow, certainty

    def forward(self, f1, f2, sim_matrix):
        coarse_scales = [8]
        all_scales = self.scales
        sizes = {scale: f1[scale].shape[-2:] for scale in f1}
        h, w = sizes[1]
        b = f1[1].shape[0]
        old_stuff = torch.zeros(
            b, self.embedding_decoder.internal_dim, *sizes[8], device=f1[8].device
        )
        dense_corresps = {}
        dense_certainty = 0.0

        for new_scale in all_scales:
            ins = int(new_scale)
            f1_s, f2_s = f1[ins], f2[ins]

            if ins in coarse_scales:
                bh = f1[ins].shape[2]
                bw = f1[ins].shape[3]
                new_stuff = self.gp(sim_matrix, b=b, h=bh, w=bw)
                dense_flow, dense_certainty = self.embedding_decoder(
                    new_stuff, f1_s, old_stuff
                )

            if new_scale in self.conv_refiner:
                hs, ws = h // ins, w // ins
                delta_certainty, displacement = self.conv_refiner[new_scale](
                    f1_s, f2_s, dense_flow
                )
                dense_flow = torch.stack(
                    (
                        dense_flow[:, 0] + ins * displacement[:, 0] / (4 * w),
                        dense_flow[:, 1] + ins * displacement[:, 1] / (4 * h),
                    ),
                    dim=1,
                )  # multiply with scale
                dense_certainty = (
                    dense_certainty + delta_certainty
                )  # predict both certainty and displacement

            dense_corresps[ins] = {
                "dense_flow": dense_flow,
                "dense_certainty": dense_certainty
            }

            if new_scale != "1":
                dense_flow = F.interpolate(
                    dense_flow,
                    size=sizes[ins // 2],
                    align_corners=False,
                    mode="bilinear",
                )

                dense_certainty = F.interpolate(
                    dense_certainty,
                    size=sizes[ins // 2],
                    align_corners=False,
                    mode="bilinear",
                )
                if self.detach:
                    dense_flow = dense_flow.detach()
                    dense_certainty = dense_certainty.detach()
        return dense_corresps

def kde(x, std = 0.1):
    # use a gaussian kernel to estimate density
    x = torch.from_numpy(x).cuda()
    scores = (-torch.cdist(x,x)**2/(2*std**2)).exp()
    density = scores.sum(dim=-1)
    return density

class PMatch(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        # Init the LoFTR Module
        loftr = LoFTR(config=default_cfg)
        decoder = Decoder()

        self.loftr = loftr
        self.decoder = decoder

        # Init Other Parameters for Testing
        self.factor = 0.0
        self.symmetric = True
        self.sample_thresh = 0.05
        self.pow = 1/3
        self.og_transforms = get_tuple_transform_ops(resize=None, normalize=True)

    def train(self, mode=True):
        self.decoder.train(mode)
        self.loftr.train(mode)

    def forward(self, batch):
        f_q_pyramid, f_s_pyramid, sim_matrix, conf_matrix_sprv = self.loftr(batch)
        dense_corresps = self.decoder(f_q_pyramid, f_s_pyramid, sim_matrix)
        dense_corresps.update({'sim_matrix': sim_matrix, 'conf_matrix_sprv': conf_matrix_sprv})
        return dense_corresps

    def forward_symmetric(self, batch):
        query = torch.clone(batch['query'])
        support = torch.clone(batch['support'])

        batch['query'] = torch.cat([query, support], dim=0).contiguous()
        batch['support'] = torch.cat([support, query], dim=0).contiguous()

        f_q_pyramid, f_s_pyramid, sim_matrix, conf_matrix_sprv = self.loftr(batch)
        dense_corresps = self.decoder(f_q_pyramid, f_s_pyramid, sim_matrix)
        dense_corresps.update({'sim_matrix': sim_matrix, 'conf_matrix_sprv': conf_matrix_sprv})
        return dense_corresps

    # ========================================== #
    @torch.no_grad()
    def match(
            self,
            im1,
            im2,
            do_pred_in_og_res=False
    ):
        finest_scale = 1
        self.eval()

        test_transform = get_tuple_transform_ops(
            resize=(self.h_resized, self.w_resized), normalize=True
        )
        query, support = test_transform((im1, im2))
        batch = {"query": query[None].cuda(), "support": support[None].cuda()}

        if self.symmetric:
            dense_corresps = self.forward_symmetric(batch)
        else:
            dense_corresps = self.forward(batch)

        dense_certainty = dense_corresps[finest_scale]["dense_certainty"]
        low_res_certainty = F.interpolate(
            dense_corresps[8]["dense_certainty"], size=(self.h_resized, self.w_resized), align_corners=False, mode="bilinear"
        )
        cert_clamp = 0
        low_res_certainty = self.factor * low_res_certainty * (low_res_certainty < cert_clamp)
        dense_certainty = dense_certainty - low_res_certainty
        query_to_support = dense_corresps[finest_scale]["dense_flow"].permute(0, 2, 3, 1)

        if do_pred_in_og_res:
            ws, hs = im1.size
            query_to_support, dense_certainty = self.upsample_preds_caller(im1, im2, query_to_support, dense_certainty,
                                                                           hs=hs, ws=ws, symmetric=self.symmetric)
        elif self.upsample_preds:
            hs, ws = 864, 1152
            query_to_support, dense_certainty = self.upsample_preds_caller(im1, im2, query_to_support, dense_certainty,
                                                                           hs=hs, ws=ws, symmetric=self.symmetric)

        warp, dense_certainty = self.wrap_output(dense_certainty, query_to_support, symmetric=self.symmetric)
        return warp, dense_certainty

    def upsample_preds_caller(self, im1, im2, query_to_support, dense_certainty, hs, ws, symmetric):
        test_transform = get_tuple_transform_ops(
            resize=(hs, ws), normalize=True
        )
        query, support = test_transform((im1, im2))
        query, support = query[None].cuda(), support[None].cuda()
        if symmetric:
            query, support = torch.cat((query, support)), torch.cat((support, query))
        query_to_support, dense_certainty = self.decoder.upsample_preds(
            query_to_support,
            dense_certainty,
            query,
            support,
        )
        return query_to_support, dense_certainty

    def wrap_output(self, dense_certainty, query_to_support, symmetric):
        b = 1
        _, hs, ws, _ = query_to_support.shape
        query_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device="cuda"),
                torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device="cuda"),
            )
        )
        query_coords = torch.stack((query_coords[1], query_coords[0]))
        query_coords = query_coords[None].expand(b, 2, hs, ws)
        dense_certainty = dense_certainty.sigmoid()  # logits -> probs
        query_coords = query_coords.permute(0, 2, 3, 1)
        if (query_to_support.abs() > 1).any() and True:
            wrong = (query_to_support.abs() > 1).sum(dim=-1) > 0
            dense_certainty[wrong[:, None]] = 0
        query_to_support = torch.clamp(query_to_support, -1, 1)
        if symmetric:
            qts, stq = query_to_support.chunk(2)
            q_warp = torch.cat((query_coords, qts), dim=-1)
            s_warp = torch.cat((stq, query_coords), dim=-1)
            warp = torch.cat((q_warp, s_warp), dim=2)
            dense_certainty = torch.cat(dense_certainty.chunk(2), dim=3)
        else:
            warp = torch.cat((query_coords, query_to_support), dim=-1)

        warp, dense_certainty = warp[0], dense_certainty[0, 0]
        return warp, dense_certainty

    def sample(
        self,
        dense_matches,
        dense_certainty,
        num=10000,
        samplesize=None
    ):
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            dense_certainty = dense_certainty.clone()
            dense_certainty[dense_certainty > upper_thresh] = 1
        elif "pow" in self.sample_mode:
            dense_certainty = dense_certainty**(self.pow)
        elif "naive" in self.sample_mode:
            dense_certainty = torch.ones_like(dense_certainty)
        matches, certainty = (
            dense_matches.reshape(-1, 4).cpu().numpy(),
            dense_certainty.reshape(-1).cpu().numpy(),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = np.random.choice(
            np.arange(len(matches)),
            size=min(expansion_factor*num, len(certainty)),
            replace=False,
            p=certainty / np.sum(certainty),
        )
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty
        density = kde(good_matches, std=0.1).cpu().numpy()
        p = 1 / (density+1)
        p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        p = p/np.sum(p)
        if samplesize is None:
            samplesize = num
        else:
            samplesize = samplesize
        balanced_samples = np.random.choice(
            np.arange(len(good_matches)),
            size=min(samplesize,len(good_certainty)),
            replace=False,
            p = p,
        )
        return good_matches[balanced_samples], good_certainty[balanced_samples]