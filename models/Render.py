import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from nerfacc.estimators.occ_grid import OccGridEstimator
from nerfacc.volrend import (
    accumulate_along_rays_,
    render_weight_from_density,
    rendering,
)


class OccGridRenderer(nn.Module):
    def __init__(self, rgb_field, sdf_field, variance_field, device):
        super(OccGridRenderer, self).__init__()
        self.device = device
        self.rgb_net = rgb_field
        self.sdf_net = sdf_field
        self.var_net = variance_field

        self.aabb = torch.Tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5])
        self.estimator = OccGridEstimator(
            roi_aabb=self.aabb, resolution=256, levels=1
        ).to(self.device)

    def sample_and_render(
        self,
        rays_o: torch.Tensor,
        rays_v: torch.Tensor,
        nears: torch.Tensor,
        fars: torch.Tensor,
        cos_anneal_ratio: float,
        bck_color: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = rays_o.shape[0]
        n_samples = rays_o.shape[0]
        rays_o = rays_o.reshape(-1, 3)
        rays_v = rays_v.reshape(-1, 3)

        def alpha_fn(
            t_starts: torch.Tensor,
            t_ends: torch.Tensor,
            ray_indices: torch.Tensor,
        ) -> torch.Tensor:
            dists = t_ends - t_starts
            dists = torch.sqrt((dists**2).sum(-1))
            t_o = rays_o[ray_indices]
            t_v = rays_v[ray_indices]

            pts = t_o + t_v * (t_starts + t_ends) / 2.0
            dirs = t_v.expand(pts.shape)

            pts = pts.reshape(-1, 3)
            dirs = dirs.reshape(-1, 3)

            sdf_vals, feats = self.sdf_net(pts, dirs)
            grads = self.sdf_net.gradient_sdf(pts).squeeze()
            rgb = self.rgb_net(pts, grads, dirs, feats)
            inv_s = self.var_net(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(batch_size * n_samples, 1)
            true_cos = (dirs * grads).sum(-1, keepdim=True)
            iter_cos = -(
                F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio)
                + F.relu(-true_cos) * cos_anneal_ratio
            )  # always non-positive
            next_sdf = sdf_vals + iter_cos * dists.reshape(-1, 1) * 0.5
            prev_sdf = sdf_vals - iter_cos * dists.reshape(-1, 1) * 0.5

            next_cdf = F.sigmoid(next_sdf)
            prev_cdf = F.sigmoid(prev_sdf)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha = (
                ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)
            )

            return alpha

        def rgb_alpha_fn(
            t_starts: torch.Tensor,
            t_ends: torch.Tensor,
            ray_indices: torch.Tensor,
        ):
            dists = t_ends - t_starts
            t_o = rays_o[ray_indices]
            t_v = rays_v[ray_indices]

            pts = t_o + t_v * (t_starts + t_ends) / 2.0
            dirs = t_v.expand(pts.shape)

            pts = pts.reshape(-1, 3)
            dirs = dirs.reshape(-1, 3)

            sdf_vals, feats = self.sdf_net(pts, dirs)
            grads = self.sdf_net.gradient_sdf(pts).squeeze()
            rgb = self.rgb_net(pts, grads, dirs, feats)
            inv_s = self.var_net(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(batch_size * n_samples, 1)
            true_cos = (dirs * grads).sum(-1, keepdim=True)
            iter_cos = -(
                F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio)
                + F.relu(-true_cos) * cos_anneal_ratio
            )  # always non-positive
            next_sdf = sdf_vals + iter_cos * dists.reshape(-1, 1) * 0.5
            prev_sdf = sdf_vals - iter_cos * dists.reshape(-1, 1) * 0.5

            next_cdf = F.sigmoid(next_sdf)
            prev_cdf = F.sigmoid(prev_sdf)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha = (
                ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)
            )
            return rgb, alpha

        ray_indices, t_starts, t_ends = self.estimator.sampling(
            rays_o,
            rays_v,
            alpha_fn=alpha_fn,
            # t_min=nears,
            # t_max=fars,
        )
        rgb, opacity, depth, extras = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=ray_indices.shape[0],
            rgb_alpha_fn=rgb_alpha_fn,
            render_bkgd=bck_color,
        )
        return rgb

    def forward(
        self,
        rays_o: torch.Tensor,
        rays_v: torch.Tensor,
        nears: torch.Tensor,
        fars: torch.Tensor,
        cos_anneal_ratio: float,
        bck_color: torch.Tensor,
    ):
        rgb = self.sample_and_render(
            rays_o,
            rays_v,
            nears,
            fars,
            cos_anneal_ratio,
            bck_color,
        )
        return rgb
