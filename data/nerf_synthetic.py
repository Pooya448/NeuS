from torch.utils.data import Dataset, Subset
import json
import os
from abc import ABC, abstractmethod
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path


### some parts of this code are influenced or directly copied from https://github.com/nerfstudio-project/nerfacc/blob/master/examples/datasets/nerf_synthetic.py
class NeRFSyntheticData(Dataset):

    SPLITS = ["train", "val", "test"]
    SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
    WIDTH = 800
    HEIGHT = 800
    NEAR = 2.0
    FAR = 6.0
    OPENGL = True

    def __init__(
        self,
        subject_id: str,
        split: str,
        data_dir: Path,
        num_rays: int,
        background_color: str,
        near: float,
        far: float,
        device: torch.device,
    ):
        assert split in self.SPLITS
        assert subject_id in self.SCENES

        self.split = split
        self.num_rays = num_rays
        self.background_color = background_color
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        self.training = split == "train" and num_rays is not None
        self.data_dir = data_dir
        self.subject_id = subject_id
        self.device = device

        images, cam2worlds, intrinsic, focal = self._load_data(
            self.data_dir, self.subject_id, self.split
        )

        self.images = images.to(self.device)
        self.cam2worlds = cam2worlds.to(self.device)
        self.intrinsic = intrinsic.to(self.device)
        self.focal = focal.to(self.device)

        # self.generator = torch.Generator(device=device)
        # self.generator.manual_seed(42)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        data = self.fetch_data(idx)
        return data

    def fetch_data(self, idx: int):

        num_rays = self.num_rays

        if self.training:
            image_idx = torch.randint(
                0,
                len(self.images),
                size=(num_rays,),
                device=self.device,
            )
            x = torch.randint(
                0,
                self.WIDTH,
                size=(num_rays,),
                device=self.device,
            )
            y = torch.randint(
                0,
                self.HEIGHT,
                size=(num_rays,),
                device=self.device,
            )
        else:
            image_idx = [idx]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH, device=self.images.device),
                torch.arange(self.HEIGHT, device=self.images.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        rays_o, rays_v, rgba = self.get_rays(x, y, image_idx)
        near, far = self.get_near_far_unit_sphere(rays_o, rays_v)
        rgb, bck_color = self.get_pixels(rgba)

        return {
            "rays_o": rays_o,
            "rays_v": rays_v,
            "near": near,
            "far": far,
            "rgb": rgb,
            "bck_color": bck_color,
        }

    def get_pixels(self, rgba):
        rgb, alpha = rgba.split([3, 1], dim=-1)
        if self.background_color == "white":
            bck_color = torch.ones(3, device=self.device)
        elif self.background_color == "black":
            bck_color = torch.zeros(3, device=self.device)

        pixels = rgb * alpha + bck_color * (1.0 - alpha)
        return pixels, bck_color

    def get_rays(self, x, y, image_idx):
        rgba = self.images[image_idx, y, x] / 255.0  # (num_rays, 4)
        c2w = self.cam2worlds[image_idx]  # (num_rays, 4, 4)

        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.intrinsic[0, 2] + 0.5) / self.intrinsic[0, 0],
                    (y - self.intrinsic[1, 2] + 0.5)
                    / self.intrinsic[1, 1]
                    * (-1.0 if self.OPENGL else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(directions, dim=-1, keepdims=True)

        if self.training:
            origins = origins.reshape((self.num_rays, 3))
            viewdirs = viewdirs.reshape((self.num_rays, 3))
            rgba = rgba.reshape((self.num_rays, 4))
        else:
            origins = origins.reshape((self.HEIGHT, self.WIDTH, 3))
            viewdirs = viewdirs.reshape((self.HEIGHT, self.WIDTH, 3))
            rgba = rgba.reshape((self.HEIGHT, self.WIDTH, 4))

        return origins, viewdirs, rgba

    def get_near_far_unit_sphere(self, rays_o, rays_v):
        a = torch.sum(rays_v**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_v, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.51
        far = mid + 1.51
        return near, far

    def _load_data(self, data_dir: Path, subject_id: str, split: str):
        subject_dir = data_dir / subject_id
        metafile = subject_dir / f"transforms_{split}.json"

        with open(metafile, "r") as fp:
            metadata = json.load(fp)

        meta_len = len(metadata["frames"])
        images = []
        cam2worlds = []

        for i in range(meta_len):
            frame = metadata["frames"][i]
            image_path = subject_dir / (frame["file_path"] + ".png")
            image = imageio.imread(image_path)
            images.append(torch.from_numpy(image).float())
            cam2worlds.append(torch.tensor(frame["transform_matrix"]).float())

        images_tensor = torch.stack(images, dim=0)
        cam2worlds_tensor = torch.stack(cam2worlds, dim=0)
        h, w = images_tensor.shape[1:3]
        angle = metadata["camera_angle_x"]
        focal = torch.tensor(0.5 * w / np.tan(0.5 * angle)).float()
        K_tensor = torch.tensor(
            [[focal, 0, 0.5 * w], [0, focal, 0.5 * h], [0, 0, 1]], dtype=torch.float32
        )

        return images_tensor, cam2worlds_tensor, K_tensor, focal

    def get_loader(self, batch_size: int):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
        )

    def get_subset_loader(self, batch_size: int, indices):
        sub = Subset(self, indices)
        return torch.utils.data.DataLoader(
            sub,
            batch_size=batch_size,
        )
