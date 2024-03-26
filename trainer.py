import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from lpips import LPIPS
from pathlib import Path
from models.RGBNet import RGBNet
from models.SDFNet import SDFNet
from models.NeRF import NeRF
from models.VarNet import VarNet
from data.nerf_synthetic import NeRFSyntheticData
from models.Render import OccGridRenderer
from lpips import LPIPS
import trimesh
from models.Renderer import NeuSRenderer


class NeuSTrainer(object):

    def __init__(
        self,
        subject: str,
        run_folder: Path,
        run: wandb.run,
        config: dict,
        device: torch.device,
        use_checkpoint: bool = False,
    ):
        ### Save arguments & configs
        self.config = config
        self.use_checkpoint = use_checkpoint
        self.run_folder = run_folder
        self.run = run
        self.subject = subject
        self.model_list = []
        self.trainable_params = []
        self.device = device
        self.iter_step = 0
        self.val_idx = [63]

        self.train_dataset = self.create_datasets(self.config, self.subject, "train")
        self.test_dataset = self.create_datasets(self.config, self.subject, "test")

        ### Save experiment configs
        self.base_run_dir = Path(self.config["experiment"]["base_run_dir"])

        ### Save training configs
        self.end_iter = self.config["train"]["end_iter"]
        self.save_freq = self.config["train"]["save_freq"]
        self.report_freq = self.config["train"]["report_freq"]
        self.val_freq = self.config["train"]["val_freq"]
        self.val_mesh_freq = self.config["train"]["val_mesh_freq"]
        self.batch_size = self.config["train"]["batch_size"]
        self.validate_resolution_level = self.config["train"][
            "validate_resolution_level"
        ]
        self.learning_rate = float(self.config["train"]["learning_rate"])
        self.learning_rate_alpha = self.config["train"]["learning_rate_alpha"]
        self.use_white_bkgd = self.config["train"]["use_white_bkgd"]
        self.warm_up_end = self.config["train"]["warm_up_end"]
        self.anneal_end = self.config["train"]["anneal_end"]

        #### Save loss configs
        self.igr_weight = self.config["train"]["igr_weight"]
        self.mask_weight = self.config["train"]["mask_weight"]

        #### Create networks
        self.nerf = NeRF(**self.config["model"]["nerf"]).to(self.device)
        self.sdf_net = SDFNet(**self.config["model"]["sdf_net"]).to(self.device)
        self.var_net = VarNet(**self.config["model"]["var_net"]).to(self.device)
        self.rgb_net = RGBNet(**self.config["model"]["rgb_net"]).to(self.device)

        ### Compile models with JIT -> Breaks weight normalization
        # self.nerf = torch.compile(self.nerf)
        # self.rgb_net = torch.compile(self.rgb_net)
        # self.sdf_net = torch.compile(self.sdf_net)
        # self.var_net = torch.compile(self.var_net)

        ### Define optimizers and trainable parameters
        self.trainable_params += list(self.nerf.parameters())
        self.trainable_params += list(self.sdf_net.parameters())
        self.trainable_params += list(self.var_net.parameters())
        self.trainable_params += list(self.rgb_net.parameters())

        self.optimizer = torch.optim.Adam(self.trainable_params, lr=self.learning_rate)

        ### Create renderer
        # self.renderer = OccGridRenderer(
        #     sdf_field=self.sdf_net,
        #     variance_field=self.var_net,
        #     rgb_field=self.rgb_net,
        #     device=self.device,
        # )  #! Implement

        self.renderer = NeuSRenderer(
            nerf=self.nerf,
            sdf_network=self.sdf_net,
            deviation_network=self.var_net,
            color_network=self.rgb_net,
            **self.config["model"]["renderer"],
        )

        if self.use_checkpoint:
            self.load_latest_checkpoint(self.run_folder)

        if run is not None:
            wandb.watch(
                (self.nerf, self.rgb_net, self.sdf_net, self.var_net),
                log="all",
                log_freq=self.val_freq,
                log_graph=True,
            )

    def set_train(self):
        self.nerf.train()
        self.rgb_net.train()
        self.sdf_net.train()
        self.var_net.train()

    def set_eval(self):
        self.nerf.eval()
        self.rgb_net.eval()
        self.sdf_net.eval()
        self.var_net.eval()

    def test(self, mesh_res=128, threshold=0.0) -> None:
        test_idx = np.random.randint(0, len(self.test_dataset), 10)
        print(f"Testing for {len(test_idx)} views.")
        test_loader = self.test_dataset.get_subset_loader(
            batch_size=self.batch_size, indices=test_idx
        )
        self.test_val(loader=test_loader, mode="test")
        print("Extracting mesh...")
        self.extract_mesh(resolution=mesh_res, threshold=threshold)
        print("Testing complete.")

    def train(self) -> None:
        self.update_learning_rate()
        res_iter = self.end_iter - self.iter_step
        epoch_len = len(self.train_dataset)
        epochs = res_iter // epoch_len

        print(f"Training for {epochs} epochs, with {epoch_len} iterations per epoch.")

        train_loader = self.train_dataset.get_loader(batch_size=self.batch_size)
        test_loader = self.test_dataset.get_subset_loader(
            batch_size=self.batch_size, indices=self.val_idx
        )

        for _ in range(epochs):

            for batch in train_loader:

                ### Get batch data
                rays_o = batch.get("rays_o").squeeze()
                rays_v = batch.get("rays_v").squeeze()
                gt_rgb = batch.get("rgb").squeeze()
                bck_color = batch.get("bck_color").squeeze()
                near = batch.get("near").squeeze()[..., None]
                far = batch.get("far").squeeze()[..., None]

                ###! RENDER -> original neus for now
                render_out = self.renderer.render(
                    rays_o,
                    rays_v,
                    near,
                    far,
                    background_rgb=bck_color,
                    cos_anneal_ratio=self.get_cos_anneal_ratio(),
                )

                render_rgb = render_out["color_fine"]
                s_val = render_out["s_val"]
                cdf_fine = render_out["cdf_fine"]
                gradient_error = render_out["gradient_error"]
                weight_max = render_out["weight_max"]
                weight_sum = render_out["weight_sum"]

                # render_rgb = self.renderer(
                #     rays_o=rays_o,
                #     rays_v=rays_v,
                #     nears=near,
                #     fars=far,
                #     cos_anneal_ratio=self.get_cos_anneal_ratio(),
                #     bck_color=bck_color,
                # )

                ### Compute loss
                color_loss = F.l1_loss(render_rgb, gt_rgb, reduction="mean")
                eikonal_loss = gradient_error

                psnr = 20.0 * torch.log10(
                    1.0
                    / (((render_rgb - gt_rgb) ** 2).sum() / (render_rgb.numel())).sqrt()
                )

                # loss = color_loss + self.igr_weight * eikonal_loss
                loss = color_loss + self.igr_weight * eikonal_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                ####Log to wandb #! Add more metrics

                self.update_learning_rate()
                self.iter_step += 1

                if self.iter_step % self.save_freq == 0:
                    self.save_checkpoint()

                if self.iter_step % self.report_freq == 0:
                    print(
                        f"iter:[{self.iter_step}/{res_iter}], PSNR: {psnr:.3f}, Loss: {loss:.6f}"
                    )
                    wandb.log(
                        data={
                            "RGB Loss": color_loss.item(),
                            "Eikonal Loss": eikonal_loss.item(),
                            "Total Loss": loss.item(),
                            "PSNR": psnr.item(),
                        },
                        step=self.iter_step,
                    )

                if self.iter_step % self.val_freq == 0:
                    self.test_val(test_loader, mode="val")

                if self.iter_step % self.val_mesh_freq == 0:
                    self.extract_mesh()

    def load_latest_checkpoint(self, run_folder: Path) -> None:

        def extract_epoch(filename):
            import re

            match = re.search(r"model_epoch-(\d+)", filename.stem)
            if match:
                return int(match.group(1))
            else:
                return -1  # Return -1 if the pattern doesn't match

        models = [x for x in run_folder.glob("model_epoch-*.pth")]
        models.sort(key=extract_epoch)
        latest_model = models[-1]
        checkpoint = torch.load(latest_model)
        self.nerf.load_state_dict(checkpoint["nerf"])
        self.rgb_net.load_state_dict(checkpoint["rgb_net"])
        self.sdf_net.load_state_dict(checkpoint["sdf_net"])
        self.var_net.load_state_dict(checkpoint["var_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.iter_step = checkpoint["iter_step"]

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (
                self.end_iter - self.warm_up_end
            )
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (
                1 - alpha
            ) + alpha

        for g in self.optimizer.param_groups:
            g["lr"] = self.learning_rate * learning_factor

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def extract_mesh(self, resolution=128, threshold=0.0):
        object_bbox_min = torch.tensor([-1.51, -1.51, -1.51], device=self.device)
        object_bbox_max = torch.tensor([1.51, 1.51, 1.51], device=self.device)

        vertices, triangles = self.renderer.extract_geometry(
            object_bbox_min, object_bbox_max, resolution=resolution, threshold=threshold
        )
        mesh = trimesh.Trimesh(vertices, triangles)
        save_folder = self.run_folder / "meshes"
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = save_folder / f"mesh_{self.iter_step}_{threshold}_{resolution}.obj"
        mesh.export(save_path)

        if self.run is not None:
            wandb.log(
                {
                    "val_mesh": [
                        wandb.Object3D(open(save_path)),
                    ]
                }
            )
        return

    def test_val(self, loader, mode: str):

        self.set_eval()

        for m, batch in enumerate(loader):
            rays_o = batch.get("rays_o").squeeze()
            rays_v = batch.get("rays_v").squeeze()
            near = batch.get("near").squeeze()[..., None]
            far = batch.get("far").squeeze()[..., None]
            bck_color = batch.get("bck_color").squeeze()

            H, W = rays_o.shape[:2]

            rays_o = rays_o.reshape(-1, 3)
            rays_v = rays_v.reshape(-1, 3)
            near = near.reshape(-1, 1)
            far = far.reshape(-1, 1)

            rgb_outs = []
            chunk_size = (800 * 800) // 1000
            n_chunks = (H * W) // chunk_size
            for i in range(n_chunks):
                start = i * chunk_size
                end = (i + 1) * chunk_size
                render_out = self.renderer.render(
                    rays_o[start:end],
                    rays_v[start:end],
                    near[start:end],
                    far[start:end],
                    cos_anneal_ratio=self.get_cos_anneal_ratio(),
                    background_rgb=bck_color,
                )
                rgb_outs.append(render_out["color_fine"].detach().cpu().numpy())

            if len(rgb_outs) == n_chunks:
                out_rgb = np.concatenate(rgb_outs, axis=0)
                out_rgb = out_rgb.reshape(H, W, 3)
                out_rgb = np.clip(out_rgb, 0, 1)
                out_rgb = (out_rgb * 255).astype(np.uint8)

                if mode == "val":
                    fp = "val_images"
                elif mode == "test":
                    fp = "test_images"

                save_folder = self.run_folder / fp
                save_folder.mkdir(parents=True, exist_ok=True)
                save_path = save_folder / f"{mode}_{self.iter_step}_{m}.png"

                import imageio

                imageio.imwrite(save_path, out_rgb)

                if self.run is not None:
                    im = wandb.Image(out_rgb, caption=f"{mode} RGB Output")
                    wandb.log({f"{mode}_image_{i}": im}, step=self.iter_step)

        if mode == "val":
            self.set_train()
        return

    def create_datasets(self, config: dict, subject: str, split: str) -> Dataset:

        dataset = NeRFSyntheticData(
            subject_id=subject,
            split=split,
            data_dir=Path(config["dataset"]["data_dir"]),
            num_rays=config["dataset"]["num_rays"],
            background_color=config["dataset"]["background_color"],
            near=config["dataset"]["near"],
            far=config["dataset"]["far"],
            device=self.device,
        )

        return dataset

    def save_checkpoint(self) -> None:
        save_path = self.run_folder / f"model_epoch-{self.iter_step}.pth"
        torch.save(
            {
                "nerf": self.nerf.state_dict(),
                "rgb_net": self.rgb_net.state_dict(),
                "sdf_net": self.sdf_net.state_dict(),
                "var_net": self.var_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "iter_step": self.iter_step,
            },
            save_path,
        )
        if self.run is not None:
            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(save_path)
            self.run.log_artifact(artifact)
