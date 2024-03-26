import torch

import wandb
import pytz
from datetime import datetime
import argparse
import shutil
import sys

from pathlib import Path

from trainer import NeuSTrainer as Trainer
from configs.config import load_configuration
import torch.multiprocessing as mp


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp.set_start_method("spawn", force=True)
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("high")

    tz = pytz.timezone("America/Vancouver")
    creation_date = datetime.now(tz).strftime("%m-%d_%H-%M")

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="./configs/neus.yaml")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--use_checkpoint", default=False, action="store_true")
    parser.add_argument("--subject", type=str, default="chair")
    parser.add_argument("--run", type=str, default="unnamed")
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()
    config = load_configuration(args.conf)

    run_ = args.run
    run = None
    if args.mode == "train":
        if Path(run_).exists():
            if args.use_checkpoint:
                run_folder = Path(run_)
                # wandb_run_name = "_".join(run_folder.name.split("_")[1:])
                wandb_run_name = "BigSphere-100K"
                print("Run exists. Resuming train on existing run.")
                run = wandb.init(
                    name=wandb_run_name, project="NeuS", config=config, resume=True
                )
            else:
                raise ValueError(
                    "Run already exists. Please provide a new run name. (or use --use_checkpoint to resume existing run)"
                )
        else:
            print("Creating new run.")
            run_name = f"{creation_date}_{args.run}_{args.subject}"
            wandb_run_name = f"{args.run}_{args.subject}"
            run_folder = Path(config["experiment"]["base_run_dir"]) / run_name
            run_folder.mkdir(parents=True, exist_ok=True)
            if args.debug:
                wandb.init(mode="disabled")
            else:
                run = wandb.init(name=wandb_run_name, project="NeuS", config=config)
                run.log_code()

    elif args.mode == "test":
        if Path(run_).exists():
            print("Run exists. Testing on existing run.")
            run_folder = Path(run_)
        else:
            raise ValueError("Run does not exist. Please provide a valid run path.")

    # elif args.mode == "test":
    #     if Path(run_).exists():
    #         pass  ### run exists, resume training
    #     else:

    # if args.mode == "train" and not args.debug:
    #     run = wandb.init(name="BigSphere-100K", project="NeuS", config=config)
    #     run.log_code()
    # else:
    #     wandb.init(mode="disabled")

    # if run_folder.exists() and args.use_checkpoint is False and not args.debug:
    #     overwrite = input("Directory already exists. Do you want to overwrite? (y/n): ")
    #     if overwrite.lower() == "y":
    #         shutil.rmtree(run_folder)
    #     else:
    #         print("Execution failed.")
    #         sys.exit(0)
    # elif args.debug:
    #     shutil.rmtree(run_folder) if run_folder.exists() else None

    # run_folder.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        subject=args.subject,
        run_folder=run_folder,
        run=run if run is not None else None,
        config=config,
        device=device,
        use_checkpoint=args.use_checkpoint or args.mode == "test",
    )

    if args.mode == "train":
        trainer.train()
    elif args.mode == "test":
        trainer.test(mesh_res=256, threshold=0.0)
    elif args.mode == "validate_mesh":
        trainer.validate_mesh(world_space=True, resolution=512, threshold=0.0)
