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
    parser.add_argument("--run", type=str, default=creation_date)
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()
    config = load_configuration(args.conf)

    if args.mode == "train" and not args.debug:
        run = wandb.init(name="init", project="NeuS", config=config)
        run.log_code()
    else:
        wandb.init(mode="disabled")

    run_folder = (
        Path(config["experiment"]["base_run_dir"]) / f"{args.run}_{args.subject}"
    )

    if run_folder.exists() and args.use_checkpoint is False and not args.debug:
        overwrite = input("Directory already exists. Do you want to overwrite? (y/n): ")
        if overwrite.lower() == "y":
            shutil.rmtree(run_folder)
        else:
            print("Execution failed.")
            sys.exit(0)
    elif args.debug:
        shutil.rmtree(run_folder) if run_folder.exists() else None

    run_folder.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        subject=args.subject,
        run_folder=run_folder,
        run=None,
        config=config,
        device=device,
        use_checkpoint=args.use_checkpoint,
    )

    if args.mode == "train":
        trainer.train()
    elif args.mode == "validate_mesh":
        trainer.validate_mesh(world_space=True, resolution=512, threshold=0.0)
