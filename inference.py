#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from models.FM.flow_matching import WIPFlow
from data.datamodule import SIS_DataModule

torch.set_float32_matmul_precision('high')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference script for WIPFlow on test dataset")

    # Paths
    parser.add_argument("--ckpt_path", type=str, required=False,
                        default="/home/joty/code/flow_matching/exp/checkpoints/last.ckpt",
                        help="Path to the trained Lightning checkpoint (.ckpt)")
    parser.add_argument('--data_path', type=str, required=False,
                        default='/home/joty/code/Solar_D2P/data',
                        help='Data root path.')
    parser.add_argument('--save_results_dir', type=str, required=False, default="./test_results",
                        help='Directory to save predictions.npy and targets.npy. '
                             'Model may also have its own default; this allows overriding.')
    parser.add_argument("--years_train", nargs="+", type=int, default=[2019, 2020])
    parser.add_argument("--years_val", nargs="+", type=int, default=[2021])
    parser.add_argument("--years_test", nargs="+", type=int, default=[2022])
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--use_possible_starts", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=16)
    # Data lengths
    parser.add_argument('--input_len', type=int, required=False, default=4,
                        help='History window length.')
    parser.add_argument('--pred_len', type=int, required=False, default=12,
                        help='Future prediction horizon.')


    # Hardware / performance
    parser.add_argument('--num_gpus', type=int, required=False, default=1,
                        help='Number of GPUs (0 to force CPU).')
    parser.add_argument('--precision', type=str, required=False,
                        choices=['bf16', 'bf16-mixed', '32', '16-mixed', '64'],
                        default='bf16',
                        help='Precision for inference. "bf16" maps to bf16-mixed on Lightning.')
    parser.add_argument('--num_workers', type=int, required=False, default=8,
                        help='Number of DataLoader workers.')
    parser.add_argument('--limit_test_batches', type=float, required=False, default=1.0,
                        help='Fraction or number of test batches to run (float in (0,1] or int).')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile for model if available (PyTorch 2.x).')


    # Logging (W&B)
    parser.add_argument('--enable_wandb', action='store_true',
                        help='Enable Weights & Biases logging during inference.')
    parser.set_defaults(enable_wandb=False)
    parser.add_argument('--wandb_project_name', type=str, required=False, default='flow_matching',
                        help='W&B project name.')
    parser.add_argument('--wandb_group', type=str, required=False, default=None,
                        help='W&B group.')
    parser.add_argument('--wandb_id', type=str, required=False, default=None,
                        help='W&B run id (resume logging).')
    parser.add_argument('--wandb_offline', action='store_true',
                        help='Run W&B in offline mode.')
    parser.set_defaults(wandb_offline=False)

    # FM/solver compatibility flags
    parser.add_argument('--num_flow_steps', type=int, required=False, default=50,
                        help='Number of flow steps for integration.')
    parser.add_argument('--eps', type=float, required=False, default=0.0,
                        help='Flow start time.')
    # Reproducibility
    parser.add_argument('--seed', type=int, required=False, default=42,
                        help='Random seed for reproducibility.')

    args = parser.parse_args()
    return args


def normalize_precision_arg(precision: str, accelerator: str) -> str:
    # Map user-friendly precision to PL-supported values
    # On CPU, enforce 32 for safety
    if accelerator == 'cpu':
        return '32'

    mapping = {
        'bf16': 'bf16',
        'bf16-mixed': 'bf16-mixed',
        '16-mixed': '16-mixed',
        '32': '32',
        '64': '64'
    }
    return mapping.get(precision, '32')


def build_trainer(args: argparse.Namespace) -> Trainer:
    accelerator = 'gpu' if torch.cuda.is_available() and args.num_gpus > 0 else 'cpu'
    precision = normalize_precision_arg(args.precision, accelerator)

    # Devices
    if accelerator == 'gpu':
        devices = args.num_gpus
        strategy = None
        if args.num_gpus > 1:
            strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
    else:
        devices = 1
        strategy = None

    # Logger
    logger = None
    if args.enable_wandb:
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"
        logger = WandbLogger(
            project=args.wandb_project_name,
            group=args.wandb_group,
            id=args.wandb_id,
        )

    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        callbacks=[lr_monitor_callback],
        log_every_n_steps=50,
        deterministic=True,
        benchmark=False,
        num_sanity_val_steps=0,
        limit_test_batches=args.limit_test_batches,
    )
    return trainer


def load_test_dataloader(args: argparse.Namespace):
    dataset_config = {
        "data_path": args.data_path,
        "years": {
            "train": args.years_train,
            "val": args.years_val,
            "test": args.years_test,
        },
        "input_len": args.input_len,
        "pred_len": args.pred_len,
        "stride": args.stride,
        "use_possible_starts": args.use_possible_starts,
    }

    datamodule = SIS_DataModule(
        dataset=dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    datamodule.setup()
    test_dataloader = datamodule.test_dataloader()
    return test_dataloader


@rank_zero_only
def log_hparams(logger: Optional[WandbLogger], args: argparse.Namespace, extra: Optional[Dict[str, Any]] = None):
    if logger is None:
        return
    hp = {
        "precision": args.precision,
        "num_gpus": args.num_gpus,
        "data_path": args.data_path,
        "input_len": args.input_len,
        "pred_len": args.pred_len,
        "ckpt_path": args.ckpt_path,
        "save_results_dir": args.save_results_dir,
        "limit_test_batches": args.limit_test_batches,
        "seed": args.seed,
        "num_flow_steps": args.num_flow_steps,
        "eps": args.eps,
    }
    if extra:
        hp.update(extra)
    logger.log_hyperparams(hp)


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)

    test_loader = load_test_dataloader(args)

    # Checkpoint
    if not os.path.isfile(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    model: WIPFlow = WIPFlow.load_from_checkpoint(args.ckpt_path, strict=False)

    model.num_flow_steps = args.num_flow_steps
    # Optional compile (PyTorch 2.x)
    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore
        except Exception as e:
            print(f"[warn] torch.compile failed: {e}")

    # Results directory harmonization
    save_dir = None
    if args.save_results_dir:
        save_dir = Path(args.save_results_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Try to set model save paths in a backward-compatible way
    if hasattr(model, "test_results_path") and save_dir is not None:
        model.test_results_path = save_dir.as_posix()
    if hasattr(model, "test_save_dir") and save_dir is not None:
        model.test_save_dir = save_dir

    # Trainer
    trainer = build_trainer(args)

    # Log hyperparams to W&B if enabled
    logger = trainer.logger if isinstance(trainer.logger, WandbLogger) else None
    log_hparams(logger, args, extra={"script": "infer_rectified_flow_v2"})

    # Run test
    trainer.test(model=model, dataloaders=test_loader, ckpt_path=args.ckpt_path)

    # Print final save path if available
    if trainer.global_rank == 0:
        out = None
        if hasattr(model, "test_save_dir") and model.test_save_dir is not None:
            # If test_save_dir is a Path or str-like, print posix path
            try:
                out = Path(model.test_save_dir).as_posix()
            except Exception:
                out = str(model.test_save_dir)
        elif hasattr(model, "test_results_path"):
            out = str(model.test_results_path)
        if out is None and save_dir is not None:
            out = save_dir.as_posix()
        print(f"Inference finished. Results saved to: {out}")


if __name__ == "__main__":
    main()