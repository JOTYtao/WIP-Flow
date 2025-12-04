import argparse
from typing import Optional

import torch
import torchvision.transforms as tvt
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from torch.utils.data import DataLoader
from models.FM.flow_matching import WIPFlow
from models.dataloader_csi import load_data
from data.datamodule import SIS_DataModule
torch.set_float32_matmul_precision('high')


def main(args):

    # W&B logger
    logger = WandbLogger(project=args.wandb_project_name,
                         group=args.wandb_group,
                         id=args.wandb_id)
    logger.log_hyperparams(vars(args))


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
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=50,
        min_delta=1e-5,
        verbose=True
    )
    ckpt_callback = ModelCheckpoint(
        dirpath="/home/joty/code/flow_matching/exp/checkpoints/",
        filename="{epoch}-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_last=True,
        every_n_epochs=1
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    # Trainer
    trainer = Trainer(
        logger=logger,
        max_epochs=args.max_epochs,
        accelerator='gpu',
        strategy='auto',
        devices=args.num_gpus,
        callbacks=[ckpt_callback, lr_monitor_callback, early_stop_callback],
        precision=args.precision,
        check_val_every_n_epoch=args.check_val_every_n_epoch
    )

    # Model
    model = WIPFlow(
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=args.betas,
        num_flow_steps=args.num_flow_steps,
        ema_decay=args.ema_decay,
        eps=args.eps,
        vae_ckpt_path=args.vae_ckpt_path,
        cot_vae_path=args.cot_vae_path,
        ot_method=args.ot_method,
        ot_reg=args.ot_reg,
        ot_normalize_cost=args.ot_normalize_cost,
        t_schedule=args.t_schedule,
    )



    # Fit
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=args.resume_from_ckpt
    )

    # trainer.test(model=model, dataloaders=test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--precision', type=str, required=False,
                        choices=['bf16', '32', 'bf16-mixed'],
                        default='bf16',
                        help='The precision used for training.')

    # Data
    parser.add_argument('--data_path', type=str, required=False,
                        default='/home/joty/code/Solar_D2P/data',
                        help='Data path.')
    parser.add_argument("--years_train", nargs="+", type=int, default=[2019, 2020])
    parser.add_argument("--years_val", nargs="+", type=int, default=[2021])
    parser.add_argument("--years_test", nargs="+", type=int, default=[2022])
    parser.add_argument("--stride", type=int, default=1)
    
    parser.add_argument("--use_possible_starts", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--pin_memory", action="store_true", default=True)
    # Window sizes
    parser.add_argument('--input_len', type=int, required=False, default=4,
                        help='Window size of input data (history length).')
    parser.add_argument('--pred_len', type=int, required=False, default=12,
                        help='Prediction horizon (target length).')

    # Flow/eval
    parser.add_argument('--num_flow_steps', type=int, required=False, default=50,
                        help='Number of flow steps for evaluation.')
    parser.add_argument('--eps', type=float, required=False, default=0.0,
                        help='Starting time of the flow (avoid exact 0).')




    # Optimizer
    parser.add_argument('--weight_decay', type=float, required=False, default=1e-2,
                        help='Optimizer weight decay.')
    parser.add_argument('--lr', type=float, required=False, default=5e-4,
                        help='Optimizer learning rate.')
    parser.add_argument('--betas', type=tuple, required=False, default=(0.9, 0.95),
                        help='Betas for the AdamW optimizer.')

    # Training schedule
    parser.add_argument('--max_epochs', type=int, required=False, default=500,
                        help='Number of training epochs.')
    parser.add_argument('--num_gpus', type=int, required=False, default=4,
                        help='Number of gpus to use.')
    parser.add_argument('--check_val_every_n_epoch', type=int, required=False, default=1,
                        help='Check validation every n epochs.')
    parser.add_argument('--num_workers', type=int, required=False, default=8,
                        help='Number of dataloader workers.')

    # EMA / checkpoints
    parser.add_argument('--ema_decay', type=float, required=False, default=0.9999,
                        help='Exponential moving average decay.')
    parser.add_argument('--resume_from_ckpt', type=str, required=False, default=None,
                        help='Resume lightning training from this checkpoint path.')

    # VAE 
    parser.add_argument('--vae_ckpt_path', type=str, required=False, default=None,
                        help='Path to pretrained VAE checkpoint. If None, uses the default in code.')
    parser.add_argument('--cot_vae_path', type=str, required=False, default=None,
                        help='Path to pretrained cot VAE checkpoint. If None, uses the default in code.')
    # Logging
    parser.add_argument('--wandb_project_name', type=str, required=True, default='flow_matching',
                        help='Project name for Weights & Biases logger.')
    parser.add_argument('--wandb_group', type=str, required=False, default=None,
                        help='Group name for W&B runs.')
    parser.add_argument('--wandb_id', type=str, required=False, default=None,
                        help='Run id to resume W&B logging.')

    main(parser.parse_args())