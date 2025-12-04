import math
import warnings
from typing import Union, Optional
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.cm as cm
from contextlib import contextmanager, nullcontext
from pathlib import Path

from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch_ema import ExponentialMovingAverage as EMA
from torchvision.utils import make_grid, save_image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from torch.nn.functional import sigmoid
from torchdyn.core import NeuralODE
from diffusers import AutoencoderKL
import wandb
from models.FM.interpolant import Interpolant
from models.denoiser.unets.denoiser import Denoiser
from models.FM.AIC.AR_Correction import forecast as anvil_forecast
from models.FM.AIC.lucaskanade import dense_lucaskanade
import torch.distributed as dist


# ----------------- CFM -----------------
def pad_t_like_x(t, x):

    if isinstance(t, (float, int)):
        return t
    if t.dim() == 1:
        return t.view(-1, *([1] * (x.dim() - 1)))
    return t.reshape(-1, *([1] * (x.dim() - 1)))

class VelocityFieldWrapper(torch.nn.Module):
    def __init__(self, denoiser, his_csi):
        super().__init__()
        self.denoiser = denoiser
        self.his_csi = his_csi

    def forward(self, t, x, args=None, kwargs=None):
        B = x.shape[0]
        if torch.is_tensor(t):
            t_batch = t.to(dtype=x.dtype, device=x.device).expand(B)
        else:
            t_batch = torch.full((B,), float(t), device=x.device, dtype=x.dtype)
        return self.denoiser(sample=x, timestep=t_batch, his_seq=self.his_csi)


def create_grid(
    img,
    normalize=False,
    num_images=5,
    nrow=4,
    cmap_name="viridis",
    shared_scale=True,
    fixed_range=(0.0, 1.2),
):
    assert img.dim() in (4, 5), f"Unsupported input shape: {img.shape}"
    img = img[:num_images]
    if img.dim() == 5:
        B, C, T, H, W = img.shape
        img = img.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
    N, C, H, W = img.shape
    assert C >= 1, f"Expected at least 1 channel, got C={C}"
    x = img.detach().float()
    eps = 1e-6
    if shared_scale:
        if fixed_range is not None:
            vmin, vmax = float(fixed_range[0]), float(fixed_range[1])
        else:
            vmin = float(x.min())
            vmax = float(x.max())
        if vmax <= vmin:
            vmax = vmin + eps
    cmap = cm.get_cmap(cmap_name)
    rgb_list = []
    for n in range(N):
        m = x[n, 0]
        if not shared_scale:
            vmin = float(m.min())
            vmax = float(m.max())
            if vmax <= vmin:
                vmax = vmin + eps
        m01 = (m - vmin) / (vmax - vmin)
        m01 = m01.clamp(0, 1)

        rgba = cmap(m01.cpu().numpy())
        rgb = torch.from_numpy((rgba[..., :3] * 255).astype(np.uint8)).permute(2, 0, 1)
        rgb_list.append(rgb.float() / 255.0)
    rgb_batch = torch.stack(rgb_list, dim=0)
    grid = make_grid(rgb_batch, padding=3, normalize=False, nrow=nrow)
    return grid


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag



class WIPFlow(LightningModule):
    def __init__(
        self,
        lr=5e-4,
        weight_decay=1e-3,
        betas=(0.9, 0.95),
        num_flow_steps=500,
        ema_decay=0.9999,
        eps=0.0,
        vae_ckpt_path=None,
        cot_vae_path=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # --------- UNet（velocity field） ---------
        self.model = Denoiser(
            sample_size=16,
            in_channels=32,
            out_channels=32,
            num_frames=1,
            down_block_types=(
                "CrossAttnDownBlockSpatioTemporal",
                "CrossAttnDownBlockSpatioTemporal",
                "CrossAttnDownBlockSpatioTemporal",
                "DownBlockSpatioTemporal",
            ),
            up_block_types=(
                "UpBlockSpatioTemporal",
                "CrossAttnUpBlockSpatioTemporal",
                "CrossAttnUpBlockSpatioTemporal",
                "CrossAttnUpBlockSpatioTemporal",
            ),
            block_out_channels=(128, 256, 512, 512),
            layers_per_block=2,
            N_T=3,
            channel_hid=256,
        )
        self.num_flow_steps = num_flow_steps

        self.ema = EMA(self.model.parameters(), decay=ema_decay) if self.ema_wanted else None
        # --------- VAE & COT-VAE ---------
        self.vae = AutoencoderKL(
            in_channels=1,
            out_channels=1,
            latent_channels=32,
            layers_per_block=2,
            down_block_types=(
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ),
            up_block_types=(
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ),
            block_out_channels=[128, 256, 512, 512],
            sample_size=128,
        )
        self.vae_path = vae_ckpt_path
        if self.vae_path is not None:
            state_dict = torch.load(self.vae_path, map_location=torch.device("cpu"))
            new_state_dict = {
                k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()
            }
            self.vae.load_state_dict(new_state_dict, strict=False)
        else:
            warnings.warn(
                f"Pretrained weights for `AutoencoderKL` not set. Run for sanity check only."
            )
        self.vae.eval()
        requires_grad(self.vae, False)

        self.cot_vae = AutoencoderKL(
            in_channels=1,
            out_channels=1,
            latent_channels=32,
            layers_per_block=2,
            down_block_types=(
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ),
            up_block_types=(
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ),
            block_out_channels=[128, 256, 512, 512],
            sample_size=128,
        )
        self.cot_vae_path = cot_vae_path
        if self.cot_vae_path is not None:
            state_dict = torch.load(self.cot_vae_path, map_location=torch.device("cpu"))
            cot_state_dict = {
                k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()
            }
            self.cot_vae.load_state_dict(cot_state_dict, strict=False)
        else:
            warnings.warn(
                f"Pretrained weights for `AutoencoderKL` not set. Run for sanity check only."
            )
        self.cot_vae.eval()
        requires_grad(self.cot_vae, False)
        self.interpolant = Interpolant(sigma_coef=self.matcher_sigma, beta_fn="t^2")
        self.test_results_path = None

    # --------- EMA ---------
    @property
    def ema_wanted(self):
        return self.hparams.ema_decay != -1

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.ema_wanted:
            checkpoint["ema"] = self.ema.state_dict()
        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if self.ema_wanted and "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])
        return super().on_load_checkpoint(checkpoint)

    def on_before_zero_grad(self, optimizer) -> None:
        if self.ema_wanted:
            self.ema.update(self.model.parameters())
        return super().on_before_zero_grad(optimizer)

    def to(self, *args, **kwargs):
        if self.ema_wanted:
            self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    @contextmanager
    def maybe_ema(self):
        ema = self.ema
        ctx = nullcontext if ema is None else ema.average_parameters
        yield ctx

    # --------- ---------
    @torch.no_grad()
    def encode(self, x, use_mean=False):
        if len(x.shape) == 5:
            B, C, T, H, W = x.shape
            latent_z = []
            for t in range(T):
                current_images = x[:, :, t, :, :]
                dist = self.vae.encode(current_images).latent_dist
                z = dist.mean if use_mean else dist.sample()
                latent_z.append(z)
            return torch.stack(latent_z, dim=2)
        elif len(x.shape) == 4:
            dist = self.vae.encode(x).latent_dist
            z = dist.mean if use_mean else dist.sample()
            return z
        else:
            raise ValueError(
                f"Invalid input shape: {x.shape}. Expected [B, C, T, H, W] or [B, C, H, W]."
            )

    @torch.no_grad()
    def decode(self, latent_z):
        if len(latent_z.shape) == 5:  # [B, C, T, latent_H, latent_W]
            B, C, T, latent_H, latent_W = latent_z.shape
            reconstructed_images = []
            for t in range(T):
                current_latent = latent_z[:, :, t, :, :]
                current_image = self.vae.decode(current_latent).sample
                reconstructed_images.append(current_image)
            return torch.stack(reconstructed_images, dim=2)
        elif len(latent_z.shape) == 4:
            reconstructed_image = self.vae.decode(latent_z).sample
            return reconstructed_image
        else:
            raise ValueError(
                f"Invalid latent representation shape: {latent_z.shape}. "
                "Expected [B, C, T, latent_H, latent_W] or [B, C, latent_H, latent_W]."
            )

    @torch.no_grad()
    def cot_encode(self, x, use_mean=False):
        if len(x.shape) == 5:
            B, C, T, H, W = x.shape
            latent_z = []
            for t in range(T):
                current_images = x[:, :, t, :, :]
                dist = self.cot_vae.encode(current_images).latent_dist
                z = dist.mean if use_mean else dist.sample()
                latent_z.append(z)
            return torch.stack(latent_z, dim=2)
        elif len(x.shape) == 4:
            dist = self.cot_vae.encode(x).latent_dist
            z = dist.mean if use_mean else dist.sample()
            return z
        else:
            raise ValueError(
                f"Invalid input shape: {x.shape}. Expected [B, C, T, H, W] or [B, C, H, W]."
            )

    @torch.no_grad()
    def cot_decode(self, latent_z):
        if len(latent_z.shape) == 5:
            B, C, T, latent_H, latent_W = latent_z.shape
            reconstructed_images = []
            for t in range(T):
                current_latent = latent_z[:, :, t, :, :]
                current_image = self.cot_vae.decode(current_latent).sample
                reconstructed_images.append(current_image)
            return torch.stack(reconstructed_images, dim=2)
        elif len(latent_z.shape) == 4:
            reconstructed_image = self.cot_vae.decode(latent_z).sample
            return reconstructed_image
        else:
            raise ValueError(
                f"Invalid latent representation shape: {latent_z.shape}. "
                "Expected [B, C, T, latent_H, latent_W] or [B, C, latent_H, latent_W]."
            )


    def inverse_rescale_data(
        self, scaled_data: torch.Tensor, min_val: float = 0.05, max_val: float = 1.2
    ) -> torch.Tensor:
        return ((scaled_data + 1.0) / 2.0) * (max_val - min_val) + min_val

    def rescale_data(
        self, x: torch.Tensor, min_val: float = 0.05, max_val: float = 1.2
    ) -> torch.Tensor:
        x_clamped = torch.clamp(x, min=min_val, max=max_val)
        return 2.0 * ((x_clamped - min_val) / (max_val - min_val)) - 1.0
    def fill_nan_timewise_pixelwise_gaussian(self, x, clip_min=None, clip_max=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        assert x.ndim == 3, "Expected shape (T, H, W)"
        T, H, W = x.shape
        out = x.copy()
        global_mu = np.nanmean(out)
        global_sigma = np.nanstd(out)
        if not np.isfinite(global_sigma) or global_sigma == 0:
            global_sigma = 1e-6

        for h in range(H):
            row = out[:, h, :]  # (T, W)
            mask_nan = np.isnan(row)
            for w in range(W):
                col = row[:, w]  # (T,)
                mask_col_nan = mask_nan[:, w]
                if not mask_col_nan.any():
                    continue

                mu = np.nanmean(col)
                sigma = np.nanstd(col)

                if not np.isfinite(mu):
                    mu = global_mu
                if not np.isfinite(sigma) or sigma == 0:
                    sigma = global_sigma if global_sigma > 0 else 1e-6

                n_fill = mask_col_nan.sum()
                noise = rng.normal(mu, sigma, size=n_fill)

                if clip_min is not None or clip_max is not None:
                    noise = np.clip(noise, clip_min if clip_min is not None else -np.inf,
                                    clip_max if clip_max is not None else np.inf)

                col[mask_col_nan] = noise
                row[:, w] = col  
            out[:, h, :] = row  
        return out
    def run_anvil_on_batch(self, input_csi, tp_future: int, ar_order: int = 2):
        device = input_csi.device
        input_csi_np = input_csi.detach().cpu().numpy()
        B, C, Th, H, W = input_csi_np.shape

        tp_future = int(tp_future)

        all_preds = []

        for b in range(B):
            seq = input_csi_np[b, 0]  # (Th, H, W)
            vil_input = seq[-4:]                     # (Th, H, W)
            fd_kwargs = {}
            fd_kwargs["max_corners"] = 1000
            fd_kwargs["quality_level"] = 0.01
            fd_kwargs["min_distance"] = 2
            fd_kwargs["block_size"] = 8

            lk_kwargs = {}
            lk_kwargs["winsize"] = (15, 15)

            oflow_kwargs = {}
            oflow_kwargs["fd_kwargs"] = fd_kwargs
            oflow_kwargs["lk_kwargs"] = lk_kwargs
            oflow_kwargs["decl_scale"] = 10
            velocity = dense_lucaskanade(vil_input, **oflow_kwargs)
            # ux = np.zeros((H, W), dtype=float)
            # uy = np.zeros((H, W), dtype=float)
            # velocity = np.stack([ux, uy], axis=0)

            pred = anvil_forecast(
                vil=vil_input,        # (Th, H, W)
                velocity=velocity,    # (2, H, W)
                timesteps=tp_future,  
                rainrate=None,
                n_cascade_levels=6,
                extrap_method="semilagrangian",
                ar_order=ar_order,
                ar_window_radius=50,
                fft_method="numpy",
                apply_rainrate_mask=False,
                num_workers=16,
                extrap_kwargs=None,
                filter_kwargs=None,
                measure_time=False,
            )  
            anvil_pred = self.fill_nan_timewise_pixelwise_gaussian(pred, clip_min=0.0, clip_max=1.2)
            all_preds.append(anvil_pred[None, None])  

        all_preds_np = np.concatenate(all_preds, axis=0)
        all_preds_t  = torch.from_numpy(all_preds_np).to(device=device, dtype=input_csi.dtype)
        return all_preds_t
    

    def forward(self, x_t, timestep, his_csi, his_cot):
        return self.model(sample=x_t, timestep=timestep, his_seq=his_csi, his_cot=his_cot)

   
    def training_step(self, batch, batch_idx):

        cot = batch["input_cot"]
        target = batch["target_csi"]
        input_csi = batch["his_csi"]
        his_csi_non = batch['his_csi_non']
        _, _, Tf, _, _ = target.shape
        
        cot_latent = self.cot_encode(cot)
        x1 = self.encode(target)
        his_csi = self.encode(input_csi)
        anvil_pred = self.run_anvil_on_batch(
            input_csi=his_csi_non,   
            tp_future=Tf,
            ar_order=2,           
        )  
        anvil_pred = self.rescale_data(anvil_pred)
        x0 = self.encode(anvil_pred)
        # ---------- x0 ----------
        with torch.no_grad():
            t_scalar, x_t, u_t = self.interpolant.sample_location_and_conditional_flow(
                x0=x0, x1=x1, t=None
            )

        if t_scalar.dim() > 1:
            t_scalar = t_scalar.view(t_scalar.shape[0])

        v_pred = self(x_t, t_scalar, his_csi, cot_latent)
        loss = F.mse_loss(v_pred, u_t, reduction="mean")

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        cot = batch["input_cot"]
        target = batch["target_csi"]
        input_csi = batch["his_csi"]
        his_csi_non = batch['his_csi_non']
        cot_latent = self.cot_encode(cot)
        x1 = self.encode(target)
        his_csi = self.encode(input_csi)
        _, _, Tf, _, _ = target.shape
        anvil_pred = self.run_anvil_on_batch(
            input_csi=his_csi_non,   
            tp_future=Tf,
            ar_order=2,           
        )  
        anvil_pred = self.rescale_data(anvil_pred)
        x0 = self.encode(anvil_pred)
    
        with torch.no_grad():
            t_scalar, x_t, u_t = self.interpolant.sample_location_and_conditional_flow(
                x0=x0, x1=x1, t=None
            )

        if t_scalar.dim() > 1:
            t_scalar = t_scalar.view(t_scalar.shape[0])

        v_pred = self(x_t, t_scalar, his_csi, cot_latent)
        loss = F.mse_loss(v_pred, u_t, reduction="mean")

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def on_test_start(self):
        self._vf_wrapper = VelocityFieldWrapper(self.model, his_csi=None).to(self.device)
        self.test_node = NeuralODE(
            self._vf_wrapper,
            solver="dopri5",
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
        )
        self._test_preds = []
        self._test_targets = []
        self._clear_ssi = []
        self._input = []
  
        default_dir = Path(getattr(self, "test_results_path", "./test_results"))
        self.test_save_dir = default_dir
        self.test_save_dir.mkdir(parents=True, exist_ok=True)
        
        
    @torch.no_grad()
    def generate_reconstructions(self, x0, cot_latent, x, y, num_flow_steps, result_device):
        with self.maybe_ema():
            source_dist_samples = x0
            dt = (1.0 / num_flow_steps) * (1.0 - self.hparams.eps)
            x_t_next = source_dist_samples.clone()
            x_t_seq = [x_t_next]
            t_one = torch.ones(x.shape[0], device=self.device)
            for i in range(num_flow_steps):
                num_t = (i / num_flow_steps) * (1.0 - self.hparams.eps) + self.hparams.eps
                v_t_next = self(x_t_next, t_one * num_t, y, cot_latent).to(x_t_next.dtype)
                x_t_next = x_t_next.clone() + v_t_next * dt
                x_t_seq.append(x_t_next.to(result_device))
            xhat = x_t_seq[-1].to(torch.float32)
            source_dist_samples = source_dist_samples.to(result_device)
            return xhat.to(result_device), x_t_seq, source_dist_samples

    def test_step(self, batch, batch_idx):
        cot = batch['input_cot']
        target = batch['target_csi']        
        input_csi = batch['his_csi']
        his_csi_non = batch['his_csi_non']
        cot_latent = self.cot_encode(cot)
        x1 = self.encode(target) 
        his_csi = self.encode(input_csi)
        _, _, Tf, _, _ = target.shape
        anvil_pred = self.run_anvil_on_batch(
            input_csi=his_csi_non,   
            tp_future=Tf,
            ar_order=2,           
        )  
        anvil_pred = self.rescale_data(anvil_pred)
        x0 = self.encode(anvil_pred)
        num_steps = int(self.num_flow_steps)
        
        with self.maybe_ema():
            xhat_latent, _, _ = self.generate_reconstructions(
                x0=x0,
                cot_latent=cot_latent,
                x=x1,
                y=his_csi,
                num_flow_steps=num_steps,
                result_device=torch.device("cpu"),
            )
        pred_main = self.decode(xhat_latent.to(self.device))
        AIC = self.decode(x0.to(self.device))
        
        pred_main_vis = self.inverse_rescale_data(pred_main)
        target_vis = self.inverse_rescale_data(target)
        input_vis = self.inverse_rescale_data(AIC)
        
        test_recon_mse = ((target_vis - pred_main_vis) ** 2).mean()
        self.log("test/loss_recon", test_recon_mse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        if not hasattr(self, "_test_preds"):   self._test_preds = []
        if not hasattr(self, "_test_targets"): self._test_targets = []
        if not hasattr(self, "_input"):        self._input = []

        self._test_preds.append(pred_main_vis.detach().cpu().float())
        self._input.append(input_vis.detach().cpu().float())
        self._test_targets.append(target_vis.detach().cpu().float())
        
        K = 10          
        num_images = 10
        nrow = 8
        try:
            wandb_logger = getattr(self.logger, "experiment", None)
            if wandb_logger is not None and batch_idx < K:
                target_grid = create_grid(target_vis,     num_images=num_images, nrow=nrow)
                input_grid  = create_grid(input_vis,      num_images=num_images, nrow=nrow)
                pred_grid   = create_grid(pred_main_vis,  num_images=num_images, nrow=nrow)
                wandb_logger.log({
                    "test_images/target":     [wandb.Image(to_pil_image(target_grid))],
                    "test_images/input":      [wandb.Image(to_pil_image(input_grid))],
                    "test_images/prediction": [wandb.Image(to_pil_image(pred_grid))],  
                })
        except Exception as e:
            self.print(f"[WARN][test_step] W&B image logging failed: {e}")

        return {"test/loss_vec": test_recon_mse.detach()}
        
    
    def on_test_epoch_end(self):
        preds = np.empty((0,))
        gts   = np.empty((0,))
        aic = np.empty((0,))
        if hasattr(self, "_test_preds") and len(self._test_preds) > 0:
            preds = torch.cat(self._test_preds, dim=0).cpu().numpy()   
        if hasattr(self, "_test_targets") and len(self._test_targets) > 0:
            gts = torch.cat(self._test_targets, dim=0).cpu().numpy()
        if hasattr(self, "_input") and len(self._input) > 0:
            aic = torch.cat(self._input, dim=0).cpu().numpy()
        save_pred = (self.test_save_dir / "predictions.npy").as_posix()
        save_gt   = (self.test_save_dir / "targets.npy").as_posix()
        save_aic   = (self.test_save_dir / "aic.npy").as_posix()
        np.save(save_pred, preds)
        np.save(save_gt, gts)
        np.save(save_aic, aic)
        self.print(f"[Test] Saved predictions to: {save_pred} (shape={preds.shape})")
        self.print(f"[Test] Saved targets to: {save_gt} (shape={gts.shape})")
        self.print(f"[Test] Saved aic to: {save_aic} (shape={aic.shape})")
        self._test_preds.clear(); self._test_targets.clear(); self._input.clear()
    
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(),
                          betas=self.hparams.betas,
                          eps=1e-8,
                          lr=self.hparams.lr,
                          weight_decay=self.hparams.weight_decay)
        return optimizer