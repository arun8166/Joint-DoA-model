"""
Paper-faithful denoising diffusion DOA estimator adapted to a 16-element ULA.

Based on:
  F. Qian, C. Zhou, Z. Shi, "Denoising Diffusion Model for DOA Estimation",
  ICASSP 2026.

Expected MATLAB variables:
  y_receive              : (16, 50, 7, 5000), complex
  y_receive_ultra_clean  : (16, 50, 7, 5000), complex; use its 0 dB slice (MATLAB index 3) as X0
  target_azimuth         : (1, 4, 5000), radians

Paper-matching defaults used here:
  - snapshot-wise denoising
  - epsilon/noise prediction objective, Eq. (8)
  - conditional parallel-path U-Net with 2 down + 2 up stages
  - sinusoidal diffusion-step embedding in the conditional path
  - Ns = 20 reverse states
  - 200 epochs, batch size 64, learning rate 1e-4
  - all 7 real SNR points are used for training
  - 1,400,000 training snapshot pairs: 4000 setups x 50 snapshots x 7 SNRs
  - 1000 test setups: last 1000 setups
  - reverse transition mean exactly follows Eq. (7)
  - reverse covariance uses (1 - alpha_t) I as written in the paper
  - final DOAs are extracted with ESPRIT

IMPORTANT PAPER DETAILS NOT SPECIFIED IN THE 5-PAGE PAPER:
  1) exact alpha_t schedule,
  2) convolution channel counts / kernel sizes / activations,
  3) optimizer choice,
  4) exact training SNR construction,
  5) 1-D ULA angle convention (the paper itself uses an 8x8 URA and 2-D DOA).

Those choices are isolated/configurable below. The mathematical pipeline is kept as close
as possible to the paper, while adapting its final ESPRIT stage to your 16-antenna,
4-source, azimuth-only data.

Example:
  python paper_diffusion_doa_all_snr.py --mat "C:/path/to/data.mat"

Evaluate an existing checkpoint only:
  python paper_diffusion_doa_all_snr.py --mat "C:/path/to/data.mat" --mode eval \
      --checkpoint diffusion_doa_paper.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    import scipy.io as sio
    from scipy.optimize import linear_sum_assignment
except Exception as exc:
    raise RuntimeError("This script requires scipy: pip install scipy") from exc

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception as exc:
    raise RuntimeError("This script requires PyTorch: https://pytorch.org/get-started/") from exc

try:
    import h5py
except Exception:
    h5py = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class Config:
    # Data geometry
    num_antennas: int = 16
    num_snapshots: int = 50
    num_snr: int = 7
    num_setups: int = 5000
    num_sources: int = 4
    snr_db: Tuple[float, ...] = (-20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0)

    # First 4000 setups are used for training and the last 1000 for testing.
    # Training uses ALL seven real SNR points:
    # {-20,-10,0,10,20,30,40} dB.
    # This gives 4000 x 50 x 7 = 1,400,000 snapshot-wise training pairs.
    train_setups: int = 4000
    test_setups: int = 1000

    # Paper hyperparameters
    ns: int = 20
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-4

    # Architecture values are not stated in the paper.
    base_channels: int = 64

    # alpha_t schedule is not specified in the paper.
    # "cosine" gives alpha_N ~ 0, consistent with Eq. (4)'s X_N ~ pure Gaussian.
    schedule: str = "cosine"  # cosine | linear_alpha
    alpha_start: float = 0.999
    alpha_end: float = 0.001

    # ULA steering convention used by ESPRIT:
    #   a_m(theta) = exp(-j*pi*m*sin(theta))  -> "sin"
    # If your simulator used exp(-j*pi*m*cos(theta)), use --angle-convention cos.
    angle_convention: str = "sin"

    # Runtime / reproducibility
    seed: int = 1234
    num_workers: int = 0
    amp: bool = False
    deterministic: bool = False
    log_every: int = 500
    eval_setup_batch: int = 64

    # Paper says to sample from CN(mu, (1-alpha_t)I) at each reverse step.
    # Leave True for the paper equation; False uses mu at t=1 only.
    sample_final_step: bool = True


# -----------------------------------------------------------------------------
# MATLAB loading
# -----------------------------------------------------------------------------

def _compound_or_group_to_complex(x):
    """Convert common MATLAB-v7.3 complex representations to NumPy complex."""
    if isinstance(x, np.ndarray) and x.dtype.fields:
        names = set(x.dtype.fields.keys())
        if {"real", "imag"}.issubset(names):
            return x["real"] + 1j * x["imag"]
    return x


def _read_hdf5_variable(path: str, var_name: str) -> np.ndarray:
    if h5py is None:
        raise RuntimeError("MAT-file appears to be v7.3/HDF5, but h5py is not installed.")
    with h5py.File(path, "r") as f:
        if var_name not in f:
            raise KeyError(f"Variable '{var_name}' not found in {path}. Keys: {list(f.keys())}")
        obj = f[var_name]
        if isinstance(obj, h5py.Group):
            keys = set(obj.keys())
            if {"real", "imag"}.issubset(keys):
                arr = np.asarray(obj["real"]) + 1j * np.asarray(obj["imag"])
            else:
                raise RuntimeError(f"Unsupported HDF5 group format for variable '{var_name}': {keys}")
        else:
            arr = np.asarray(obj)
            arr = _compound_or_group_to_complex(arr)
    return arr


def load_mat_variable(path: str, var_name: str) -> np.ndarray:
    """Load one variable at a time to keep peak RAM lower for classic MAT files."""
    try:
        d = sio.loadmat(path, variable_names=[var_name], squeeze_me=False, struct_as_record=False)
        if var_name not in d:
            raise KeyError(f"Variable '{var_name}' not found in {path}")
        return np.asarray(d[var_name])
    except (NotImplementedError, ValueError, OSError):
        return _read_hdf5_variable(path, var_name)


def canonicalize_signal_array(a: np.ndarray, cfg: Config, name: str) -> np.ndarray:
    """Return complex array in canonical MATLAB order (M,T,SNR,Nsetup)."""
    target = (cfg.num_antennas, cfg.num_snapshots, cfg.num_snr, cfg.num_setups)
    a = np.asarray(a)

    if a.shape == target:
        out = a
    elif a.shape == target[::-1]:
        # MATLAB v7.3 arrays often appear with reversed dimensions via h5py.
        out = np.transpose(a, (3, 2, 1, 0))
    else:
        # General permutation recovery if all dimensions are unique.
        # Here 16, 50, 7, 5000 are unique, so this is unambiguous.
        shape = list(a.shape)
        if len(shape) != 4 or sorted(shape) != sorted(target):
            raise ValueError(f"{name} has shape {a.shape}, expected a permutation of {target}")
        perm = [shape.index(v) for v in target]
        out = np.transpose(a, perm)

    if not np.iscomplexobj(out):
        raise ValueError(f"{name} is not complex-valued after loading (dtype={out.dtype}).")
    return np.asarray(out, dtype=np.complex64)


def canonicalize_angles(a: np.ndarray, cfg: Config) -> np.ndarray:
    """Return target angles as (Nsetup,K), radians."""
    a = np.asarray(a)
    a = np.squeeze(a)
    if a.shape == (cfg.num_sources, cfg.num_setups):
        out = a.T
    elif a.shape == (cfg.num_setups, cfg.num_sources):
        out = a
    else:
        raise ValueError(
            f"target_azimuth becomes shape {a.shape} after squeeze; expected "
            f"({cfg.num_sources},{cfg.num_setups}) or ({cfg.num_setups},{cfg.num_sources})."
        )
    return np.asarray(out, dtype=np.float64)


def choose_clean_reference(clean: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Use the 0 dB slice of y_receive_ultra_clean as X0 for every SNR condition.

    MATLAB indexing:
        y_receive_ultra_clean(x, y, 3, z)

    Python/NumPy uses zero-based indexing, so MATLAB index 3 corresponds to
    Python index 2. With cfg.snr_db = (-20,-10,0,10,20,30,40), index 2 is 0 dB.

    Returns shape (M,T,Nsetup).
    """
    zero_db_indices = np.where(np.isclose(np.asarray(cfg.snr_db), 0.0))[0]
    if len(zero_db_indices) != 1:
        raise ValueError(
            f"Expected exactly one 0 dB entry in cfg.snr_db, got {cfg.snr_db}"
        )

    idx = int(zero_db_indices[0])
    if idx != 2:
        raise ValueError(
            f"0 dB is at Python index {idx}, but expected index 2 "
            f"(MATLAB index 3) for this dataset."
        )

    print(
        "[data] X0 reference = y_receive_ultra_clean[:,:,2,:] in Python "
        "(MATLAB third SNR index, 0 dB)"
    )
    return clean[:, :, idx, :]


def prepare_arrays(mat_path: str, cfg: Config):
    if cfg.train_setups + cfg.test_setups > cfg.num_setups:
        raise ValueError("train_setups + test_setups exceeds num_setups")
    test_start = cfg.num_setups - cfg.test_setups
    if cfg.train_setups > test_start:
        raise ValueError("Training and test setup ranges overlap.")

    print("[data] loading y_receive ...")
    y_all = canonicalize_signal_array(load_mat_variable(mat_path, "y_receive"), cfg, "y_receive")

    print("[data] loading y_receive_ultra_clean ...")
    x0_all = canonicalize_signal_array(
        load_mat_variable(mat_path, "y_receive_ultra_clean"), cfg, "y_receive_ultra_clean"
    )

    print("[data] loading target_azimuth ...")
    az_all = canonicalize_angles(load_mat_variable(mat_path, "target_azimuth"), cfg)

    # Use ONLY the 0 dB ultra-clean slice as X0 for every noisy SNR condition.
    # MATLAB: y_receive_ultra_clean(x,y,3,z)
    # Python: y_receive_ultra_clean[:,:,2,:]
    # Shape after selection: (M,T,Ntrain)
    x0_train = choose_clean_reference(x0_all, cfg)[:, :, : cfg.train_setups].copy()

    # IMPORTANT: use ALL seven real SNR points for training.
    # Shape: (M,T,SNR,Ntrain). No -5 dB synthesis or interpolation is performed.
    x_train = y_all[:, :, :, : cfg.train_setups].copy()

    # Keep all seven SNRs for the held-out test setups.
    x_test = y_all[:, :, :, test_start:].copy()
    az_test = az_all[test_start:].copy()

    del y_all, x0_all, az_all

    expected = cfg.train_setups * cfg.num_snapshots * cfg.num_snr

    if x_train.shape != (
        cfg.num_antennas,
        cfg.num_snapshots,
        cfg.num_snr,
        cfg.train_setups,
    ):
        raise RuntimeError(f"Unexpected training-array shape: {x_train.shape}")

    if x0_train.shape != (
        cfg.num_antennas,
        cfg.num_snapshots,
        cfg.train_setups,
    ):
        raise RuntimeError(f"Unexpected clean-training-array shape: {x0_train.shape}")

    print(
        "[data] training uses ALL provided SNR points: "
        + ", ".join(f"{v:g}" for v in cfg.snr_db)
        + " dB"
    )
    print(f"[data] training snapshot pairs: {expected:,}")
    print(
        f"[data] = {cfg.train_setups:,} setups x "
        f"{cfg.num_snapshots} snapshots x {cfg.num_snr} SNR points"
    )
    print(f"[data] test setups: {cfg.test_setups:,} at {cfg.num_snr} SNR points")

    return x_train, x0_train, x_test, az_test


# -----------------------------------------------------------------------------
# Complex <-> 2-real-channel conversion
# -----------------------------------------------------------------------------

def complex_np_to_2ch(x: np.ndarray) -> np.ndarray:
    """(...,M) complex -> (...,2,M) float32."""
    return np.stack((x.real, x.imag), axis=-2).astype(np.float32, copy=False)


def twoch_to_complex_np(x: np.ndarray) -> np.ndarray:
    """(...,2,M) real -> (...,M) complex64."""
    return (x[..., 0, :] + 1j * x[..., 1, :]).astype(np.complex64, copy=False)


def complex_standard_normal_like(x: torch.Tensor) -> torch.Tensor:
    """
    CN(0,I) represented by two real channels.
    Each real/imag component has variance 1/2, so E|z|^2 = 1.
    """
    return torch.randn_like(x) / math.sqrt(2.0)


class AllSNRSnapshotPairs(Dataset):
    """
    Snapshot-wise training pairs over every real SNR point.

    x_cond_complex: (M,T,S,N)
    x0_complex:     (M,T,N)

    For every setup and snapshot, the same clean X0 snapshot is paired with
    each of the S noisy received versions. Complex vectors are converted to
    two real channels lazily in __getitem__ to avoid replicating X0 in RAM.
    """

    def __init__(self, x_cond_complex: np.ndarray, x0_complex: np.ndarray):
        if x_cond_complex.ndim != 4:
            raise ValueError(
                f"x_cond_complex must have shape (M,T,S,N), got {x_cond_complex.shape}"
            )
        if x0_complex.ndim != 3:
            raise ValueError(
                f"x0_complex must have shape (M,T,N), got {x0_complex.shape}"
            )

        m, t, s, n = x_cond_complex.shape
        if x0_complex.shape != (m, t, n):
            raise ValueError(
                "condition/clean geometry mismatch: "
                f"{x_cond_complex.shape} vs {x0_complex.shape}"
            )

        self.x = x_cond_complex
        self.x0 = x0_complex
        self.m = m
        self.t = t
        self.s = s
        self.n = n
        self.samples_per_setup = t * s

    def __len__(self):
        return self.n * self.samples_per_setup

    def __getitem__(self, idx):
        # Linear layout: setup -> SNR -> snapshot.
        setup = idx // self.samples_per_setup
        rem = idx % self.samples_per_setup
        snr = rem // self.t
        snap = rem % self.t

        x = self.x[:, snap, snr, setup]
        x0 = self.x0[:, snap, setup]

        x_2ch = np.stack((x.real, x.imag), axis=0).astype(np.float32, copy=False)
        x0_2ch = np.stack((x0.real, x0.imag), axis=0).astype(np.float32, copy=False)

        return torch.from_numpy(x_2ch), torch.from_numpy(x0_2ch)


# -----------------------------------------------------------------------------
# Diffusion schedule (alpha_t in the paper is the cumulative signal coefficient)
# -----------------------------------------------------------------------------

def make_alpha_schedule(cfg: Config, device: torch.device) -> torch.Tensor:
    ns = cfg.ns
    alpha = np.ones(ns + 1, dtype=np.float64)

    if cfg.schedule == "linear_alpha":
        alpha[1:] = np.linspace(cfg.alpha_start, cfg.alpha_end, ns, dtype=np.float64)

    elif cfg.schedule == "cosine":
        # Nichol-Dhariwal style cosine cumulative alpha schedule, converted to a
        # stable discrete sequence with beta_variance <= 0.999. This makes alpha_N
        # very small, matching the paper's statement X_N ~ CN(0,I).
        s = 0.008
        t = np.arange(ns + 1, dtype=np.float64)
        f = np.cos(((t / ns) + s) / (1.0 + s) * math.pi / 2.0) ** 2
        alpha_raw = f / f[0]
        # Convert to per-step variance beta, clip, then recumulate.
        var_beta = 1.0 - (alpha_raw[1:] / alpha_raw[:-1])
        var_beta = np.clip(var_beta, 1e-8, 0.999)
        alpha[0] = 1.0
        alpha[1:] = np.cumprod(1.0 - var_beta)

    else:
        raise ValueError(f"Unknown schedule: {cfg.schedule}")

    if not np.all(np.diff(alpha) < 0):
        raise RuntimeError("alpha_t must be strictly decreasing")
    if not (0.0 < alpha[-1] < alpha[1] < 1.0):
        raise RuntimeError(f"Invalid alpha schedule endpoints: alpha_1={alpha[1]}, alpha_N={alpha[-1]}")

    out = torch.tensor(alpha, dtype=torch.float32, device=device)
    print(
        f"[diffusion] schedule={cfg.schedule}, Ns={ns}, "
        f"alpha_1={alpha[1]:.6g}, alpha_N={alpha[-1]:.6g}"
    )
    return out


# -----------------------------------------------------------------------------
# Parallel conditional U-Net, adapted from Fig. 1 to a 1-D antenna axis
# -----------------------------------------------------------------------------

class SinusoidalStepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t is integer diffusion index in [1,Ns]. This is standard sinusoidal
        # positional embedding [Vaswani et al., cited by the paper].
        half = self.dim // 2
        if half == 0:
            return t.float().unsqueeze(1)
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb


class ActConv1d(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, kernel_size=kernel, stride=stride, padding=padding)

    def forward(self, x):
        return F.silu(self.conv(x))


class ActConvTranspose1d(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(cin, cout, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return F.silu(self.conv(x))


class ParallelConditionalUNet1D(nn.Module):
    """
    Paper Fig. 1 adapted to a ULA:
      main path      : noisy diffusion state X_t
      auxiliary path : received signal X + sinusoidal t embedding
      fusion         : element-wise additions at multiple scales
      depth          : 2 downsampling + 2 transposed-convolution upsampling layers
      output         : predicted complex noise Z_t (2 real channels)

    Exact channel counts/kernels are not given in the paper; base_channels=64 and
    common 3/4-tap 1-D kernels are therefore implementation choices.
    """

    def __init__(self, base: int = 64):
        super().__init__()
        c = base

        # Input convolutional layers
        self.main_in = ActConv1d(2, c, 3, 1, 1)
        self.cond_in = ActConv1d(2, c, 3, 1, 1)

        # Sinusoidal positional embedding concatenated channel-wise with X features.
        self.t_embed = SinusoidalStepEmbedding(c)
        self.cond_time_fuse = ActConv1d(2 * c, c, 1, 1, 0)

        # Two convolutional downsampling layers, structurally identical paths.
        self.main_down1 = ActConv1d(c, 2 * c, 4, 2, 1)
        self.main_down2 = ActConv1d(2 * c, 4 * c, 4, 2, 1)
        self.cond_down1 = ActConv1d(c, 2 * c, 4, 2, 1)
        self.cond_down2 = ActConv1d(2 * c, 4 * c, 4, 2, 1)

        # Two transposed-convolution upsampling layers.
        self.main_up1 = ActConvTranspose1d(4 * c, 2 * c)
        self.main_up2 = ActConvTranspose1d(2 * c, c)
        self.cond_up1 = ActConvTranspose1d(4 * c, 2 * c)
        self.cond_up2 = ActConvTranspose1d(2 * c, c)

        # Output convolutional layer, predicts real/imag noise channels.
        self.out = nn.Conv1d(c, 2, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        # Auxiliary conditional path + positional embedding.
        c0_x = self.cond_in(x_cond)
        te = self.t_embed(t).unsqueeze(-1).expand(-1, -1, c0_x.shape[-1])
        c0 = self.cond_time_fuse(torch.cat([c0_x, te], dim=1))

        # Main input and multi-scale condition addition.
        m0 = self.main_in(x_t) + c0

        c1 = self.cond_down1(c0)
        m1 = self.main_down1(m0) + c1

        c2 = self.cond_down2(c1)
        m2 = self.main_down2(m1) + c2

        # U-Net skip connections + condition fusion in the decoder.
        c1u = self.cond_up1(c2) + c1
        m1u = self.main_up1(m2) + m1 + c1u

        c0u = self.cond_up2(c1u) + c0
        m0u = self.main_up2(m1u) + m0 + c0u

        return self.out(m0u)


# -----------------------------------------------------------------------------
# Training: Eq. (4) + Eq. (8)
# -----------------------------------------------------------------------------

def seed_everything(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True


def train_model(
    model: nn.Module,
    loader: DataLoader,
    alpha: torch.Tensor,
    cfg: Config,
    device: torch.device,
    checkpoint_path: str,
    resume: bool,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))
    start_epoch = 1

    if resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[train] resumed from epoch {start_epoch - 1}: {checkpoint_path}")

    ns = cfg.ns
    model.train()
    global_step = 0

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        loss_sum = 0.0
        n_seen = 0

        for step, (x_cond, x0) in enumerate(loader, start=1):
            x_cond = x_cond.to(device, non_blocking=True)
            x0 = x0.to(device, non_blocking=True)
            b = x0.shape[0]

            # Uniform random tau in {1,...,Ns}, as stated in the paper.
            tau = torch.randint(1, ns + 1, (b,), device=device, dtype=torch.long)
            a = alpha[tau].view(b, 1, 1)
            z = complex_standard_normal_like(x0)

            # Eq. (4): X_tau = sqrt(alpha_tau) X0 + sqrt(1-alpha_tau) Z_tau.
            x_tau = torch.sqrt(a) * x0 + torch.sqrt(1.0 - a) * z

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                z_hat = model(x_tau, tau, x_cond)
                # Eq. (8). Mean-MSE differs from squared Euclidean norm only by
                # a constant scale and has the same optimum.
                loss = F.mse_loss(z_hat, z, reduction="mean")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += float(loss.detach()) * b
            n_seen += b
            global_step += 1

            if cfg.log_every > 0 and step % cfg.log_every == 0:
                print(
                    f"[train] epoch {epoch:03d}/{cfg.epochs} step {step:05d}/{len(loader):05d} "
                    f"loss={loss_sum / n_seen:.7f}"
                )

        epoch_loss = loss_sum / max(n_seen, 1)
        dt = time.time() - t0
        print(f"[train] epoch {epoch:03d}/{cfg.epochs} loss={epoch_loss:.8f} time={dt:.1f}s")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(cfg),
        }
        torch.save(ckpt, checkpoint_path)

    print(f"[train] final checkpoint saved to: {checkpoint_path}")


# -----------------------------------------------------------------------------
# Reverse diffusion: Eq. (7) + p-hat covariance from Sec. 3.2
# -----------------------------------------------------------------------------

@torch.no_grad()
def reverse_denoise(
    model: nn.Module,
    x_cond_2ch: torch.Tensor,
    alpha: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    """Denoise a batch of independent snapshots. Input/output shape (B,2,M)."""
    device = x_cond_2ch.device
    b = x_cond_2ch.shape[0]
    model.eval()

    # Paper: X_Ns becomes pure Gaussian Z ~ CN(0,I).
    x_t = complex_standard_normal_like(x_cond_2ch)

    for tau_i in range(cfg.ns, 0, -1):
        tau = torch.full((b,), tau_i, device=device, dtype=torch.long)
        a_t = alpha[tau_i]
        a_prev = alpha[tau_i - 1]

        # Paper notation beta_tau = alpha_tau / alpha_{tau-1}.
        beta_ret = a_t / a_prev
        eps = model(x_t, tau, x_cond_2ch)

        # Eq. (7):
        # mu = 1/sqrt(beta) * X_t
        #    + (beta-1)/sqrt(beta*(1-alpha_t)) * eps_theta(...)
        denom = torch.sqrt(torch.clamp(beta_ret * (1.0 - a_t), min=1e-12))
        mu = x_t / torch.sqrt(beta_ret) + ((beta_ret - 1.0) / denom) * eps

        # Paper's approximation:
        # p_hat(X_{t-1}|X_t,X) = CN(mu_hat, (1-alpha_t) I).
        if tau_i == 1 and not cfg.sample_final_step:
            x_t = mu
        else:
            z = complex_standard_normal_like(x_t)
            x_t = mu + torch.sqrt(torch.clamp(1.0 - a_t, min=0.0)) * z

    return x_t


# -----------------------------------------------------------------------------
# 1-D ESPRIT and requested metric
# -----------------------------------------------------------------------------

def esprit_ula(x_mt: np.ndarray, k: int, angle_convention: str = "sin") -> np.ndarray:
    """
    Standard 1-D ULA ESPRIT for half-wavelength spacing.

    x_mt: (M,T) complex denoised snapshot matrix.
    Returns K angles in degrees.

    Default steering convention:
      a_m(theta) = exp(-j*pi*m*sin(theta))
    """
    m, t = x_mt.shape
    if k >= m:
        raise ValueError("ESPRIT requires num_sources < num_antennas")

    r = (x_mt @ x_mt.conj().T) / float(t)
    # Hermitian eigendecomposition, ascending eigenvalues.
    evals, evecs = np.linalg.eigh(r)
    us = evecs[:, -k:]

    u1 = us[:-1, :]
    u2 = us[1:, :]
    psi = np.linalg.pinv(u1) @ u2
    lam = np.linalg.eigvals(psi)
    phase = np.angle(lam)

    # For exp(-j*pi*m*f(theta)), eigenphase = -pi*f(theta).
    u = np.clip(-phase / np.pi, -1.0, 1.0)

    if angle_convention == "sin":
        theta = np.arcsin(u)
    elif angle_convention == "cos":
        theta = np.arccos(u)
    else:
        raise ValueError("angle_convention must be 'sin' or 'cos'")

    return np.sort(np.rad2deg(theta).astype(np.float64))


def matched_rmse_deg(pred_deg: np.ndarray, true_rad: np.ndarray) -> Tuple[float, np.ndarray]:
    true_deg = np.rad2deg(np.asarray(true_rad, dtype=np.float64).reshape(-1))
    pred_deg = np.asarray(pred_deg, dtype=np.float64).reshape(-1)
    if len(pred_deg) != len(true_deg):
        raise ValueError("Prediction/target source counts differ")

    cost = (pred_deg[:, None] - true_deg[None, :]) ** 2
    rows, cols = linear_sum_assignment(cost)
    ordered_pred = np.empty_like(true_deg)
    ordered_pred[cols] = pred_deg[rows]
    rmse = float(np.sqrt(np.mean((ordered_pred - true_deg) ** 2)))
    return rmse, ordered_pred


def requested_db_metric(rmse_deg: np.ndarray) -> float:
    """Exactly requested: mean_i [10*log10(RMSE_i_in_degrees)]."""
    r = np.asarray(rmse_deg, dtype=np.float64)
    return float(np.mean(10.0 * np.log10(np.maximum(r, 1e-12))))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    x_test: np.ndarray,
    az_test: np.ndarray,
    alpha: torch.Tensor,
    cfg: Config,
    device: torch.device,
    output_npz: Optional[str] = None,
):
    """
    x_test: (M,T,SNR,Ntest) complex
    az_test: (Ntest,K) radians
    """
    m, t, n_snr, n_test = x_test.shape
    assert (m, t, n_snr, n_test) == (
        cfg.num_antennas,
        cfg.num_snapshots,
        cfg.num_snr,
        cfg.test_setups,
    )

    all_rmse = np.empty((n_snr, n_test), dtype=np.float64)
    all_pred = np.empty((n_snr, n_test, cfg.num_sources), dtype=np.float64)

    for si, snr in enumerate(cfg.snr_db):
        t_snr = time.time()
        print(f"[eval] SNR {snr:+g} dB")

        for s0 in range(0, n_test, cfg.eval_setup_batch):
            s1 = min(s0 + cfg.eval_setup_batch, n_test)
            bsetup = s1 - s0

            # (M,T,B) -> (B,T,M) -> (B*T,M) snapshot batch.
            xb = np.transpose(x_test[:, :, si, s0:s1], (2, 1, 0)).reshape(-1, m)
            xb2 = torch.from_numpy(complex_np_to_2ch(xb)).to(device, non_blocking=True)

            x0hat2 = reverse_denoise(model, xb2, alpha, cfg)
            x0hat = twoch_to_complex_np(x0hat2.detach().cpu().numpy())
            x0hat = x0hat.reshape(bsetup, t, m)

            for bi in range(bsetup):
                # ESPRIT expects M x T.
                x_mt = x0hat[bi].T
                pred = esprit_ula(x_mt, cfg.num_sources, cfg.angle_convention)
                rmse, pred_matched = matched_rmse_deg(pred, az_test[s0 + bi])
                all_rmse[si, s0 + bi] = rmse
                all_pred[si, s0 + bi] = pred_matched

            if s0 == 0 or s1 == n_test or (s1 % max(cfg.eval_setup_batch * 5, 1) == 0):
                print(f"       processed {s1:4d}/{n_test} setups")

        mean_rmse = float(np.mean(all_rmse[si]))
        metric = requested_db_metric(all_rmse[si])
        print(
            f"       mean RMSE = {mean_rmse:.6f} deg | "
            f"mean[10log10(sample RMSE)] = {metric:.6f} dB | "
            f"time={time.time()-t_snr:.1f}s"
        )

    overall_metric = requested_db_metric(all_rmse.reshape(-1))
    overall_rmse = float(np.mean(all_rmse))

    print("\n================ FINAL RESULTS ================")
    print("SNR(dB) | mean RMSE (deg) | average 10log10(single-sample RMSE)")
    print("--------+-----------------+-------------------------------------")
    for si, snr in enumerate(cfg.snr_db):
        print(
            f"{snr:>+7.1f} | {np.mean(all_rmse[si]):>15.6f} | "
            f"{requested_db_metric(all_rmse[si]):>35.6f} dB"
        )
    print("--------+-----------------+-------------------------------------")
    print(f"OVERALL | {overall_rmse:>15.6f} | {overall_metric:>35.6f} dB")
    print("=================================================")
    print(
        "Requested final metric = average over test samples of "
        "10*log10(RMSE_i_deg), where each RMSE_i uses the 4 matched DOAs."
    )

    if output_npz:
        np.savez_compressed(
            output_npz,
            rmse_deg=all_rmse,
            pred_deg=all_pred,
            true_rad=az_test,
            snr_db=np.asarray(cfg.snr_db),
            final_metric_db=np.asarray(overall_metric),
        )
        print(f"[eval] saved detailed results: {output_npz}")

    return all_rmse, all_pred, overall_metric


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mat", required=True, help="Path to MATLAB file")
    p.add_argument("--mode", choices=["train_eval", "train", "eval"], default="train_eval")
    p.add_argument("--checkpoint", default="diffusion_doa_paper.pt")
    p.add_argument("--results", default="diffusion_doa_results.npz")
    p.add_argument("--resume", action="store_true")

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ns", type=int, default=20)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--schedule", choices=["cosine", "linear_alpha"], default="cosine")
    p.add_argument("--alpha-start", type=float, default=0.999)
    p.add_argument("--alpha-end", type=float, default=0.001)
    p.add_argument("--angle-convention", choices=["sin", "cos"], default="sin")

    p.add_argument("--train-setups", type=int, default=4000)
    p.add_argument("--test-setups", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--eval-setup-batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--no-final-noise", action="store_true", help="Use mu instead of sampling at t=1 (not paper-exact)")
    p.add_argument("--log-every", type=int, default=500)
    return p.parse_args()



def apply_checkpoint_model_diffusion_config(cfg: Config, checkpoint_path: str, device: torch.device):
    """
    For evaluation, restore architecture + diffusion-schedule parameters from
    the training checkpoint before model construction.

    Parameters restored:
      - base_channels
      - ns
      - schedule
      - alpha_start
      - alpha_end

    Other runtime/evaluation choices (test_setups, eval batch size, angle
    convention, etc.) remain controlled by the current command line.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location=device)
    saved = ckpt.get("config", None)
    if saved is None:
        raise KeyError(
            "Checkpoint has no saved 'config'. Cannot safely reconstruct "
            "the model architecture/diffusion schedule for evaluation."
        )

    restored = {}
    for key in ("base_channels", "ns", "schedule", "alpha_start", "alpha_end"):
        if key not in saved:
            raise KeyError(
                f"Checkpoint config is missing required evaluation parameter '{key}'."
            )
        old = getattr(cfg, key)
        new = saved[key]
        setattr(cfg, key, new)
        restored[key] = (old, new)

    print("[eval] restored architecture/diffusion configuration from checkpoint:")
    for key, (old, new) in restored.items():
        if old == new:
            print(f"       {key}: {new}")
        else:
            print(f"       {key}: {old} -> {new}  [checkpoint]")

    return ckpt

def main():
    args = parse_args()

    cfg = Config(
        train_setups=args.train_setups,
        test_setups=args.test_setups,
        ns=args.ns,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        base_channels=args.base_channels,
        schedule=args.schedule,
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        angle_convention=args.angle_convention,
        seed=args.seed,
        num_workers=args.num_workers,
        amp=args.amp,
        deterministic=args.deterministic,
        eval_setup_batch=args.eval_setup_batch,
        sample_final_step=not args.no_final_noise,
        log_every=args.log_every,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # In pure evaluation mode, restore the architecture and diffusion schedule
    # from the checkpoint BEFORE constructing the network and alpha schedule.
    eval_ckpt = None
    if args.mode == "eval":
        eval_ckpt = apply_checkpoint_model_diffusion_config(
            cfg, args.checkpoint, device
        )

    seed_everything(cfg.seed, cfg.deterministic)

    print(f"[system] PyTorch {torch.__version__}, device={device}")
    if device.type == "cuda":
        print(f"[system] GPU: {torch.cuda.get_device_name(0)}")
    print("[config] " + json.dumps(asdict(cfg), indent=2))

    # Load train/test arrays. Training data contain all seven real SNR points.
    x_train, x0_train, x_test, az_test = prepare_arrays(args.mat, cfg)

    model = ParallelConditionalUNet1D(cfg.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] parameters: {n_params:,}")

    alpha = make_alpha_schedule(cfg, device)

    if args.mode in ("train", "train_eval"):
        ds = AllSNRSnapshotPairs(x_train, x0_train)
        expected = cfg.train_setups * cfg.num_snapshots * cfg.num_snr
        if len(ds) != expected:
            raise RuntimeError(
                f"Training dataset length {len(ds):,} != expected {expected:,}"
            )

        loader = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            persistent_workers=(cfg.num_workers > 0),
        )
        train_model(model, loader, alpha, cfg, device, args.checkpoint, args.resume)

    if args.mode in ("eval", "train_eval"):
        if args.mode == "eval":
            # eval_ckpt was already loaded above so that its config could be used
            # to construct the correct model and diffusion schedule.
            model.load_state_dict(eval_ckpt["model"])
            print(f"[eval] loaded checkpoint weights: {args.checkpoint}")

        evaluate(model, x_test, az_test, alpha, cfg, device, args.results)


if __name__ == "__main__":
    main()