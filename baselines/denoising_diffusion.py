from __future__ import annotations
import argparse
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple
import numpy as np
try:
    import scipy.io as sio
    from scipy.optimize import linear_sum_assignment
except Exception as exc:
    raise RuntimeError('This script requires scipy: pip install scipy') from exc
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception as exc:
    raise RuntimeError('This script requires PyTorch: https://pytorch.org/get-started/') from exc
try:
    import h5py
except Exception:
    h5py = None

@dataclass
class Config:
    num_antennas: int = 16
    num_snapshots: int = 50
    num_snr: int = 7
    num_setups: int = 5000
    num_sources: int = 4
    snr_db: Tuple[float, ...] = (-20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0)
    train_setups: int = 4000
    test_setups: int = 1000
    ns: int = 20
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 0.0001
    base_channels: int = 128
    schedule: str = 'cosine'
    alpha_start: float = 0.999
    alpha_end: float = 0.001
    angle_convention: str = 'sin'
    seed: int = 1234
    num_workers: int = 0
    amp: bool = False
    deterministic: bool = False
    log_every: int = 500
    eval_setup_batch: int = 64
    sample_final_step: bool = True

def _to_complex(x):
    if isinstance(x, np.ndarray) and x.dtype.fields:
        names = set(x.dtype.fields.keys())
        if {'real', 'imag'}.issubset(names):
            return x['real'] + 1j * x['imag']
    return x

def _read_h5(path: str, var_name: str) -> np.ndarray:
    if h5py is None:
        raise RuntimeError('MAT-file appears to be v7.3/HDF5, but h5py is not installed.')
    with h5py.File(path, 'r') as f:
        if var_name not in f:
            raise KeyError(f"Variable '{var_name}' not found in {path}. Keys: {list(f.keys())}")
        obj = f[var_name]
        if isinstance(obj, h5py.Group):
            keys = set(obj.keys())
            if {'real', 'imag'}.issubset(keys):
                arr = np.asarray(obj['real']) + 1j * np.asarray(obj['imag'])
            else:
                raise RuntimeError(f"Unsupported HDF5 group format for variable '{var_name}': {keys}")
        else:
            arr = np.asarray(obj)
            arr = _to_complex(arr)
    return arr

def load_mat_var(path: str, var_name: str) -> np.ndarray:
    try:
        d = sio.loadmat(path, variable_names=[var_name], squeeze_me=False, struct_as_record=False)
        if var_name not in d:
            raise KeyError(f"Variable '{var_name}' not found in {path}")
        return np.asarray(d[var_name])
    except (NotImplementedError, ValueError, OSError):
        return _read_h5(path, var_name)

def format_signal_array(a: np.ndarray, config: Config, name: str) -> np.ndarray:
    target = (config.num_antennas, config.num_snapshots, config.num_snr, config.num_setups)
    a = np.asarray(a)
    if a.shape == target:
        out = a
    elif a.shape == target[::-1]:
        out = np.transpose(a, (3, 2, 1, 0))
    else:
        shape = list(a.shape)
        if len(shape) != 4 or sorted(shape) != sorted(target):
            raise ValueError(f'{name} has shape {a.shape}, expected a permutation of {target}')
        perm = [shape.index(v) for v in target]
        out = np.transpose(a, perm)
    if not np.iscomplexobj(out):
        raise ValueError(f'{name} is not complex-valued after loading (dtype={out.dtype}).')
    return np.asarray(out, dtype=np.complex64)

def format_angles(a: np.ndarray, config: Config) -> np.ndarray:
    a = np.asarray(a)
    a = np.squeeze(a)
    if a.shape == (config.num_sources, config.num_setups):
        out = a.T
    elif a.shape == (config.num_setups, config.num_sources):
        out = a
    else:
        raise ValueError(f'target_azimuth becomes shape {a.shape} after squeeze; expected ({config.num_sources},{config.num_setups}) or ({config.num_setups},{config.num_sources}).')
    return np.asarray(out, dtype=np.float64)

def get_clean_reference(clean: np.ndarray, config: Config) -> np.ndarray:
    zero_db_indices = np.where(np.isclose(np.asarray(config.snr_db), 0.0))[0]
    if len(zero_db_indices) != 1:
        raise ValueError(f'Expected exactly one 0 dB entry in config.snr_db, got {config.snr_db}')
    idx = int(zero_db_indices[0])
    if idx != 2:
        raise ValueError(f'0 dB is at Python index {idx}, but expected index 2 (MATLAB index 3) for this dataset.')
    return clean[:, :, idx, :]

def load_dataset(mat_path: str, config: Config):
    if config.train_setups + config.test_setups > config.num_setups:
        raise ValueError('train_setups + test_setups exceeds num_setups')
    test_start = config.num_setups - config.test_setups
    if config.train_setups > test_start:
        raise ValueError('Training and test setup ranges overlap.')
    y_all = format_signal_array(load_mat_var(mat_path, 'y_receive'), config, 'y_receive')
    x0_all = format_signal_array(load_mat_var(mat_path, 'y_receive_ultra_clean'), config, 'y_receive_ultra_clean')
    az_all = format_angles(load_mat_var(mat_path, 'target_azimuth'), config)
    x0_train = x0_all[:, :, :, :config.train_setups].copy()
    x_train = y_all[:, :, :, :config.train_setups].copy()
    x_test = y_all[:, :, :, test_start:].copy()
    az_test = az_all[test_start:].copy()
    del y_all, x0_all, az_all
    expected = config.train_setups * config.num_snapshots * config.num_snr
    if x_train.shape != (config.num_antennas, config.num_snapshots, config.num_snr, config.train_setups):
        raise RuntimeError(f'Unexpected training-array shape: {x_train.shape}')
    return (x_train, x0_train, x_test, az_test)

def complex_to_channels(x: np.ndarray) -> np.ndarray:
    return np.stack((x.real, x.imag), axis=-2).astype(np.float32, copy=False)

def channels_to_complex(x: np.ndarray) -> np.ndarray:
    return (x[..., 0, :] + 1j * x[..., 1, :]).astype(np.complex64, copy=False)

def complex_noise_like(x: torch.Tensor) -> torch.Tensor:
    return torch.randn_like(x) / math.sqrt(2.0)

class SnapshotDataset(Dataset):
    def __init__(self, x_cond_complex: np.ndarray, x0_complex: np.ndarray):
        if x_cond_complex.ndim != 4:
            raise ValueError(f'x_cond_complex must have shape (M,T,S,N), got {x_cond_complex.shape}')
        m, t, s, n = x_cond_complex.shape
        if x0_complex.shape != (m, t, s, n):
            raise ValueError(f'condition/clean geometry mismatch: {x_cond_complex.shape} vs {x0_complex.shape}')
        self.x = x_cond_complex
        self.clean = x0_complex
        self.m = m
        self.t = t
        self.s = s
        self.n = n
        self.samples_per_setup = t * s
    def __len__(self):
        return self.n * self.samples_per_setup
    def __getitem__(self, idx):
        setup = idx // self.samples_per_setup
        rem = idx % self.samples_per_setup
        snr = rem // self.t
        snap = rem % self.t
        x = self.x[:, snap, snr, setup]
        clean = self.clean[:, snap, snr, setup]
        x_2ch = np.stack((x.real, x.imag), axis=0).astype(np.float32, copy=False)
        x0_2ch = np.stack((clean.real, clean.imag), axis=0).astype(np.float32, copy=False)
        return (torch.from_numpy(x_2ch), torch.from_numpy(x0_2ch))

def make_schedule(config: Config, device: torch.device) -> torch.Tensor:
    ns = config.ns
    alpha = np.ones(ns + 1, dtype=np.float64)
    if config.schedule == 'linear_alpha':
        alpha[1:] = np.linspace(config.alpha_start, config.alpha_end, ns, dtype=np.float64)
    elif config.schedule == 'cosine':
        s = 0.008
        t = np.arange(ns + 1, dtype=np.float64)
        f = np.cos((t / ns + s) / (1.0 + s) * math.pi / 2.0) ** 2
        alpha_raw = f / f[0]
        var_beta = 1.0 - alpha_raw[1:] / alpha_raw[:-1]
        var_beta = np.clip(var_beta, 1e-08, 0.999)
        alpha[0] = 1.0
        alpha[1:] = np.cumprod(1.0 - var_beta)
    else:
        raise ValueError(f'Unknown schedule: {config.schedule}')
    if not np.all(np.diff(alpha) < 0):
        raise RuntimeError('alpha_t must be strictly decreasing')
    if not 0.0 < alpha[-1] < alpha[1] < 1.0:
        raise RuntimeError(f'Invalid alpha schedule endpoints: alpha_1={alpha[1]}, alpha_N={alpha[-1]}')
    out = torch.tensor(alpha, dtype=torch.float32, device=device)
    return out

class SinusoidalStepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        if half == 0:
            return t.float().unsqueeze(1)
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb

class ActConv1d(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int=3, stride: int=1, padding: int=1):
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

class DiffusionUNet(nn.Module):
    def __init__(self, base: int=64):
        super().__init__()
        c = base
        self.main_in = ActConv1d(2, c, 3, 1, 1)
        self.cond_in = ActConv1d(2, c, 3, 1, 1)
        self.t_embed = SinusoidalStepEmbedding(c)
        self.cond_time_fuse = ActConv1d(2 * c, c, 1, 1, 0)
        self.main_down1 = ActConv1d(c, 2 * c, 4, 2, 1)
        self.main_down2 = ActConv1d(2 * c, 4 * c, 4, 2, 1)
        self.cond_down1 = ActConv1d(c, 2 * c, 4, 2, 1)
        self.cond_down2 = ActConv1d(2 * c, 4 * c, 4, 2, 1)
        self.main_up1 = ActConvTranspose1d(4 * c, 2 * c)
        self.main_up2 = ActConvTranspose1d(2 * c, c)
        self.cond_up1 = ActConvTranspose1d(4 * c, 2 * c)
        self.cond_up2 = ActConvTranspose1d(2 * c, c)
        self.out = nn.Conv1d(c, 2, kernel_size=3, padding=1)
    def forward(self, state: torch.Tensor, t: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        c0_x = self.cond_in(noisy)
        te = self.t_embed(t).unsqueeze(-1).expand(-1, -1, c0_x.shape[-1])
        c0 = self.cond_time_fuse(torch.cat([c0_x, te], dim=1))
        m0 = self.main_in(state) + c0
        c1 = self.cond_down1(c0)
        m1 = self.main_down1(m0) + c1
        c2 = self.cond_down2(c1)
        m2 = self.main_down2(m1) + c2
        c1u = self.cond_up1(c2) + c1
        m1u = self.main_up1(m2) + m1 + c1u
        c0u = self.cond_up2(c1u) + c0
        m0u = self.main_up2(m1u) + m0 + c0u
        return self.out(m0u)

def set_seed(seed: int, deterministic: bool=False):
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

def train(model: nn.Module, loader: DataLoader, alpha: torch.Tensor, config: Config, device: torch.device, checkpoint_path: str, resume: bool):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp and device.type == 'cuda')
    start_epoch = 1
    if resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = int(ckpt.get('epoch', 0)) + 1
    ns = config.ns
    model.train()
    global_step = 0

    for epoch in range(start_epoch, config.epochs + 1):
        t0 = time.time()
        loss_sum = 0.0
        n_seen = 0
        for step, (noisy, clean) in enumerate(loader, start=1):
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            b = clean.shape[0]
            tau = torch.randint(1, ns + 1, (b,), device=device, dtype=torch.long)
            a = alpha[tau].view(b, 1, 1)
            z = complex_noise_like(clean)
            diffused = torch.sqrt(a) * clean + torch.sqrt(1.0 - a) * z
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=config.amp and device.type == 'cuda'):
                noise_hat = model(diffused, tau, noisy)
                loss = F.mse_loss(noise_hat, z, reduction='mean')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * b
            n_seen += b
            global_step += 1
        epoch_loss = loss_sum / max(n_seen, 1)
        dt = time.time() - t0
        print(f'[train] epoch {epoch:03d}/{config.epochs} loss={epoch_loss:.8f} time={dt:.1f}s')
        ckpt = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': asdict(config)}
        torch.save(ckpt, checkpoint_path)

@torch.no_grad()
def denoise(model: nn.Module, x_cond_2ch: torch.Tensor, alpha: torch.Tensor, config: Config) -> torch.Tensor:
    device = x_cond_2ch.device
    b = x_cond_2ch.shape[0]
    model.eval()
    state = complex_noise_like(x_cond_2ch)
    for tau_i in range(config.ns, 0, -1):
        tau = torch.full((b,), tau_i, device=device, dtype=torch.long)
        a_t = alpha[tau_i]
        a_prev = alpha[tau_i - 1]
        beta_ret = a_t / a_prev
        eps = model(state, tau, x_cond_2ch)
        denom = torch.sqrt(torch.clamp(beta_ret * (1.0 - a_t), min=1e-12))
        mu = state / torch.sqrt(beta_ret) + (beta_ret - 1.0) / denom * eps
        if tau_i == 1 and (not config.sample_final_step):
            state = mu
        else:
            z = complex_noise_like(state)
            state = mu + torch.sqrt(torch.clamp(1.0 - a_t, min=0.0)) * z
    return state

def esprit_ula(x_mt: np.ndarray, k: int, angle_convention: str='sin') -> np.ndarray:
    m, t = x_mt.shape
    if k >= m:
        raise ValueError('ESPRIT requires num_sources < num_antennas')
    r = x_mt @ x_mt.conj().T / float(t)
    evals, evecs = np.linalg.eigh(r)
    us = evecs[:, -k:]
    u1 = us[:-1, :]
    u2 = us[1:, :]
    psi = np.linalg.pinv(u1) @ u2
    lam = np.linalg.eigvals(psi)
    phase = np.angle(lam)
    u = np.clip(-phase / np.pi, -1.0, 1.0)
    if angle_convention == 'sin':
        theta = np.arcsin(u)
    elif angle_convention == 'cos':
        theta = np.arccos(u)
    else:
        raise ValueError("angle_convention must be 'sin' or 'cos'")
    return np.sort(np.rad2deg(theta).astype(np.float64))

def match_rmse(pred_deg: np.ndarray, true_rad: np.ndarray) -> Tuple[float, np.ndarray]:
    true_deg = np.rad2deg(np.asarray(true_rad, dtype=np.float64).reshape(-1))
    pred_deg = np.asarray(pred_deg, dtype=np.float64).reshape(-1)
    if len(pred_deg) != len(true_deg):
        raise ValueError('Prediction/target source counts differ')
    cost = (pred_deg[:, None] - true_deg[None, :]) ** 2
    rows, cols = linear_sum_assignment(cost)
    ordered_pred = np.empty_like(true_deg)
    ordered_pred[cols] = pred_deg[rows]
    rmse = float(np.sqrt(np.mean((ordered_pred - true_deg) ** 2)))
    return (rmse, ordered_pred)

def rmse_db(rmse_deg: np.ndarray) -> float:
    r = np.asarray(rmse_deg, dtype=np.float64)
    return float(np.mean(10.0 * np.log10(np.maximum(r, 1e-12))))

def esprit_covariance(X, diag_loading=0.001, forward_backward=True, spatial_smoothing=True):
    M, L = X.shape
    if spatial_smoothing:
        P = M // 2 + 1
        n_subarrays = M - P + 1
        R = np.zeros((P, P), dtype=np.complex128)
        for i in range(n_subarrays):
            Xi = X[i:i + P, :]
            R += Xi @ Xi.conj().T / L
        R /= n_subarrays
        M_eff = P
    else:
        R = X @ X.conj().T / L
        M_eff = M
    R = 0.5 * (R + R.conj().T)
    if forward_backward:
        J = np.fliplr(np.eye(M_eff))
        R = 0.5 * (R + J @ R.conj() @ J)
    scale = np.trace(R).real / M_eff
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    R += diag_loading * scale * np.eye(M_eff)
    return R

def esprit_doa(X, K, angle_convention='sin'):
    R = esprit_covariance(X, diag_loading=0.001, forward_backward=True, spatial_smoothing=True)
    M = R.shape[0]
    if K >= M:
        raise ValueError('K must be smaller than effective antenna count')
    eigvals, eigvecs = np.linalg.eigh(R)
    Es = eigvecs[:, -K:]
    Es1 = Es[:-1, :]
    Es2 = Es[1:, :]
    Z = np.hstack((Es1, Es2))
    _, _, Vh = np.linalg.svd(Z, full_matrices=False)
    V = Vh.conj().T
    V12 = V[:K, K:]
    V22 = V[K:, K:]
    Psi = -V12 @ np.linalg.inv(V22)
    mu, _ = np.linalg.eig(Psi)
    phase = np.angle(mu)
    sin_theta = phase / (2 * np.pi * 0.5)
    sin_theta = np.clip(sin_theta, -1, 1)
    theta = np.arcsin(sin_theta)
    return np.sort(theta)

@torch.no_grad()
def evaluate_model(model: nn.Module, x_test: np.ndarray, az_test: np.ndarray, alpha: torch.Tensor, config: Config, device: torch.device, output_npz: Optional[str]=None):
    m, t, n_snr, n_test = x_test.shape
    assert (m, t, n_snr, n_test) == (config.num_antennas, config.num_snapshots, config.num_snr, config.test_setups)
    all_rmse = np.empty((n_snr, n_test), dtype=np.float64)
    all_pred = np.empty((n_snr, n_test, config.num_sources), dtype=np.float64)

    for si, snr in enumerate(config.snr_db):
        t_snr = time.time()
        for s0 in range(0, n_test, config.eval_setup_batch):
            s1 = min(s0 + config.eval_setup_batch, n_test)
            bsetup = s1 - s0
            xb = np.transpose(x_test[:, :, si, s0:s1], (2, 1, 0)).reshape(-1, m)
            xb2 = torch.from_numpy(complex_to_channels(xb)).to(device, non_blocking=True)
            x0hat2 = denoise(model, xb2, alpha, config)
            x0hat = channels_to_complex(x0hat2.detach().cpu().numpy())
            x0hat = x0hat.reshape(bsetup, t, m)
            for bi in range(bsetup):
                x_mt = x0hat[bi].T
                pred = np.rad2deg(esprit_doa(x_mt.astype(np.complex128), config.num_sources))
                rmse, pred_matched = match_rmse(pred, az_test[s0 + bi])
                all_rmse[si, s0 + bi] = rmse
                all_pred[si, s0 + bi] = pred_matched
        mean_rmse = float(np.mean(all_rmse[si]))
        metric = rmse_db(all_rmse[si])
    overall_metric = rmse_db(all_rmse.reshape(-1))
    overall_rmse = float(np.mean(all_rmse))
    print('\n================ FINAL RESULTS ================')
    print('SNR(dB) | mean RMSE (deg) | average 10log10(single-sample RMSE)')
    print('--------+-----------------+-------------------------------------')

    for si, snr in enumerate(config.snr_db):
        print(f'{snr:>+7.1f} | {np.mean(all_rmse[si]):>15.6f} | {rmse_db(all_rmse[si]):>35.6f} dB')
    print('--------+-----------------+-------------------------------------')
    print(f'OVERALL | {overall_rmse:>15.6f} | {overall_metric:>35.6f} dB')
    print('=================================================')
    print('Requested final metric = average over test samples of 10*log10(RMSE_i_deg), where each RMSE_i uses the 4 matched DOAs.')
    if output_npz:
        np.savez_compressed(output_npz, rmse_deg=all_rmse, pred_deg=all_pred, true_rad=az_test, snr_db=np.asarray(config.snr_db), final_metric_db=np.asarray(overall_metric))
    return (all_rmse, all_pred, overall_metric)

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--mat', required=True, help='Path to MATLAB file')
    p.add_argument('--mode', choices=['train_eval', 'train', 'eval'], default='train_eval')
    p.add_argument('--checkpoint', default='diffusion_doa_paper.pt')
    p.add_argument('--results', default='diffusion_doa_results.npz')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--ns', type=int, default=20)
    p.add_argument('--base-channels', type=int, default=128)
    p.add_argument('--schedule', choices=['cosine', 'linear_alpha'], default='cosine')
    p.add_argument('--alpha-start', type=float, default=0.999)
    p.add_argument('--alpha-end', type=float, default=0.001)
    p.add_argument('--angle-convention', choices=['sin', 'cos'], default='sin')
    p.add_argument('--train-setups', type=int, default=4000)
    p.add_argument('--test-setups', type=int, default=1000)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--eval-setup-batch', type=int, default=64)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--no-final-noise', action='store_true', help='Use mu instead of sampling at t=1 (not paper-exact)')
    p.add_argument('--log-every', type=int, default=500)
    return p.parse_args()

def load_checkpoint_config(config: Config, checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    saved = ckpt.get('config', None)
    if saved is None:
        raise KeyError("Checkpoint has no saved 'config'. Cannot safely reconstruct the model architecture/diffusion schedule for evaluation.")
    restored = {}
    for key in ('base_channels', 'ns', 'schedule', 'alpha_start', 'alpha_end'):
        if key not in saved:
            raise KeyError(f"Checkpoint config is missing required evaluation parameter '{key}'.")
        old = getattr(config, key)
        new = saved[key]
        setattr(config, key, new)
        restored[key] = (old, new)
    return ckpt

def main():
    args = parse_args()
    config = Config(train_setups=args.train_setups, test_setups=args.test_setups, ns=args.ns, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, base_channels=args.base_channels, schedule=args.schedule, alpha_start=args.alpha_start, alpha_end=args.alpha_end, angle_convention=args.angle_convention, seed=args.seed, num_workers=args.num_workers, amp=args.amp, deterministic=args.deterministic, eval_setup_batch=args.eval_setup_batch, sample_final_step=not args.no_final_noise, log_every=args.log_every)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_ckpt = None
    if args.mode == 'eval':
        eval_ckpt = load_checkpoint_config(config, args.checkpoint, device)
    set_seed(config.seed, config.deterministic)
    x_train, x0_train, x_test, az_test = load_dataset(args.mat, config)
    model = DiffusionUNet(config.base_channels).to(device)
    n_params = sum((p.numel() for p in model.parameters()))
    alpha = make_schedule(config, device)
    if args.mode in ('train', 'train_eval'):
        ds = SnapshotDataset(x_train, x0_train)
        expected = config.train_setups * config.num_snapshots * config.num_snr
        if len(ds) != expected:
            raise RuntimeError(f'Training dataset length {len(ds):,} != expected {expected:,}')
        loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=device.type == 'cuda', drop_last=False, persistent_workers=config.num_workers > 0)
        train(model, loader, alpha, config, device, args.checkpoint, args.resume)
    if args.mode in ('eval', 'train_eval'):
        if args.mode == 'eval':
            model.load_state_dict(eval_ckpt['model'])
        evaluate_model(model, x_test, az_test, alpha, config, device, args.results)
if __name__ == '__main__':
    main()
