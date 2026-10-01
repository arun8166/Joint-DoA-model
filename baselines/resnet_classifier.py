import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    import h5py
except ImportError:
    h5py = None
try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None
NUM_ANTS = 16
NUM_SNAPS = 50
NUM_SNRS = 7
NUM_TARGETS = 4
SNR_DB = (-20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0)
GRID_STEP_DEG = 0.1
ANGLE_MIN_DEG = -90.0
ANGLE_MAX_DEG = 90.0
DEFAULT_EPOCHS = 200
DEFAULT_BATCH = 32
DEFAULT_LR = 0.001
RMSE_FLOOR = 1e-12

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _to_complex(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return arr
    if arr.dtype.names is not None:
        names = set(arr.dtype.names)
        if {'real', 'imag'}.issubset(names):
            return arr['real'] + 1j * arr['imag']
        if {'r', 'i'}.issubset(names):
            return arr['r'] + 1j * arr['i']
    return arr

def normalize_target_shape(target: np.ndarray) -> np.ndarray:
    target = np.asarray(target)
    target = np.squeeze(target)
    if target.ndim != 2:
        raise ValueError(f'target_azimuth must become a 2-D array after squeeze; got {target.shape}')
    if target.shape[1] == NUM_TARGETS:
        out = target
    elif target.shape[0] == NUM_TARGETS:
        out = target.T
    else:
        raise ValueError(f'Could not convert target_azimuth to shape (N,4). Observed shape after squeeze: {target.shape}')
    out = np.asarray(out, dtype=np.float32)
    if not np.all(np.isfinite(out)):
        bad = np.argwhere(~np.isfinite(out))
        raise ValueError(f'target_azimuth contains non-finite values. First bad indices: {bad[:10]}')
    out = np.rad2deg(out).astype(np.float32)
    if out.shape[1] != NUM_TARGETS:
        raise ValueError(f'Expected 4 sources; got target shape {out.shape}')
    return out

class MatData:

    def __init__(self, path: str):
        self.path = str(path)
        self.mode = None
        self.h5 = None
        self.y = None
        self.targets = None
        self.y_shape = None
        self.axes = None
        if h5py is not None:
            try:
                self.h5 = h5py.File(self.path, 'r')
                if 'y_receive' not in self.h5 or 'target_azimuth' not in self.h5:
                    keys = list(self.h5.keys())
                    self.h5.close()
                    self.h5 = None
                    raise KeyError(f'Required variables y_receive and target_azimuth were not found. Top-level HDF5 keys: {keys}')
                self.mode = 'hdf5'
                self.y = self.h5['y_receive']
                target_raw = np.asarray(self.h5['target_azimuth'])
                target_raw = _to_complex(target_raw)
                if np.iscomplexobj(target_raw):
                    target_raw = target_raw.real
                self.targets = normalize_target_shape(target_raw)
                self.y_shape = tuple(self.y.shape)
                self._infer_axes()
                return
            except OSError:
                if self.h5 is not None:
                    self.h5.close()
                self.h5 = None
        if loadmat is None:
            raise RuntimeError('MAT file is not readable as HDF5 and scipy.io.loadmat is unavailable. Install scipy and h5py.')
        data = loadmat(self.path, variable_names=['y_receive', 'target_azimuth'])
        if 'y_receive' not in data or 'target_azimuth' not in data:
            raise KeyError('Required variables y_receive and target_azimuth were not found.')
        self.mode = 'scipy'
        self.y = _to_complex(data['y_receive'])
        target_raw = _to_complex(data['target_azimuth'])
        if np.iscomplexobj(target_raw):
            target_raw = target_raw.real
        self.targets = normalize_target_shape(target_raw)
        self.y_shape = tuple(self.y.shape)
        self._infer_axes()

    def _infer_axes(self) -> None:
        if len(self.y_shape) != 4:
            raise ValueError(f'y_receive must be 4-D. Observed shape: {self.y_shape}')
        n_setups = self.targets.shape[0]
        wanted = {'ant': NUM_ANTS, 'snap': NUM_SNAPS, 'snr': NUM_SNRS, 'setup': n_setups}
        axes = {}
        used = set()
        for name, size in wanted.items():
            candidates = [axis for axis, dim in enumerate(self.y_shape) if dim == size and axis not in used]
            if len(candidates) != 1:
                raise ValueError(f'Could not uniquely infer {name!r} axis of size {size} from y_receive shape {self.y_shape}. Candidates={candidates}')
            axes[name] = candidates[0]
            used.add(candidates[0])
        self.axes = axes

    @property
    def n_setups(self) -> int:
        return int(self.targets.shape[0])

    def get_setup_all_snrs(self, setup_id: int) -> np.ndarray:
        sl = [slice(None)] * 4
        sl[self.axes['setup']] = int(setup_id)
        raw = np.asarray(self.y[tuple(sl)])
        raw = _to_complex(raw)
        raw = np.squeeze(raw)
        if raw.ndim != 3:
            raise ValueError(f'After selecting one setup, y_receive must be 3-D; got {raw.shape}')
        remaining_original_axes = [axis for axis in range(4) if axis != self.axes['setup']]
        ant_pos = remaining_original_axes.index(self.axes['ant'])
        snap_pos = remaining_original_axes.index(self.axes['snap'])
        snr_pos = remaining_original_axes.index(self.axes['snr'])
        out = np.transpose(raw, (ant_pos, snap_pos, snr_pos))
        if out.shape != (NUM_ANTS, NUM_SNAPS, NUM_SNRS):
            raise ValueError(f'Expected setup data shape (16,50,7), got {out.shape}')
        if not np.iscomplexobj(out):
            raise ValueError("y_receive does not appear to be complex. The paper's correlation construction assumes complex array data.")
        return np.asarray(out)

    def close(self) -> None:
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

def make_corr_feature(y: np.ndarray) -> np.ndarray:
    if y.shape != (NUM_ANTS, NUM_SNAPS):
        raise ValueError(f'Expected y shape (16,50); got {y.shape}')
    r = y @ y.conj().T / float(NUM_SNAPS)
    feature = np.stack((r.real, r.imag), axis=0).astype(np.float32, copy=False)
    return feature

def _cache_is_valid(meta: dict, mat_file: str) -> bool:
    try:
        stat = os.stat(mat_file)
        return os.path.abspath(meta['source_file']) == os.path.abspath(mat_file) and int(meta['source_size']) == int(stat.st_size) and (abs(float(meta['source_mtime']) - float(stat.st_mtime)) < 1e-06) and (meta['feature_shape'][1:] == [NUM_SNRS, 2, NUM_ANTS, NUM_ANTS])
    except Exception:
        return False

def build_cache(mat_file: str, cache_dir: str, force: bool=False):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    feat_file = cache_dir / 'correlation_features.npy'
    angle_file = cache_dir / 'target_azimuth.npy'
    info_file = cache_dir / 'cache_meta.json'
    if not force and feat_file.exists() and angle_file.exists() and info_file.exists():
        try:
            with open(info_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            if _cache_is_valid(meta, mat_file):
                targets = np.load(angle_file)
                return (str(feat_file), targets, meta)
        except Exception:
            pass
    with MatData(mat_file) as reader:
        n_setups = reader.n_setups
        mmap = np.lib.format.open_memmap(feat_file, mode='w+', dtype=np.float32, shape=(n_setups, NUM_SNRS, 2, NUM_ANTS, NUM_ANTS))
        for setup_id in range(n_setups):
            setup_data = reader.get_setup_all_snrs(setup_id)
            for snr_id in range(NUM_SNRS):
                mmap[setup_id, snr_id] = make_corr_feature(setup_data[:, :, snr_id])
        mmap.flush()
        targets = reader.targets.astype(np.float32, copy=True)
        np.save(angle_file, targets)
        stat = os.stat(mat_file)
        meta = {'source_file': os.path.abspath(mat_file), 'source_size': int(stat.st_size), 'source_mtime': float(stat.st_mtime), 'feature_shape': [int(v) for v in mmap.shape], 'target_shape': [int(v) for v in targets.shape], 'formula': 'R_hat = Y @ Y^H / 50; channels=[real,imag]'}
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
    return (str(feat_file), targets, meta)

def build_angle_grid(targets: np.ndarray, resolution_deg: float, angle_min: Optional[float], angle_max: Optional[float]) -> np.ndarray:
    data_min = float(np.min(targets))
    data_max = float(np.max(targets))
    if angle_min is None and angle_max is None:
        if data_min >= ANGLE_MIN_DEG - 1e-06 and data_max <= ANGLE_MAX_DEG + 1e-06:
            angle_min = ANGLE_MIN_DEG
            angle_max = ANGLE_MAX_DEG
        else:
            angle_min = math.floor(data_min / resolution_deg) * resolution_deg
            angle_max = math.ceil(data_max / resolution_deg) * resolution_deg
    elif (angle_min is None) != (angle_max is None):
        raise ValueError('Specify both --angle-min and --angle-max, or specify neither.')
    angle_min = float(angle_min)
    angle_max = float(angle_max)
    if data_min < angle_min - resolution_deg / 2:
        raise ValueError(f'Target minimum {data_min} is below angle grid minimum {angle_min}.')
    if data_max > angle_max + resolution_deg / 2:
        raise ValueError(f'Target maximum {data_max} is above angle grid maximum {angle_max}.')
    n_classes_float = (angle_max - angle_min) / resolution_deg
    n_classes = int(round(n_classes_float)) + 1
    if not np.isclose(n_classes_float, round(n_classes_float), atol=1e-06):
        raise ValueError('Angular span must be an integer multiple of grid resolution.')
    grid = (angle_min + np.arange(n_classes, dtype=np.float64) * resolution_deg).astype(np.float32)
    return grid

def encode_multihot_targets(angles: np.ndarray, grid: np.ndarray) -> np.ndarray:
    n_setups = angles.shape[0]
    n_classes = len(grid)
    resolution = float(grid[1] - grid[0]) if len(grid) > 1 else 1.0
    grid_min = float(grid[0])
    labels = np.zeros((n_setups, n_classes), dtype=np.float32)
    for setup_id in range(n_setups):
        for theta in angles[setup_id]:
            idx = int(round((float(theta) - grid_min) / resolution))
            idx = max(0, min(n_classes - 1, idx))
            labels[setup_id, idx] = 1.0
        active = int(labels[setup_id].sum())
        if active != NUM_TARGETS:
            raise ValueError(f'Setup {setup_id} produces only {active} unique grid classes from angles {angles[setup_id].tolist()}. Use a finer grid resolution.')
    return labels

class CorrelationDataset(Dataset):

    def __init__(self, feat_file: str, labels: np.ndarray, truth: np.ndarray, setups: Sequence[int], snrs: Sequence[int]):
        self.features = np.load(feat_file, mmap_mode='r')
        self.labels = labels
        self.true_angles = truth
        setups = np.asarray(setups, dtype=np.int64)
        snrs = np.asarray(snrs, dtype=np.int64)
        self.pairs = np.asarray([(int(setup_id), int(snr_id)) for setup_id in setups for snr_id in snrs], dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.pairs))

    def __getitem__(self, item: int):
        setup_id, snr_id = self.pairs[item]
        x = np.array(self.features[setup_id, snr_id], dtype=np.float32, copy=True)
        label = np.array(self.labels[setup_id], dtype=np.float32, copy=True)
        angles = np.array(self.true_angles[setup_id], dtype=np.float32, copy=True)
        return (torch.from_numpy(x), torch.from_numpy(label), torch.from_numpy(angles), int(setup_id), int(snr_id))

class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out

class DOAResNet(nn.Module):

    def __init__(self, n_classes: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(2, 512, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.block1 = ResidualBlock(in_channels=512, out_channels=256, stride=1)
        self.block2 = ResidualBlock(in_channels=256, out_channels=128, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def doa_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()

def select_topk_angles(probs: np.ndarray, grid: np.ndarray, k: int=NUM_TARGETS, min_separation_deg: float=0.0) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    if min_separation_deg <= 0:
        idx = np.argpartition(probs, -k)[-k:]
        idx = idx[np.argsort(probs[idx])[::-1]]
        return np.sort(grid[idx].astype(np.float64))
    order = np.argsort(probs)[::-1]
    chosen = []
    for idx in order:
        theta = float(grid[idx])
        if all((abs(theta - prior) >= min_separation_deg for prior in chosen)):
            chosen.append(theta)
        if len(chosen) == k:
            break
    if len(chosen) != k:
        raise RuntimeError(f'Could not select {k} angles with separation {min_separation_deg} deg.')
    return np.sort(np.asarray(chosen, dtype=np.float64))

def rmse_deg(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.sort(np.asarray(pred, dtype=np.float64))
    true = np.sort(np.asarray(truth, dtype=np.float64))
    if pred.shape != (NUM_TARGETS,) or true.shape != (NUM_TARGETS,):
        raise ValueError(f'Expected four predicted/true angles; got {pred.shape}, {true.shape}')
    return float(np.sqrt(np.mean((pred - true) ** 2)))

def log_rmse_db(rmse_deg: float) -> float:
    safe_rmse = max(float(rmse_deg), RMSE_FLOOR)
    return float(10.0 * np.log10(safe_rmse))

def get_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=device.type == 'cuda', persistent_workers=num_workers > 0, drop_last=False)

def autocast_context(device: torch.device, enabled: bool):
    if device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.float16, enabled=enabled)
    return torch.autocast(device_type='cpu', enabled=False)

def train_one_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, epochs: int, lr: float, use_amp: bool, ckpt_file: Path, ckpt_info: dict):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp and device.type == 'cuda')
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == 'cuda')
    best_loss = float('inf')
    best_weights = None
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_sum = 0.0
        train_n = 0
        for x, y, _, _, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, use_amp):
                logits = model(x)
                loss = doa_loss(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bs = x.shape[0]
            train_sum += float(loss.detach().cpu()) * bs
            train_n += bs
        train_loss = train_sum / max(train_n, 1)
        model.eval()
        val_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for x, y, _, _, _ in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with autocast_context(device, use_amp):
                    logits = model(x)
                    loss = doa_loss(logits, y)
                bs = x.shape[0]
                val_sum += float(loss.detach().cpu()) * bs
                val_n += bs
        val_loss = val_sum / max(val_n, 1)
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        elapsed = time.time() - t0
        print(f'  epoch {epoch:03d}/{epochs} | train CE={train_loss:.6f} | val CE={val_loss:.6f} | best val={best_loss:.6f} | {elapsed / 60.0:.1f} min')
    payload = {'model_state': model.state_dict(), 'best_state': best_weights, 'optimizer_state': optimizer.state_dict(), 'best_val_loss': best_loss, 'meta': ckpt_info}
    torch.save(payload, ckpt_file)

@torch.no_grad()
def test_model(model: nn.Module, loader: DataLoader, device: torch.device, grid: np.ndarray, snr_db: np.ndarray, min_sep: float, use_amp: bool):
    model.eval()
    rows = []
    for x, _, truth, setup_id, snr_id in loader:
        x = x.to(device, non_blocking=True)
        with autocast_context(device, use_amp):
            logits = model(x)
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        true_angles_np = truth.numpy()
        setup_idx_np = np.asarray(setup_id)
        snr_idx_np = np.asarray(snr_id)
        for b in range(len(probs)):
            pred = select_topk_angles(probs=probs[b], grid=grid, k=NUM_TARGETS, min_separation_deg=min_sep)
            true = np.sort(np.asarray(true_angles_np[b], dtype=np.float64))
            rmse = rmse_deg(pred, true)
            metric = log_rmse_db(rmse)
            row = {'setup_idx': int(setup_idx_np[b]), 'snr_idx': int(snr_idx_np[b]), 'snr_db': float(snr_db[int(snr_idx_np[b])]), 'rmse_deg': float(rmse), 'metric_10log10_rmse': float(metric)}
            for i in range(NUM_TARGETS):
                row[f'true_{i + 1}_deg'] = float(true[i])
                row[f'pred_{i + 1}_deg'] = float(pred[i])
            rows.append(row)
    return rows

def write_prediction_csv(rows, path: Path) -> None:
    fieldnames = ['setup_idx', 'snr_idx', 'snr_db', 'true_1_deg', 'true_2_deg', 'true_3_deg', 'true_4_deg', 'pred_1_deg', 'pred_2_deg', 'pred_3_deg', 'pred_4_deg', 'rmse_deg', 'metric_10log10_rmse']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def summarize_results(rows, snr_db: np.ndarray, summary_path: Path):
    if not rows:
        raise RuntimeError('No test predictions were produced.')
    metric_all = np.asarray([r['metric_10log10_rmse'] for r in rows], dtype=np.float64)
    rmse_all = np.asarray([r['rmse_deg'] for r in rows], dtype=np.float64)
    final_metric = float(np.mean(metric_all))
    mean_rmse = float(np.mean(rmse_all))
    summary_rows = []
    for snr in snr_db:
        subset = [r for r in rows if np.isclose(r['snr_db'], snr)]
        if not subset:
            continue
        values = np.asarray([r['metric_10log10_rmse'] for r in subset], dtype=np.float64)
        rmses = np.asarray([r['rmse_deg'] for r in subset], dtype=np.float64)
        snr_metric = float(np.mean(values))
        snr_mean_rmse = float(np.mean(rmses))
        summary_rows.append({'snr_db': float(snr), 'samples': int(len(subset)), 'mean_rmse_deg': snr_mean_rmse, 'mean_10log10_rmse': snr_metric})
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['snr_db', 'samples', 'mean_rmse_deg', 'mean_10log10_rmse']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
        writer.writerow({'snr_db': 'OVERALL', 'samples': len(rows), 'mean_rmse_deg': mean_rmse, 'mean_10log10_rmse': final_metric})
    return final_metric

def discover_mat_file() -> str:
    mats = sorted(Path.cwd().glob('*.mat'))
    if len(mats) == 1:
        return str(mats[0])
    if len(mats) == 0:
        raise FileNotFoundError('No --mat-file was supplied and no .mat file exists in the current directory.')
    raise RuntimeError('No --mat-file was supplied and multiple .mat files exist in the current directory:\n  ' + '\n  '.join((str(p) for p in mats)))

def save_run_config(args, grid, split, out_dir: Path) -> None:
    config = {'arguments': vars(args), 'grid': {'min': float(grid[0]), 'max': float(grid[-1]), 'resolution': float(args.grid_resolution), 'classes': int(len(grid))}, 'split': {'train_setups': [int(x) for x in split['train']], 'val_setups': [int(x) for x in split['val']], 'test_setups': [int(x) for x in split['test']]}, 'requested_metric': 'mean_samples(10*log10(sqrt(mean_4_DOAs((pred_deg-true_deg)^2))))', 'paper_architecture': '2x16x16 -> Conv512 -> ResBlock256 -> ResBlock128(stride2) -> AdaptiveAvgPool -> FC(classes)'}
    with open(out_dir / 'run_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Paper-faithful ResNet DOA classifier for y_receive / target_azimuth.')
    parser.add_argument('--mat-file', type=str, default=None, help='Path to MAT file. If omitted, auto-detects when exactly one .mat exists in the current directory.')
    parser.add_argument('--cache-dir', type=str, default='./doa_resnet_cache')
    parser.add_argument('--output-dir', type=str, default='./doa_resnet_output')
    parser.add_argument('--train-mode', choices=('per_snr', 'mixed'), default='per_snr', help="per_snr is closer to the paper's SNR experiments; mixed trains one model on all 7 SNR conditions.")
    parser.add_argument('--grid-resolution', type=float, default=GRID_STEP_DEG, help='Paper studies 0.25 and 0.1; default uses its finer 0.1 grid.')
    parser.add_argument('--angle-min', type=float, default=None)
    parser.add_argument('--angle-max', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--snr-values', type=str, default=','.join((str(v) for v in SNR_DB)), help='Seven comma-separated SNR labels in y_receive axis order.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=0, help='0 is safest on Windows with memory-mapped features.')
    parser.add_argument('--amp', action='store_true', help='Use CUDA mixed precision to reduce memory. This is a numerical/performance implementation option, not specified in the paper.')
    parser.add_argument('--force-preprocess', action='store_true', help='Rebuild cached correlation features.')
    parser.add_argument('--retrain', action='store_true', help='Ignore existing model checkpoint(s) and train again.')
    parser.add_argument('--use-best-val', action='store_true', help='Evaluate the best validation-loss state rather than the final 100-epoch state. OFF by default to stay closer to fixed-epoch paper training.')
    parser.add_argument('--min-output-separation', type=float, default=0.0, help="Optional peak separation in degrees. 0 means literal top-4 classes, matching the paper's stated inference.")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    if args.mat_file is None:
        args.mat_file = discover_mat_file()
    args.mat_file = os.path.abspath(args.mat_file)
    if not os.path.isfile(args.mat_file):
        raise FileNotFoundError(args.mat_file)
    snr_db = np.asarray([float(v.strip()) for v in args.snr_values.split(',')], dtype=np.float64)
    if len(snr_db) != NUM_SNRS:
        raise ValueError(f'Expected exactly {NUM_SNRS} SNR labels; got {snr_db.tolist()}')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_file, angles, meta = build_cache(mat_file=args.mat_file, cache_dir=args.cache_dir, force=args.force_preprocess)
    n_setups = int(angles.shape[0])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_setups)
    n_train = int(0.8 * n_setups)
    n_val = int(0.1 * n_setups)
    train_ids = perm[:n_train]
    val_ids = perm[n_train:n_train + n_val]
    test_ids = perm[n_train + n_val:]
    split = {'train': train_ids, 'val': val_ids, 'test': test_ids}
    grid = build_angle_grid(targets=angles, resolution_deg=float(args.grid_resolution), angle_min=args.angle_min, angle_max=args.angle_max)
    labels = encode_multihot_targets(angles=angles, grid=grid)
    save_run_config(args=args, grid=grid, split=split, out_dir=out_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    model_probe = DOAResNet(n_classes=len(grid))
    n_params = sum((p.numel() for p in model_probe.parameters()))
    del model_probe
    all_test_rows = []
    if args.train_mode == 'per_snr':
        train_jobs = [{'name': f'snr_{snr_db[i]:+g}dB'.replace('+', 'p').replace('-', 'm'), 'train_snrs': [i], 'eval_snrs': [i], 'display': f'SNR {snr_db[i]:+g} dB'} for i in range(NUM_SNRS)]
    else:
        train_jobs = [{'name': 'mixed_all_snrs', 'train_snrs': list(range(NUM_SNRS)), 'eval_snrs': list(range(NUM_SNRS)), 'display': 'mixed SNR model'}]
    for job_number, job in enumerate(train_jobs, start=1):
        train_ds = CorrelationDataset(feat_file=feat_file, labels=labels, truth=angles, setups=train_ids, snrs=job['train_snrs'])
        val_ds = CorrelationDataset(feat_file=feat_file, labels=labels, truth=angles, setups=val_ids, snrs=job['train_snrs'])
        test_ds = CorrelationDataset(feat_file=feat_file, labels=labels, truth=angles, setups=test_ids, snrs=job['eval_snrs'])
        train_loader = get_loader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, device=device)
        val_loader = get_loader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, device=device)
        test_loader = get_loader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, device=device)
        model = DOAResNet(n_classes=len(grid)).to(device)
        ckpt_file = out_dir / f"{job['name']}_final.pt"
        ckpt_info = {'job': job, 'grid': grid.tolist(), 'grid_resolution': float(args.grid_resolution), 'train_setups': train_ids.tolist(), 'val_setups': val_ids.tolist(), 'test_setups': test_ids.tolist(), 'epochs': int(args.epochs), 'batch_size': int(args.batch_size), 'lr': float(args.lr), 'paper_cross_entropy': 'unnormalized multi-hot CE'}
        if ckpt_file.exists() and (not args.retrain):
            checkpoint = torch.load(ckpt_file, map_location='cpu', weights_only=False)
            saved_grid = np.asarray(checkpoint['meta']['grid'], dtype=np.float32)
            if saved_grid.shape != grid.shape or not np.allclose(saved_grid, grid, atol=1e-06):
                raise ValueError(f'Checkpoint grid does not match current grid. Use --retrain or a different output directory.')
            state = checkpoint['best_state'] if args.use_best_val else checkpoint['model_state']
            model.load_state_dict(state)
            model.to(device)
        else:
            try:
                train_one_model(model=model, train_loader=train_loader, val_loader=val_loader, device=device, epochs=args.epochs, lr=args.lr, use_amp=args.amp, ckpt_file=ckpt_file, ckpt_info=ckpt_info)
            except torch.cuda.OutOfMemoryError:
                print(f'\nCUDA OUT OF MEMORY.\nThe paper batch size is 1024, which may exceed your GPU memory with the 512-channel first layer.\nRe-run with, for example:\n  python {Path(__file__).name} --mat-file "{args.mat_file}" --batch-size 256 --amp\n')
                raise
            if args.use_best_val:
                checkpoint = torch.load(ckpt_file, map_location='cpu', weights_only=False)
                model.load_state_dict(checkpoint['best_state'])
                model.to(device)
        test_rows = test_model(model=model, loader=test_loader, device=device, grid=grid, snr_db=snr_db, min_sep=float(args.min_output_separation), use_amp=args.amp)
        all_test_rows.extend(test_rows)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    all_test_rows.sort(key=lambda r: (r['snr_idx'], r['setup_idx']))
    prediction_csv = out_dir / 'test_predictions.csv'
    summary_csv = out_dir / 'test_metric_summary.csv'
    write_prediction_csv(all_test_rows, prediction_csv)
    final_metric = summarize_results(rows=all_test_rows, snr_db=snr_db, summary_path=summary_csv)
    print(f'[DONE] final requested metric = {final_metric:.9f}')
if __name__ == '__main__':
    main()
