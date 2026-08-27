
"""

y_receive:
    MATLAB shape (16, 50, 7, 5000), complex
    16 antennas, 50 snapshots, 7 SNR values, 5000 setups

target_azimuth:
    MATLAB shape (1, 4, 5000), real, degrees
    4 source azimuths for each setup


H. Al Kassir et al.,
"Improving DOA Estimation via an Optimal Deep Residual Neural Network
Classifier on Uniform Linear Arrays," IEEE Open Journal of Antennas and
Propagation, vol. 5, no. 2, 2024.


1. 16-element ULA input assumption.
2. Correlation/covariance input represented as:
       [real(Rxx), imag(Rxx)] -> shape (2, 16, 16)
3. Full correlation matrix (not the later simplified upper triangle).
4. ResNet:
       Conv 2 -> 512, 3x3, no bias
       BatchNorm + ReLU
       Residual block -> 256 channels
       Residual block -> 128 channels, stride 2
       Adaptive average pool -> 1x1
       Fully connected -> angular classes
5. Angular grid classification.
6. Default grid resolution = 0.1 degree.
7. Softmax probabilities at inference.
8. Multi-hot label vector for simultaneous sources.
9. Cross-entropy formulation using the multi-hot target.
10. Adam optimizer.
11. Learning rate = 0.001.
12. Batch size = 1024.
13. 100 epochs.
14. 80/10/10 train/validation/test split.
15. Top-K highest-probability classes are the DOA estimates.


Examples
--------
Simplest:
    python doa_resnet_paper.py --mat-file "C:/path/to/data.mat"

If the file is the only .mat in the current directory:
    python doa_resnet_paper.py

Paper defaults but with AMP to reduce GPU memory:
    python doa_resnet_paper.py --mat-file data.mat --amp

If batch 1024 does not fit your GPU:
    python doa_resnet_paper.py --mat-file data.mat --batch-size 256 --amp

To use one model trained on all 7 SNR values instead:
    python doa_resnet_paper.py --mat-file data.mat --train-mode mixed

Dependencies
------------
    pip install numpy scipy h5py torch
"""

import argparse
import csv
import json
import math
import os
import random
import sys
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


# ============================================================================
# Paper/data constants
# ============================================================================

N_ANTENNAS = 16
N_SNAPSHOTS = 50
N_SNRS = 7
N_SOURCES = 4

DEFAULT_SNR_DB = (-20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0)

# Paper settings
PAPER_GRID_RESOLUTION_DEG = 0.1
PAPER_ANGLE_MIN_DEG = 30.0
PAPER_ANGLE_MAX_DEG = 150.0
PAPER_EPOCHS = 100
PAPER_BATCH_SIZE = 1024
PAPER_LR = 1e-3

# Numerical guard only. If RMSE is exactly zero, log10(0) is -infinity.
METRIC_RMSE_FLOOR = 1e-12


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# MAT loading
# ============================================================================

def _to_complex(arr: np.ndarray) -> np.ndarray:
    """
    Convert common MATLAB v7.3/HDF5 complex representations to NumPy complex.
    """
    arr = np.asarray(arr)

    if np.iscomplexobj(arr):
        return arr

    if arr.dtype.names is not None:
        names = set(arr.dtype.names)

        if {"real", "imag"}.issubset(names):
            return arr["real"] + 1j * arr["imag"]

        if {"r", "i"}.issubset(names):
            return arr["r"] + 1j * arr["i"]

    return arr


def normalize_target_shape(target: np.ndarray) -> np.ndarray:
    """
    Normalize target_azimuth to (num_setups, 4).

    Handles MATLAB shape (1,4,N), squeezed (4,N), and HDF5-reversed forms.
    """
    target = np.asarray(target)
    target = np.squeeze(target)

    if target.ndim != 2:
        raise ValueError(
            "target_azimuth must become a 2-D array after squeeze; "
            f"got {target.shape}"
        )

    if target.shape[1] == N_SOURCES:
        out = target
    elif target.shape[0] == N_SOURCES:
        out = target.T
    else:
        raise ValueError(
            "Could not convert target_azimuth to shape (N,4). "
            f"Observed shape after squeeze: {target.shape}"
        )

    out = np.asarray(out, dtype=np.float32)

    if not np.all(np.isfinite(out)):
        bad = np.argwhere(~np.isfinite(out))
        raise ValueError(
            f"target_azimuth contains non-finite values. First bad indices: {bad[:10]}"
        )
    out = np.rad2deg(out).astype(np.float32)
    if out.shape[1] != N_SOURCES:
        raise ValueError(f"Expected 4 sources; got target shape {out.shape}")

    return out


class MatReader:
    """
    Reads y_receive lazily when the MAT file is MATLAB v7.3/HDF5.

    The code identifies axes by their unique sizes, so it supports both
    MATLAB's logical shape (16,50,7,5000) and the reversed ordering often
    exposed by h5py.
    """

    def __init__(self, path: str):
        self.path = str(path)
        self.mode = None
        self.h5 = None
        self.y = None
        self.targets = None
        self.y_shape = None
        self.axes = None

        # First try MATLAB v7.3 / HDF5.
        if h5py is not None:
            try:
                self.h5 = h5py.File(self.path, "r")

                if "y_receive" not in self.h5 or "target_azimuth" not in self.h5:
                    keys = list(self.h5.keys())
                    self.h5.close()
                    self.h5 = None
                    raise KeyError(
                        "Required variables y_receive and target_azimuth were not "
                        f"found. Top-level HDF5 keys: {keys}"
                    )

                self.mode = "hdf5"
                self.y = self.h5["y_receive"]

                target_raw = np.asarray(self.h5["target_azimuth"])
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

        # Fall back to old MAT format.
        if loadmat is None:
            raise RuntimeError(
                "MAT file is not readable as HDF5 and scipy.io.loadmat is "
                "unavailable. Install scipy and h5py."
            )

        print(
            "[MAT] File is not MATLAB v7.3/HDF5. Using scipy.io.loadmat.\n"
            "[MAT] WARNING: this loads y_receive into RAM."
        )

        data = loadmat(
            self.path,
            variable_names=["y_receive", "target_azimuth"],
        )

        if "y_receive" not in data or "target_azimuth" not in data:
            raise KeyError(
                "Required variables y_receive and target_azimuth were not found."
            )

        self.mode = "scipy"
        self.y = _to_complex(data["y_receive"])
        target_raw = _to_complex(data["target_azimuth"])
        if np.iscomplexobj(target_raw):
            target_raw = target_raw.real
        self.targets = normalize_target_shape(target_raw)

        self.y_shape = tuple(self.y.shape)
        self._infer_axes()

    def _infer_axes(self) -> None:
        if len(self.y_shape) != 4:
            raise ValueError(
                f"y_receive must be 4-D. Observed shape: {self.y_shape}"
            )

        n_setups = self.targets.shape[0]

        wanted = {
            "ant": N_ANTENNAS,
            "snap": N_SNAPSHOTS,
            "snr": N_SNRS,
            "setup": n_setups,
        }

        axes = {}
        used = set()

        for name, size in wanted.items():
            candidates = [
                axis
                for axis, dim in enumerate(self.y_shape)
                if dim == size and axis not in used
            ]

            if len(candidates) != 1:
                raise ValueError(
                    f"Could not uniquely infer {name!r} axis of size {size} "
                    f"from y_receive shape {self.y_shape}. Candidates={candidates}"
                )

            axes[name] = candidates[0]
            used.add(candidates[0])

        self.axes = axes

        print(f"[MAT] backend               : {self.mode}")
        print(f"[MAT] raw y_receive shape   : {self.y_shape}")
        print(f"[MAT] inferred axes         : {self.axes}")
        print(f"[MAT] target_azimuth shape  : {self.targets.shape}")
        print(
            f"[MAT] target angle range    : "
            f"{self.targets.min():.6f} .. {self.targets.max():.6f} deg"
        )

        
    @property
    def n_setups(self) -> int:
        return int(self.targets.shape[0])

    def get_setup_all_snrs(self, setup_idx: int) -> np.ndarray:
        """
        Return all 7 SNR observations for one setup as:
            complex array of shape (16, 50, 7)
        """
        sl = [slice(None)] * 4
        sl[self.axes["setup"]] = int(setup_idx)

        raw = np.asarray(self.y[tuple(sl)])
        raw = _to_complex(raw)
        raw = np.squeeze(raw)

        if raw.ndim != 3:
            raise ValueError(
                "After selecting one setup, y_receive must be 3-D; "
                f"got {raw.shape}"
            )

        # The sliced setup axis disappears. Find the positions of the remaining
        # original axes in the returned 3-D array.
        remaining_original_axes = [
            axis for axis in range(4) if axis != self.axes["setup"]
        ]

        ant_pos = remaining_original_axes.index(self.axes["ant"])
        snap_pos = remaining_original_axes.index(self.axes["snap"])
        snr_pos = remaining_original_axes.index(self.axes["snr"])

        out = np.transpose(raw, (ant_pos, snap_pos, snr_pos))

        if out.shape != (N_ANTENNAS, N_SNAPSHOTS, N_SNRS):
            raise ValueError(
                f"Expected setup data shape (16,50,7), got {out.shape}"
            )

        if not np.iscomplexobj(out):
            raise ValueError(
                "y_receive does not appear to be complex. "
                "The paper's correlation construction assumes complex array data."
            )

        return np.asarray(out)

    def close(self) -> None:
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


# ============================================================================
# Correlation preprocessing
# ============================================================================

def correlation_feature(y: np.ndarray) -> np.ndarray:
    """
    Adapt the paper's complex correlation input to the available snapshots.

    y: (16,50), complex

    R_hat = Y Y^H / T

    returns:
        (2,16,16), float32
        channel 0 = real(R_hat)
        channel 1 = imag(R_hat)

    No trace/power normalization is performed because the paper does not
    describe such normalization for its principal full-correlation experiment.
    """
    if y.shape != (N_ANTENNAS, N_SNAPSHOTS):
        raise ValueError(f"Expected y shape (16,50); got {y.shape}")

    r = (y @ y.conj().T) / float(N_SNAPSHOTS)

    feature = np.stack(
        (r.real, r.imag),
        axis=0,
    ).astype(np.float32, copy=False)

    return feature


def _cache_is_valid(meta: dict, mat_file: str) -> bool:
    try:
        stat = os.stat(mat_file)
        return (
            os.path.abspath(meta["source_file"]) == os.path.abspath(mat_file)
            and int(meta["source_size"]) == int(stat.st_size)
            and abs(float(meta["source_mtime"]) - float(stat.st_mtime)) < 1e-6
            and meta["feature_shape"][1:] == [N_SNRS, 2, N_ANTENNAS, N_ANTENNAS]
        )
    except Exception:
        return False


def preprocess_to_cache(
    mat_file: str,
    cache_dir: str,
    force: bool = False,
):
    """
    Cache the large MAT file into ~72 MB of float32 correlation features.

    Saved feature shape:
        (N_setups, 7, 2, 16, 16)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    feature_path = cache_dir / "correlation_features.npy"
    target_path = cache_dir / "target_azimuth.npy"
    meta_path = cache_dir / "cache_meta.json"

    if (
        not force
        and feature_path.exists()
        and target_path.exists()
        and meta_path.exists()
    ):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            if _cache_is_valid(meta, mat_file):
                print(f"[CACHE] Reusing: {feature_path}")
                targets = np.load(target_path)
                return str(feature_path), targets, meta

            print("[CACHE] Existing cache does not match source file; rebuilding.")

        except Exception as exc:
            print(f"[CACHE] Could not validate existing cache ({exc}); rebuilding.")

    print("[CACHE] Building correlation-feature cache...")
    t0 = time.time()

    with MatReader(mat_file) as reader:
        n_setups = reader.n_setups

        mmap = np.lib.format.open_memmap(
            feature_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_setups, N_SNRS, 2, N_ANTENNAS, N_ANTENNAS),
        )

        for setup_idx in range(n_setups):
            setup_data = reader.get_setup_all_snrs(setup_idx)

            for snr_idx in range(N_SNRS):
                mmap[setup_idx, snr_idx] = correlation_feature(
                    setup_data[:, :, snr_idx]
                )

            if (
                (setup_idx + 1) % 100 == 0
                or setup_idx + 1 == n_setups
            ):
                elapsed = time.time() - t0
                print(
                    f"\r[CACHE] {setup_idx + 1:5d}/{n_setups} setups "
                    f"({elapsed / 60.0:.1f} min)",
                    end="",
                    flush=True,
                )

        print()
        mmap.flush()

        targets = reader.targets.astype(np.float32, copy=True)
        np.save(target_path, targets)

        stat = os.stat(mat_file)
        meta = {
            "source_file": os.path.abspath(mat_file),
            "source_size": int(stat.st_size),
            "source_mtime": float(stat.st_mtime),
            "feature_shape": [int(v) for v in mmap.shape],
            "target_shape": [int(v) for v in targets.shape],
            "formula": "R_hat = Y @ Y^H / 50; channels=[real,imag]",
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print(
        f"[CACHE] Complete in {(time.time() - t0) / 60.0:.2f} min\n"
        f"[CACHE] Features: {feature_path}"
    )

    return str(feature_path), targets, meta


# ============================================================================
# Angular grid and labels
# ============================================================================

def build_angle_grid(
    targets: np.ndarray,
    resolution_deg: float,
    angle_min: Optional[float],
    angle_max: Optional[float],
) -> np.ndarray:
    """
    Use the paper's exact [30,150] support whenever compatible with the data.
    Otherwise adapt endpoints to the observed target support unless explicitly
    specified by the user.
    """
    data_min = float(np.min(targets))
    data_max = float(np.max(targets))

    if angle_min is None and angle_max is None:
        if (
            data_min >= PAPER_ANGLE_MIN_DEG - 1e-6
            and data_max <= PAPER_ANGLE_MAX_DEG + 1e-6
        ):
            angle_min = PAPER_ANGLE_MIN_DEG
            angle_max = PAPER_ANGLE_MAX_DEG
            print(
                "[GRID] Targets fit paper range; using exact paper support "
                "[30,150] degrees."
            )
        else:
            angle_min = (
                math.floor(data_min / resolution_deg) * resolution_deg
            )
            angle_max = (
                math.ceil(data_max / resolution_deg) * resolution_deg
            )
            print(
                "[GRID] Dataset does not fit the paper's [30,150] degree range.\n"
                f"[GRID] Adapting support to [{angle_min},{angle_max}] degrees."
            )

    elif (angle_min is None) != (angle_max is None):
        raise ValueError(
            "Specify both --angle-min and --angle-max, or specify neither."
        )

    angle_min = float(angle_min)
    angle_max = float(angle_max)

    if data_min < angle_min - resolution_deg / 2:
        raise ValueError(
            f"Target minimum {data_min} is below angle grid minimum {angle_min}."
        )

    if data_max > angle_max + resolution_deg / 2:
        raise ValueError(
            f"Target maximum {data_max} is above angle grid maximum {angle_max}."
        )

    n_classes_float = (angle_max - angle_min) / resolution_deg
    n_classes = int(round(n_classes_float)) + 1

    if not np.isclose(n_classes_float, round(n_classes_float), atol=1e-6):
        raise ValueError(
            "Angular span must be an integer multiple of grid resolution."
        )

    grid = (
        angle_min
        + np.arange(n_classes, dtype=np.float64) * resolution_deg
    ).astype(np.float32)

    print(
        f"[GRID] range      : {grid[0]:.4f} .. {grid[-1]:.4f} deg\n"
        f"[GRID] resolution : {resolution_deg:.4f} deg\n"
        f"[GRID] classes    : {len(grid)}"
    )

    return grid


def encode_multihot_targets(
    target_angles: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """
    Paper-style multi-hot target.

    For four sources there are four entries equal to 1, assuming no two true
    sources quantize to the same angular class.
    """
    n_setups = target_angles.shape[0]
    n_classes = len(grid)
    resolution = float(grid[1] - grid[0]) if len(grid) > 1 else 1.0
    grid_min = float(grid[0])

    labels = np.zeros((n_setups, n_classes), dtype=np.float32)

    for setup_idx in range(n_setups):
        for theta in target_angles[setup_idx]:
            idx = int(round((float(theta) - grid_min) / resolution))
            idx = max(0, min(n_classes - 1, idx))
            labels[setup_idx, idx] = 1.0

        active = int(labels[setup_idx].sum())
        if active != N_SOURCES:
            raise ValueError(
                f"Setup {setup_idx} produces only {active} unique grid classes "
                f"from angles {target_angles[setup_idx].tolist()}. "
                "Use a finer grid resolution."
            )

    return labels


# ============================================================================
# Dataset
# ============================================================================

class CorrelationDataset(Dataset):
    def __init__(
        self,
        feature_path: str,
        labels: np.ndarray,
        true_angles: np.ndarray,
        setup_indices: Sequence[int],
        snr_indices: Sequence[int],
    ):
        self.features = np.load(feature_path, mmap_mode="r")
        self.labels = labels
        self.true_angles = true_angles

        setup_indices = np.asarray(setup_indices, dtype=np.int64)
        snr_indices = np.asarray(snr_indices, dtype=np.int64)

        # Preserve setup/SNR identity.
        self.pairs = np.asarray(
            [
                (int(setup_idx), int(snr_idx))
                for setup_idx in setup_indices
                for snr_idx in snr_indices
            ],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return int(len(self.pairs))

    def __getitem__(self, item: int):
        setup_idx, snr_idx = self.pairs[item]

        # Copy from mmap to writable normal NumPy memory before torch conversion.
        x = np.array(
            self.features[setup_idx, snr_idx],
            dtype=np.float32,
            copy=True,
        )
        label = np.array(
            self.labels[setup_idx],
            dtype=np.float32,
            copy=True,
        )
        angles = np.array(
            self.true_angles[setup_idx],
            dtype=np.float32,
            copy=True,
        )

        return (
            torch.from_numpy(x),
            torch.from_numpy(label),
            torch.from_numpy(angles),
            int(setup_idx),
            int(snr_idx),
        )


# ============================================================================
# Paper ResNet
# ============================================================================

class ResidualBlock(nn.Module):
    """
    Standard projection residual block used to realize the channel changes
    shown in the paper's Figure 3(a).

    The paper diagram/text specifies two 3x3 convolutions in each residual
    block and skip connections. A 1x1 projection on the shortcut is necessary
    when channel count or stride changes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
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


class PaperDOAResNet(nn.Module):
    """
    ResNet in Figure 3(a) / Section IV-A of the paper:

        input       : 2 x 16 x 16
        stem        : Conv 3x3, 2 -> 512, no bias + BN + ReLU
        residual #1 : -> 256 channels
        residual #2 : -> 128 channels, stride 2
        pool        : AdaptiveAvgPool -> 1 x 1 x 128
        FC          : 128 -> C angular classes

    The network emits logits. Softmax is applied at inference. During training
    paper_cross_entropy() internally uses log-softmax, matching the paper's
    statement that cross-entropy incorporates the Softmax operation.
    """

    def __init__(self, n_classes: int):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                2,
                512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.block1 = ResidualBlock(
            in_channels=512,
            out_channels=256,
            stride=1,
        )

        self.block2 = ResidualBlock(
            in_channels=256,
            out_channels=128,
            stride=2,
        )

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


def paper_cross_entropy(
    logits: torch.Tensor,
    multi_hot_targets: torch.Tensor,
) -> torch.Tensor:
    """
    Literal multi-hot version of the paper's cross-entropy equation.

    For a sample with four true angles:
        L = -sum_c y_c log softmax(logit)_c
    where y has four entries equal to 1.

    This deliberately does NOT normalize the four active labels to sum to one,
    because the paper explicitly illustrates a multi-hot label vector and writes
    cross entropy in terms of that label vector.

    Dividing this loss by 4 would only apply a constant scale to every sample;
    we leave it unnormalized to remain close to the paper's stated formulation.
    """
    log_probs = F.log_softmax(logits, dim=1)
    return -(multi_hot_targets * log_probs).sum(dim=1).mean()


# ============================================================================
# Prediction and requested metric
# ============================================================================

def select_topk_angles(
    probabilities: np.ndarray,
    grid: np.ndarray,
    k: int = N_SOURCES,
    min_separation_deg: float = 0.0,
) -> np.ndarray:
    """
    Paper-faithful default: simply use the K highest-probability classes.

    If min_separation_deg > 0, optional greedy suppression is used. This is NOT
    the paper default and is supplied only as a diagnostic option.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if min_separation_deg <= 0:
        idx = np.argpartition(probabilities, -k)[-k:]
        idx = idx[np.argsort(probabilities[idx])[::-1]]
        return np.sort(grid[idx].astype(np.float64))

    order = np.argsort(probabilities)[::-1]
    chosen = []

    for idx in order:
        theta = float(grid[idx])

        if all(
            abs(theta - prior) >= min_separation_deg
            for prior in chosen
        ):
            chosen.append(theta)

        if len(chosen) == k:
            break

    if len(chosen) != k:
        raise RuntimeError(
            f"Could not select {k} angles with separation "
            f"{min_separation_deg} deg."
        )

    return np.sort(np.asarray(chosen, dtype=np.float64))


def sample_rmse_deg(
    predicted_angles: np.ndarray,
    true_angles: np.ndarray,
) -> float:
    """
    For four scalar 1-D DOAs with squared-error cost, sorting both vectors gives
    the minimum-cost permutation assignment.
    """
    pred = np.sort(np.asarray(predicted_angles, dtype=np.float64))
    true = np.sort(np.asarray(true_angles, dtype=np.float64))

    if pred.shape != (N_SOURCES,) or true.shape != (N_SOURCES,):
        raise ValueError(
            f"Expected four predicted/true angles; got {pred.shape}, {true.shape}"
        )

    return float(np.sqrt(np.mean((pred - true) ** 2)))


def requested_log_rmse_metric(rmse_deg: float) -> float:
    """
    User-requested per-sample value:
        10 * log10(RMSE_sample)

    A tiny floor only prevents numerical -inf for an exactly-zero RMSE.
    """
    safe_rmse = max(float(rmse_deg), METRIC_RMSE_FLOOR)
    return float(10.0 * np.log10(safe_rmse))


# ============================================================================
# Train / validation / test
# ============================================================================

def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )


def autocast_context(device: torch.device, enabled: bool):
    # torch.amp.autocast API for current PyTorch.
    if device.type == "cuda":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=enabled,
        )
    # CPU no-op context through disabled autocast.
    return torch.autocast(
        device_type="cpu",
        enabled=False,
    )


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    use_amp: bool,
    checkpoint_path: Path,
    checkpoint_meta: dict,
):
    """
    The paper uses Adam with lr=0.001 and trains for 100 epochs.
    No scheduler or early stopping is used here by default.
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    # Current torch.amp GradScaler API; keep compatibility with older versions.
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    best_val_loss = float("inf")
    best_state = None

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # ----------------------
        # Train
        # ----------------------
        model.train()

        train_loss_sum = 0.0
        train_count = 0

        for x, y, _, _, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(device, use_amp):
                logits = model(x)
                loss = paper_cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = x.shape[0]
            train_loss_sum += float(loss.detach().cpu()) * bs
            train_count += bs

        train_loss = train_loss_sum / max(train_count, 1)

        # ----------------------
        # Validation
        # ----------------------
        model.eval()

        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for x, y, _, _, _ in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with autocast_context(device, use_amp):
                    logits = model(x)
                    loss = paper_cross_entropy(logits, y)

                bs = x.shape[0]
                val_loss_sum += float(loss.detach().cpu()) * bs
                val_count += bs

        val_loss = val_loss_sum / max(val_count, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Best is saved for reference, but final paper-style evaluation uses
            # the model after the requested fixed number of epochs.
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        elapsed = time.time() - t0

        print(
            f"  epoch {epoch:03d}/{epochs} | "
            f"train CE={train_loss:.6f} | "
            f"val CE={val_loss:.6f} | "
            f"best val={best_val_loss:.6f} | "
            f"{elapsed / 60.0:.1f} min"
        )

    # Save final model. This is the state used by default, matching fixed
    # 100-epoch paper training rather than early-stopping selection.
    payload = {
        "model_state": model.state_dict(),
        "best_state": best_state,
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "meta": checkpoint_meta,
    }
    torch.save(payload, checkpoint_path)

    print(f"[CKPT] saved final model: {checkpoint_path}")


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    grid: np.ndarray,
    snr_values: np.ndarray,
    min_output_separation: float,
    use_amp: bool,
):
    model.eval()
    rows = []

    for x, _, true_angles, setup_idx, snr_idx in loader:
        x = x.to(device, non_blocking=True)

        with autocast_context(device, use_amp):
            logits = model(x)

        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        true_angles_np = true_angles.numpy()
        setup_idx_np = np.asarray(setup_idx)
        snr_idx_np = np.asarray(snr_idx)

        for b in range(len(probs)):
            pred = select_topk_angles(
                probabilities=probs[b],
                grid=grid,
                k=N_SOURCES,
                min_separation_deg=min_output_separation,
            )

            true = np.sort(
                np.asarray(true_angles_np[b], dtype=np.float64)
            )

            rmse = sample_rmse_deg(pred, true)
            metric = requested_log_rmse_metric(rmse)

            row = {
                "setup_idx": int(setup_idx_np[b]),
                "snr_idx": int(snr_idx_np[b]),
                "snr_db": float(snr_values[int(snr_idx_np[b])]),
                "rmse_deg": float(rmse),
                "metric_10log10_rmse": float(metric),
            }

            for i in range(N_SOURCES):
                row[f"true_{i+1}_deg"] = float(true[i])
                row[f"pred_{i+1}_deg"] = float(pred[i])

            rows.append(row)

    return rows


# ============================================================================
# Reporting
# ============================================================================

def write_prediction_csv(rows, path: Path) -> None:
    fieldnames = [
        "setup_idx",
        "snr_idx",
        "snr_db",
        "true_1_deg",
        "true_2_deg",
        "true_3_deg",
        "true_4_deg",
        "pred_1_deg",
        "pred_2_deg",
        "pred_3_deg",
        "pred_4_deg",
        "rmse_deg",
        "metric_10log10_rmse",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def summarize_results(
    rows,
    snr_values: np.ndarray,
    summary_path: Path,
):
    if not rows:
        raise RuntimeError("No test predictions were produced.")

    metric_all = np.asarray(
        [r["metric_10log10_rmse"] for r in rows],
        dtype=np.float64,
    )
    rmse_all = np.asarray(
        [r["rmse_deg"] for r in rows],
        dtype=np.float64,
    )

    final_metric = float(np.mean(metric_all))
    mean_rmse = float(np.mean(rmse_all))

    summary_rows = []

    print("\n" + "=" * 78)
    print("TEST RESULTS")
    print("=" * 78)
    print(
        "SNR (dB) | samples | mean RMSE (deg) | "
        "mean[10 log10(RMSE_sample)]"
    )
    print("-" * 78)

    for snr in snr_values:
        subset = [r for r in rows if np.isclose(r["snr_db"], snr)]

        if not subset:
            continue

        values = np.asarray(
            [r["metric_10log10_rmse"] for r in subset],
            dtype=np.float64,
        )
        rmses = np.asarray(
            [r["rmse_deg"] for r in subset],
            dtype=np.float64,
        )

        snr_metric = float(np.mean(values))
        snr_mean_rmse = float(np.mean(rmses))

        print(
            f"{snr:8.1f} | "
            f"{len(subset):7d} | "
            f"{snr_mean_rmse:15.6f} | "
            f"{snr_metric:27.6f}"
        )

        summary_rows.append(
            {
                "snr_db": float(snr),
                "samples": int(len(subset)),
                "mean_rmse_deg": snr_mean_rmse,
                "mean_10log10_rmse": snr_metric,
            }
        )

    print("-" * 78)
    print(f"Overall test samples : {len(rows)}")
    print(f"Supplementary mean RMSE (deg): {mean_rmse:.9f}")
    print()
    print("FINAL REQUESTED METRIC")
    print(
        "mean over test samples of "
        "10*log10(sqrt(mean(error_of_4_DOAs^2)))"
    )
    print(f"= {final_metric:.9f}")
    print("=" * 78)

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "snr_db",
            "samples",
            "mean_rmse_deg",
            "mean_10log10_rmse",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

        writer.writerow(
            {
                "snr_db": "OVERALL",
                "samples": len(rows),
                "mean_rmse_deg": mean_rmse,
                "mean_10log10_rmse": final_metric,
            }
        )

    return final_metric


# ============================================================================
# Utility
# ============================================================================

def discover_mat_file() -> str:
    mats = sorted(Path.cwd().glob("*.mat"))

    if len(mats) == 1:
        print(f"[MAT] Auto-detected: {mats[0]}")
        return str(mats[0])

    if len(mats) == 0:
        raise FileNotFoundError(
            "No --mat-file was supplied and no .mat file exists in the current "
            "directory."
        )

    raise RuntimeError(
        "No --mat-file was supplied and multiple .mat files exist in the "
        "current directory:\n  "
        + "\n  ".join(str(p) for p in mats)
    )


def save_run_config(args, grid, split, output_dir: Path) -> None:
    config = {
        "arguments": vars(args),
        "grid": {
            "min": float(grid[0]),
            "max": float(grid[-1]),
            "resolution": float(args.grid_resolution),
            "classes": int(len(grid)),
        },
        "split": {
            "train_setups": [int(x) for x in split["train"]],
            "val_setups": [int(x) for x in split["val"]],
            "test_setups": [int(x) for x in split["test"]],
        },
        "requested_metric": (
            "mean_samples(10*log10("
            "sqrt(mean_4_DOAs((pred_deg-true_deg)^2))))"
        ),
        "paper_architecture": (
            "2x16x16 -> Conv512 -> ResBlock256 -> "
            "ResBlock128(stride2) -> AdaptiveAvgPool -> FC(classes)"
        ),
    }

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Paper-faithful ResNet DOA classifier for "
            "y_receive / target_azimuth."
        ),
    )

    parser.add_argument(
        "--mat-file",
        type=str,
        default=None,
        help=(
            "Path to MAT file. If omitted, auto-detects when exactly one .mat "
            "exists in the current directory."
        ),
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./doa_resnet_cache",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./doa_resnet_output",
    )

    parser.add_argument(
        "--train-mode",
        choices=("per_snr", "mixed"),
        default="per_snr",
        help=(
            "per_snr is closer to the paper's SNR experiments; mixed trains "
            "one model on all 7 SNR conditions."
        ),
    )

    parser.add_argument(
        "--grid-resolution",
        type=float,
        default=PAPER_GRID_RESOLUTION_DEG,
        help="Paper studies 0.25 and 0.1; default uses its finer 0.1 grid.",
    )

    parser.add_argument(
        "--angle-min",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--angle-max",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=PAPER_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=PAPER_BATCH_SIZE,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=PAPER_LR,
    )

    parser.add_argument(
        "--snr-values",
        type=str,
        default=",".join(str(v) for v in DEFAULT_SNR_DB),
        help="Seven comma-separated SNR labels in y_receive axis order.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="0 is safest on Windows with memory-mapped features.",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help=(
            "Use CUDA mixed precision to reduce memory. This is a numerical/"
            "performance implementation option, not specified in the paper."
        ),
    )

    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Rebuild cached correlation features.",
    )

    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Ignore existing model checkpoint(s) and train again.",
    )

    parser.add_argument(
        "--use-best-val",
        action="store_true",
        help=(
            "Evaluate the best validation-loss state rather than the final "
            "100-epoch state. OFF by default to stay closer to fixed-epoch "
            "paper training."
        ),
    )

    parser.add_argument(
        "--min-output-separation",
        type=float,
        default=0.0,
        help=(
            "Optional peak separation in degrees. 0 means literal top-4 "
            "classes, matching the paper's stated inference."
        ),
    )

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.mat_file is None:
        args.mat_file = discover_mat_file()

    args.mat_file = os.path.abspath(args.mat_file)

    if not os.path.isfile(args.mat_file):
        raise FileNotFoundError(args.mat_file)

    snr_values = np.asarray(
        [float(v.strip()) for v in args.snr_values.split(",")],
        dtype=np.float64,
    )

    if len(snr_values) != N_SNRS:
        raise ValueError(
            f"Expected exactly {N_SNRS} SNR labels; got {snr_values.tolist()}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PAPER-STYLE RESNET DOA ESTIMATION")
    print("=" * 78)
    print(f"MAT file    : {args.mat_file}")
    print(f"train mode  : {args.train_mode}")
    print(f"SNRs        : {snr_values.tolist()}")
    print(f"epochs      : {args.epochs}")
    print(f"batch size  : {args.batch_size}")
    print(f"learning rt : {args.lr}")
    print(f"grid res    : {args.grid_resolution} deg")
    print()

    # ----------------------------------------------------------------------
    # 1. Convert huge y_receive into compact 2x16x16 correlation features.
    # ----------------------------------------------------------------------
    feature_path, target_angles, meta = preprocess_to_cache(
        mat_file=args.mat_file,
        cache_dir=args.cache_dir,
        force=args.force_preprocess,
    )

    n_setups = int(target_angles.shape[0])

    # ----------------------------------------------------------------------
    # 2. Paper 80/10/10 split, but by SETUP to prevent SNR leakage.
    # ----------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_setups)

    n_train = int(0.80 * n_setups)
    n_val = int(0.10 * n_setups)

    train_setups = perm[:n_train]
    val_setups = perm[n_train:n_train + n_val]
    test_setups = perm[n_train + n_val:]

    split = {
        "train": train_setups,
        "val": val_setups,
        "test": test_setups,
    }

    print(
        f"[SPLIT] setups train/val/test = "
        f"{len(train_setups)}/{len(val_setups)}/{len(test_setups)}"
    )

    # ----------------------------------------------------------------------
    # 3. Angular classes.
    # ----------------------------------------------------------------------
    grid = build_angle_grid(
        targets=target_angles,
        resolution_deg=float(args.grid_resolution),
        angle_min=args.angle_min,
        angle_max=args.angle_max,
    )

    labels = encode_multihot_targets(
        target_angles=target_angles,
        grid=grid,
    )

    save_run_config(
        args=args,
        grid=grid,
        split=split,
        output_dir=output_dir,
    )

    # ----------------------------------------------------------------------
    # 4. Device
    # ----------------------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[DEVICE] {device}")

    if device.type == "cuda":
        print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[DEVICE] VRAM: {total_mem:.2f} GiB")

    model_probe = PaperDOAResNet(n_classes=len(grid))
    n_params = sum(p.numel() for p in model_probe.parameters())
    del model_probe
    print(f"[MODEL] parameters: {n_params:,} ({n_params/1e6:.3f} M)")

    # ----------------------------------------------------------------------
    # 5. Train/evaluate.
    # ----------------------------------------------------------------------
    all_test_rows = []

    if args.train_mode == "per_snr":
        train_jobs = [
            {
                "name": f"snr_{snr_values[i]:+g}dB".replace("+", "p").replace("-", "m"),
                "train_snrs": [i],
                "eval_snrs": [i],
                "display": f"SNR {snr_values[i]:+g} dB",
            }
            for i in range(N_SNRS)
        ]
    else:
        train_jobs = [
            {
                "name": "mixed_all_snrs",
                "train_snrs": list(range(N_SNRS)),
                "eval_snrs": list(range(N_SNRS)),
                "display": "mixed SNR model",
            }
        ]

    for job_number, job in enumerate(train_jobs, start=1):
        print("\n" + "#" * 78)
        print(
            f"JOB {job_number}/{len(train_jobs)}: {job['display']}"
        )
        print("#" * 78)

        train_ds = CorrelationDataset(
            feature_path=feature_path,
            labels=labels,
            true_angles=target_angles,
            setup_indices=train_setups,
            snr_indices=job["train_snrs"],
        )

        val_ds = CorrelationDataset(
            feature_path=feature_path,
            labels=labels,
            true_angles=target_angles,
            setup_indices=val_setups,
            snr_indices=job["train_snrs"],
        )

        test_ds = CorrelationDataset(
            feature_path=feature_path,
            labels=labels,
            true_angles=target_angles,
            setup_indices=test_setups,
            snr_indices=job["eval_snrs"],
        )

        train_loader = make_loader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            device=device,
        )

        val_loader = make_loader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            device=device,
        )

        test_loader = make_loader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            device=device,
        )

        print(
            f"[DATA] train/val/test observations = "
            f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)}"
        )

        model = PaperDOAResNet(
            n_classes=len(grid)
        ).to(device)

        checkpoint_path = output_dir / f"{job['name']}_final.pt"

        checkpoint_meta = {
            "job": job,
            "grid": grid.tolist(),
            "grid_resolution": float(args.grid_resolution),
            "train_setups": train_setups.tolist(),
            "val_setups": val_setups.tolist(),
            "test_setups": test_setups.tolist(),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "paper_cross_entropy": "unnormalized multi-hot CE",
        }

        if checkpoint_path.exists() and not args.retrain:
            print(f"[CKPT] Loading existing model: {checkpoint_path}")

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

            saved_grid = np.asarray(
                checkpoint["meta"]["grid"],
                dtype=np.float32,
            )

            if (
                saved_grid.shape != grid.shape
                or not np.allclose(saved_grid, grid, atol=1e-6)
            ):
                raise ValueError(
                    f"Checkpoint grid does not match current grid. "
                    f"Use --retrain or a different output directory."
                )

            state = (
                checkpoint["best_state"]
                if args.use_best_val
                else checkpoint["model_state"]
            )

            model.load_state_dict(state)
            model.to(device)

        else:
            try:
                train_one_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    epochs=args.epochs,
                    lr=args.lr,
                    use_amp=args.amp,
                    checkpoint_path=checkpoint_path,
                    checkpoint_meta=checkpoint_meta,
                )
            except torch.cuda.OutOfMemoryError:
                print(
                    "\nCUDA OUT OF MEMORY.\n"
                    "The paper batch size is 1024, which may exceed your GPU "
                    "memory with the 512-channel first layer.\n"
                    "Re-run with, for example:\n"
                    f"  python {Path(__file__).name} --mat-file "
                    f"\"{args.mat_file}\" --batch-size 256 --amp\n"
                )
                raise

            # train_one_model leaves model at the final epoch state.
            if args.use_best_val:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
                model.load_state_dict(checkpoint["best_state"])
                model.to(device)

        # Evaluation
        test_rows = evaluate_model(
            model=model,
            loader=test_loader,
            device=device,
            grid=grid,
            snr_values=snr_values,
            min_output_separation=float(args.min_output_separation),
            use_amp=args.amp,
        )

        all_test_rows.extend(test_rows)

        # Free model before the next per-SNR job.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ----------------------------------------------------------------------
    # 6. Final requested metric.
    # ----------------------------------------------------------------------
    all_test_rows.sort(
        key=lambda r: (r["snr_idx"], r["setup_idx"])
    )

    prediction_csv = output_dir / "test_predictions.csv"
    summary_csv = output_dir / "test_metric_summary.csv"

    write_prediction_csv(
        all_test_rows,
        prediction_csv,
    )

    final_metric = summarize_results(
        rows=all_test_rows,
        snr_values=snr_values,
        summary_path=summary_csv,
    )

    print(f"\n[OUTPUT] predictions : {prediction_csv}")
    print(f"[OUTPUT] summary     : {summary_csv}")
    print(f"[OUTPUT] config      : {output_dir / 'run_config.json'}")
    print(f"[DONE] final requested metric = {final_metric:.9f}")


if __name__ == "__main__":
    main()
