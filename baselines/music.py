
import glob
import json
import math
import os
import numpy as np
from scipy.signal import find_peaks

class CFG:
#set dataset & output dir before running
    DATA_PATH = r""
    OUT_DIR = r""
    SNR_DB = [-20, -10, 0, 10, 20, 30, 40]
    D_OVER_LAMBDA = 0.5
    SPATIAL_SMOOTHING = True
    SUBARRAY_SIZE = None
    K = None
    N_SNAPSHOTS = 50
    DIAG_LOADING = 1e-3
    GRID_POINTS = 361
    N_TEST_SETUPS = None
    SINGLE_SNAPSHOT_DRAWS = 5

def find_data_path():
    if CFG.DATA_PATH and os.path.exists(CFG.DATA_PATH):
        return CFG.DATA_PATH
    cands = sorted(glob.glob("/kaggle/input/**/*.mat", recursive=True), key=os.path.getsize, reverse=True)
    if not cands:
        cands = sorted(glob.glob("*.mat") + glob.glob("data/*.mat"), key=os.path.getsize, reverse=True)
    if not cands:
        raise FileNotFoundError("No .mat file found.")
    return cands[0]

def _to_complex64(a):
    if a.dtype.names and {"real", "imag"} <= set(a.dtype.names):
        a = a["real"].astype(np.float32) + 1j * a["imag"].astype(np.float32)
    return np.ascontiguousarray(a).astype(np.complex64, copy=False)

def load_matfile(path):
    try:
        from scipy.io import loadmat
        d = loadmat(path)
        y = d["y_receive_clean"]
        az = d["target_azimuth"]
    except NotImplementedError:
        import h5py
        with h5py.File(path, "r") as f:
            y = _to_complex64(np.array(f["y_receive_clean"])).transpose()
            az = np.array(f["target_azimuth"]).transpose()
    y = _to_complex64(np.asarray(y))
    az = np.squeeze(np.asarray(az, dtype=np.float64))
    if y.ndim != 4:
        raise ValueError(f"Expected antennas x snapshots x SNR x setups, got {y.shape}")
    n_setups = y.shape[-1]
    if az.ndim == 1:
        az = az[:, None] if n_setups > 1 else az.reshape(1, -1)
    elif az.ndim == 2 and az.shape[1] == n_setups:
        az = az.T
    if az.shape[0] != n_setups:
        raise ValueError("Azimuth labels do not match setups.")
    return y, az.astype(np.float32)

def compute_split(N):
    perm = np.random.RandomState(123).permutation(N)
    n_tr = int(0.05 * N)
    n_va = int(0.05 * N)
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

def steering_vector(theta, n_elem, d_over_lambda=0.5):
    n = np.arange(n_elem)[:, None]
    return np.exp(1j * 2 * np.pi * d_over_lambda * n * np.sin(theta)[None, :])

def conventional_covariance(X, diag_loading=1e-3, forward_backward=True, spatial_smoothing=True):
    M, L = X.shape
    if spatial_smoothing:
        P = M // 2 + 1 if CFG.SUBARRAY_SIZE is None else CFG.SUBARRAY_SIZE
        if P <= 1 or P > M:
            raise ValueError("Invalid subarray size")
        n_subarrays = M - P + 1
        R = np.zeros((P, P), dtype=np.complex128)
        for i in range(n_subarrays):
            Xi = X[i:i + P, :]
            R += (Xi @ Xi.conj().T) / L
        R /= n_subarrays
        M_eff = P
    else:
        R = (X @ X.conj().T) / L
        M_eff = M
    R = 0.5 * (R + R.conj().T)
    if forward_backward:
        J = np.fliplr(np.eye(M_eff))
        R = 0.5 * (R + J @ R.conj() @ J)
        R = 0.5 * (R + R.conj().T)
    scale = np.trace(R).real / M_eff
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    R += diag_loading * scale * np.eye(M_eff)
    return R

def music_spectrum(R, K, angle_grid):
    M = R.shape[0]
    if K >= M:
        raise ValueError("MUSIC requires K < number of antennas.")
    eigvals, eigvecs = np.linalg.eigh(R)
    En = eigvecs[:, np.argsort(eigvals)[:M - K]]
    A = steering_vector(angle_grid, M, CFG.D_OVER_LAMBDA)
    denom = np.sum(np.abs(En.conj().T @ A) ** 2, axis=0)
    return 1 / (denom + 1e-12)

def pick_peaks(spec, grid, K):
    peaks, _ = find_peaks(spec)
    if len(peaks):
        peaks = peaks[np.argsort(spec[peaks])[::-1]]
    else:
        peaks = np.argsort(spec)[::-1]
    selected = []
    min_sep = max(3, len(grid) // 200)
    for p in peaks:
        if all(abs(int(p) - q) > min_sep for q in selected):
            selected.append(int(p))
        if len(selected) == K:
            break
    return np.sort(grid[selected])

def music_doa(X, K, angle_grid):
    R = conventional_covariance(
        X,
        CFG.DIAG_LOADING,
        forward_backward=True,
        spatial_smoothing=CFG.SPATIAL_SMOOTHING
    )
    spec = music_spectrum(R, K, angle_grid)
    return pick_peaks(spec, angle_grid, K)

def snapshot_score_db(est, truth):
    est = np.sort(np.asarray(est))
    truth = np.sort(np.asarray(truth))
    rmse = np.sqrt(np.mean(np.degrees(est - truth) ** 2))
    return 10 * np.log10(max(rmse, 1e-12))

def evaluate_music(y, az, idx_te, snr_idx, grid, mode, K):
    setups = idx_te if CFG.N_TEST_SETUPS is None else idx_te[:CFG.N_TEST_SETUPS]
    scores = []
    for s in setups:
        truth = az[s]
        if mode == "multi":
            L = y.shape[1] if CFG.N_SNAPSHOTS is None else min(CFG.N_SNAPSHOTS, y.shape[1])
            X = y[:, :L, snr_idx, s].astype(np.complex128)
            scores.append(snapshot_score_db(music_doa(X, K, grid), truth))
        elif mode == "single":
            draws = min(CFG.SINGLE_SNAPSHOT_DRAWS, y.shape[1])
            for n in range(draws):
                X = y[:, n:n + 1, snr_idx, s].astype(np.complex128)
                scores.append(snapshot_score_db(music_doa(X, K, grid), truth))
    return float(np.mean(scores))

def main():
    os.makedirs(CFG.OUT_DIR, exist_ok=True)
    y, az = load_matfile(find_data_path())
    M = y.shape[0]
    N = y.shape[-1]
    K = az.shape[1] if CFG.K is None else CFG.K
    if K >= M:
        raise ValueError("Number of targets must be smaller than antennas.")
    _, _, idx_te = compute_split(N)
    grid = np.linspace(-math.pi / 2, math.pi / 2, CFG.GRID_POINTS)
    results = {}
    for snr_idx, snr in enumerate(CFG.SNR_DB):
        multi = evaluate_music(y, az, idx_te, snr_idx, grid, "multi", K)
        single = evaluate_music(y, az, idx_te, snr_idx, grid, "single", K)
        results[str(snr)] = {
            "unit": "dB",
            "number_targets": int(K),
            "number_antennas": int(M),
            "snapshots_used": CFG.N_SNAPSHOTS,
            "avg_10log10_rmse_deg_multi": multi,
            "avg_10log10_rmse_deg_single": single
        }
    with open(os.path.join(CFG.OUT_DIR, "music_rmse.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
