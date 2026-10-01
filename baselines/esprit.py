
import glob
import json
import os
import time
import numpy as np

class CFG:
    DATA_PATH = r"" #dataset dir
    OUT_DIR = r"" #output dir
    SNR_DB = [-20, -10, 0, 10, 20, 30, 40]
    SPATIAL_SMOOTHING = True
    SUBARRAY_SIZE = None
    D_OVER_LAMBDA = 0.5
    K = None
    N_SNAPSHOTS = 50
    DIAG_LOADING = 1e-3
    N_TEST_SETUPS = None
    SINGLE_SNAPSHOT_DRAWS = 5

def find_data_path():
    if CFG.DATA_PATH and os.path.exists(CFG.DATA_PATH):
        return CFG.DATA_PATH
    cands = sorted(glob.glob("/kaggle/input/**/*.mat", recursive=True), key=os.path.getsize, reverse=True)
    if not cands:
        cands = sorted(glob.glob("*.mat") + glob.glob("data/*.mat"), key=os.path.getsize, reverse=True)
    if not cands:
        raise FileNotFoundError("No mat file found")
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
    y = _to_complex64(y)
    az = np.squeeze(np.asarray(az, dtype=np.float64))
    if y.ndim != 4:
        raise ValueError(f"Expected antennas x snapshots x SNR x setups, got {y.shape}")
    n_setups = y.shape[-1]
    if az.ndim == 1:
        az = az[:, None] if n_setups > 1 else az.reshape(1, -1)
    elif az.ndim == 2 and az.shape[1] == n_setups:
        az = az.T
    if az.shape[0] != n_setups:
        raise ValueError("Azimuth labels do not match setups")
    return y, az.astype(np.float32)

def esprit_covariance(X, diag_loading=1e-3, forward_backward=True, spatial_smoothing=True):
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
    scale = np.trace(R).real / M_eff
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    R += diag_loading * scale * np.eye(M_eff)
    return R

def esprit_doa(X, K):
    R = esprit_covariance(
        X,
        CFG.DIAG_LOADING,
        forward_backward=True,
        spatial_smoothing=CFG.SPATIAL_SMOOTHING
    )
    M = R.shape[0]
    if K >= M:
        raise ValueError("K must be smaller than number of antennas")
    _, eigvecs = np.linalg.eigh(R)
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
    sin_theta = phase / (2 * np.pi * CFG.D_OVER_LAMBDA)
    sin_theta = np.clip(sin_theta, -1, 1)
    theta = np.arcsin(sin_theta)
    return np.sort(theta)

def snapshot_score_db(est, truth):
    est = np.sort(np.asarray(est))
    truth = np.sort(np.asarray(truth))
    rmse = np.sqrt(np.mean(np.degrees(est - truth) ** 2))
    return 10 * np.log10(max(rmse, 1e-12))

def evaluate_esprit(y, az, idx_te, snr_idx, mode, K):
    setups = idx_te if CFG.N_TEST_SETUPS is None else idx_te[:CFG.N_TEST_SETUPS]
    scores = []
    for s in setups:
        truth = az[s]
        if mode == "multi":
            L = y.shape[1] if CFG.N_SNAPSHOTS is None else min(CFG.N_SNAPSHOTS, y.shape[1])
            X = y[:, :, snr_idx, s][:, :L]
            est = esprit_doa(X.astype(np.complex128), K)
            scores.append(snapshot_score_db(est, truth))
        elif mode == "single":
            draws = min(CFG.SINGLE_SNAPSHOT_DRAWS, y.shape[1])
            for n in range(draws):
                X = y[:, n:n + 1, snr_idx, s]
                est = esprit_doa(X.astype(np.complex128), K)
                scores.append(snapshot_score_db(est, truth))
    return float(np.mean(scores))

def compute_split(N):
    perm = np.random.RandomState(123).permutation(N)
    n_tr = int(0.05 * N)
    n_va = int(0.05 * N)
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

def main():
    os.makedirs(CFG.OUT_DIR, exist_ok=True)
    y, az = load_matfile(find_data_path())
    M = y.shape[0]
    N = y.shape[-1]
    K = az.shape[1] if CFG.K is None else CFG.K
    if K >= M:
        raise ValueError("Targets must be less than antennas")
    _, _, idx_te = compute_split(N)
    results = {}
    for snr_idx, snr in enumerate(CFG.SNR_DB):
        multi = evaluate_esprit(y, az, idx_te, snr_idx, "multi", K)
        single = evaluate_esprit(y, az, idx_te, snr_idx, "single", K)
        results[str(snr)] = {
            "unit": "dB",
            "number_targets": int(K),
            "number_antennas": int(M),
            "snapshots_used": CFG.N_SNAPSHOTS,
            "avg_10log10_rmse_deg_multi": multi,
            "avg_10log10_rmse_deg_single": single
        }
    with open(os.path.join(CFG.OUT_DIR, "esprit_rmse.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
