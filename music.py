import glob
import json
import math
import os
import time

import numpy as np
from scipy.signal import find_peaks


class CFG:
    DATA_PATH = r"C:\Users\ugp.DESKTOP-7Q13T9G\Downloads\meanflow_gdm_attention\meanflow_gdm_attention\receive_echo_dataset.mat"
    OUT_DIR = r"C:\Users\ugp.DESKTOP-7Q13T9G\Downloads\meanflow_gdm_attention\meanflow_gdm_attention\training_output_legacy"
    SNR_DB = [-20, -10, 0, 10, 20, 30,40]
    D_OVER_LAMBDA = 0.5
    K = None
    SUBARRAY_LEN = None
    DIAG_LOADING = 1e-3
    GRID_POINTS = 3001

    N_TEST_SETUPS = None
    SINGLE_SNAPSHOT_DRAWS = 5

    NN_RESULTS_JSON = "test_rmse.json"


def find_data_path():
    if CFG.DATA_PATH:
        return CFG.DATA_PATH

    cands = sorted(
        glob.glob("/kaggle/input/**/*.mat", recursive=True),
        key=os.path.getsize,
        reverse=True,
    )

    if not cands:
        cands = sorted(
            glob.glob("*.mat") + glob.glob("data/*.mat"),
            key=os.path.getsize,
            reverse=True,
        )

    if not cands:
        raise FileNotFoundError(
            "No .mat file found. Attach your dataset or set CFG.DATA_PATH."
        )

    print(f"[data] using {cands[0]} " f"({os.path.getsize(cands[0]) / 1e6:.0f} MB)")

    return cands[0]


def _to_complex64(a):
    if a.dtype.names and {"real", "imag"} <= set(a.dtype.names):
        a = a["real"].astype(np.float32) + 1j * a["imag"].astype(np.float32)

    return np.ascontiguousarray(a).astype(np.complex64, copy=False)


def load_matfile(path):
    try:
        from scipy.io import loadmat

        d = loadmat(path)

        y = d["y_receive_ultra_clean"]
        az = d["target_azimuth"]

    except NotImplementedError:
        import h5py

        with h5py.File(path, "r") as f:
            y = _to_complex64(np.array(f["y_receive_ultra_clean"])).transpose()

            az = np.array(f["target_azimuth"]).transpose()

    y = _to_complex64(np.asarray(y))

    az = np.squeeze(np.asarray(az, dtype=np.float64))

    if y.ndim != 4:
        raise ValueError(
            f"Unexpected y_receive shape {y.shape}. "
            f"Expected antennas x snapshots x SNR x setups."
        )

    n_setups = y.shape[-1]

    if az.ndim == 0:
        if n_setups != 1:
            raise ValueError(
                f"Cannot interpret target_azimuth shape {az.shape} "
                f"for {n_setups} setups."
            )

        az = np.array([[float(az)]], dtype=np.float64)

    elif az.ndim == 1:
        if n_setups == 1:
            az = az.reshape(1, -1)

        elif az.shape[0] == n_setups:
            az = az[:, None]

        else:
            raise ValueError(
                f"Cannot interpret target_azimuth shape {az.shape} "
                f"for {n_setups} setups."
            )

    elif az.ndim == 2:
        if az.shape[0] == n_setups:
            pass

        elif az.shape[1] == n_setups:
            az = az.T

        else:
            raise ValueError(
                f"Cannot interpret target_azimuth shape {az.shape} "
                f"for {n_setups} setups."
            )

    else:
        raise ValueError(f"Cannot interpret target_azimuth shape {az.shape}.")

    if az.shape[0] != n_setups:
        raise ValueError(
            f"Number of target-label setups ({az.shape[0]}) "
            f"does not match y_receive setups ({n_setups})."
        )

    if not np.all(np.isfinite(az)):
        raise ValueError("target_azimuth contains NaN or infinite values.")

    amax = float(np.abs(az).max())

    if amax > math.pi / 2 + 1e-3:
        raise ValueError(
            f"Labels expected in radians within [-pi/2, pi/2]. "
            f"Got max |angle| = {amax:.4f}."
        )

    print(f"[data] y_receive shape: {y.shape}")

    print(f"[data] target_azimuth shape after formatting: {az.shape}")

    print(f"[data] detected targets: K={az.shape[1]}")

    print(f"[data] azimuth range: " f"[{az.min():.4f}, {az.max():.4f}] rad")

    return y, az.astype(np.float32)


def compute_split(N):
    perm = np.random.RandomState(123).permutation(N)

    n_tr = int(0.05 * N)

    n_va = int(0.05 * N)

    idx_tr = perm[:n_tr]

    idx_va = perm[n_tr : n_tr + n_va]

    idx_te = perm[n_tr + n_va :]

    return (idx_tr, idx_va, idx_te)


def steering_vector(theta, n_elem, d_over_lambda=0.5):
    theta = np.asarray(theta, dtype=np.float64)

    n = np.arange(n_elem)[:, None]

    return np.exp(1j * 2 * np.pi * d_over_lambda * n * np.sin(theta)[None, :])


def fbss_covariance(X, p, d_over_lambda=0.5, diag_loading=1e-3):
    M, L = X.shape

    if p > M:
        raise ValueError(f"Subarray length p={p} cannot exceed M={M} antennas.")

    if p < 1:
        raise ValueError("Subarray length must be positive.")

    Q = M - p + 1

    Rf = np.zeros((p, p), dtype=np.complex128)

    for i in range(Q):
        Xi = X[i : i + p, :]

        Rf += (Xi @ Xi.conj().T) / L

    Rf /= Q

    J = np.eye(M, dtype=np.complex128)[::-1]

    Xb = J @ X.conj()

    Rb = np.zeros((p, p), dtype=np.complex128)

    for i in range(Q):
        Xi = Xb[i : i + p, :]

        Rb += (Xi @ Xi.conj().T) / L

    Rb /= Q

    R = 0.5 * (Rf + Rb)

    R = 0.5 * (R + R.conj().T)

    scale = np.trace(R).real / p

    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    R += diag_loading * scale * np.eye(p, dtype=np.complex128)

    return R


def music_spectrum(R, K, angle_grid, d_over_lambda=0.5):
    p = R.shape[0]

    if K >= p:
        raise ValueError(f"MUSIC requires K < p. Got K={K}, p={p}.")

    eigvals, eigvecs = np.linalg.eigh(R)

    order = np.argsort(eigvals)

    eigvecs = eigvecs[:, order]

    En = eigvecs[:, : p - K]

    A_grid = steering_vector(angle_grid, p, d_over_lambda)

    proj = En.conj().T @ A_grid

    denom = np.sum(np.abs(proj) ** 2, axis=0)

    spectrum = 1.0 / (denom + 1e-12)

    return spectrum


def pick_peaks(spectrum, angle_grid, K):
    if K <= 0:
        raise ValueError("K must be positive.")

    if K > len(angle_grid):
        raise ValueError(f"K={K} exceeds number of grid points " f"{len(angle_grid)}.")

    spectrum = np.asarray(spectrum)

    peaks, _ = find_peaks(spectrum)

    selected = []

    min_sep = max(3, len(angle_grid) // 200)

    if len(peaks) > 0:
        peak_order = peaks[np.argsort(spectrum[peaks])[::-1]]

        for idx in peak_order:
            idx = int(idx)

            if all(abs(idx - old) > min_sep for old in selected):
                selected.append(idx)

            if len(selected) == K:
                break

    if len(selected) < K:
        order = np.argsort(spectrum)[::-1]

        for idx in order:
            idx = int(idx)

            if idx in selected:
                continue

            if all(abs(idx - old) > min_sep for old in selected):
                selected.append(idx)

            if len(selected) == K:
                break

    if len(selected) < K:
        order = np.argsort(spectrum)[::-1]

        for idx in order:
            idx = int(idx)

            if idx not in selected:
                selected.append(idx)

            if len(selected) == K:
                break

    selected = np.asarray(selected[:K], dtype=int)

    estimates = np.sort(angle_grid[selected])

    if estimates.shape[0] != K:
        raise RuntimeError(
            f"Peak selection returned {estimates.shape[0]} " f"angles instead of K={K}."
        )

    return estimates


def music_doa(X, K, p, angle_grid, d_over_lambda=0.5, diag_loading=1e-3):
    R = fbss_covariance(X, p, d_over_lambda, diag_loading)

    spec = music_spectrum(R, K, angle_grid, d_over_lambda)

    est = pick_peaks(spec, angle_grid, K)

    return est


def snapshot_score_db(est_rad, tru_rad):
    est_rad = np.sort(np.asarray(est_rad, dtype=np.float64))

    tru_rad = np.sort(np.asarray(tru_rad, dtype=np.float64))

    if est_rad.shape != tru_rad.shape:
        raise ValueError(
            f"Estimate shape {est_rad.shape} does not match "
            f"truth shape {tru_rad.shape}."
        )

    err_deg = np.degrees(est_rad - tru_rad)

    rmse_deg = np.sqrt(np.mean(err_deg**2))

    return 10.0 * math.log10(max(float(rmse_deg), 1e-12))


def evaluate_music(y, az, idx_te, snr_idx, angle_grid, mode, K, p):
    if CFG.N_TEST_SETUPS is None:
        setups = idx_te
    else:
        setups = idx_te[: CFG.N_TEST_SETUPS]

    if len(setups) == 0:
        raise ValueError("No test setups available.")

    scores_db = []

    n_snapshots = y.shape[1]

    for s in setups:
        tru = np.sort(az[s].astype(np.float64))

        if tru.shape[0] != K:
            raise ValueError(
                f"Setup {s} contains {tru.shape[0]} target labels " f"but K={K}."
            )

        if mode == "multi":
            X = y[:, :, snr_idx, s].astype(np.complex128)

            est = music_doa(X, K, p, angle_grid, CFG.D_OVER_LAMBDA, CFG.DIAG_LOADING)

            scores_db.append(snapshot_score_db(est, tru))

        elif mode == "single":
            n_draws = min(CFG.SINGLE_SNAPSHOT_DRAWS, n_snapshots)

            for n in range(n_draws):
                X = y[:, n : n + 1, snr_idx, s].astype(np.complex128)

                est = music_doa(
                    X, K, p, angle_grid, CFG.D_OVER_LAMBDA, CFG.DIAG_LOADING
                )

                scores_db.append(snapshot_score_db(est, tru))

        else:
            raise ValueError(f"Unknown evaluation mode: {mode}")

    return float(np.mean(scores_db))


def plot_comparison(results):
    import matplotlib.pyplot as plt

    xs = CFG.SNR_DB

    single = [results[str(s)]["avg_10log10_rmse_deg_single"] for s in xs]

    multi = [results[str(s)]["avg_10log10_rmse_deg_multi"] for s in xs]

    plt.figure(figsize=(7, 4.5))

    plt.plot(xs, single, "s--", lw=2, label="MUSIC, 1 snapshot + FBSS")

    plt.plot(xs, multi, "o-", lw=2, label="MUSIC, all snapshots + FBSS")

    nn_path = os.path.join(CFG.OUT_DIR, CFG.NN_RESULTS_JSON)

    if os.path.exists(nn_path):
        try:
            with open(nn_path, "r") as f:
                nn = json.load(f)

            if "avg_10log10_rmse_deg" in nn:
                nn_metric = nn["avg_10log10_rmse_deg"]

                keys = [f"{s}dB" for s in xs]

                if all(key in nn_metric for key in keys):
                    nn_db = [nn_metric[key] for key in keys]

                    plt.plot(
                        xs,
                        nn_db,
                        "^-",
                        lw=2,
                        label="Joint generative model, 1 snapshot",
                    )

        except Exception as e:
            print(f"[warn] could not overlay NN results: {e}")

    plt.xlabel("SNR (dB)")

    plt.ylabel("Average 10*log10(RMSE in degrees)")

    plt.title("DoA accuracy vs SNR")

    plt.grid(True, alpha=0.4)

    plt.legend()

    plt.xticks(xs)

    path = os.path.join(CFG.OUT_DIR, "music_vs_nn_rmse.png")

    plt.savefig(path, dpi=150, bbox_inches="tight")

    print(f"[plot] saved {path}")

    try:
        plt.show()
    except Exception:
        pass


def main():
    t0 = time.time()

    os.makedirs(CFG.OUT_DIR, exist_ok=True)

    print(f"[MUSIC] D_OVER_LAMBDA={CFG.D_OVER_LAMBDA}")

    y, az = load_matfile(find_data_path())

    M = y.shape[0]
    N = y.shape[-1]
    n_snr = y.shape[2]

    if n_snr != len(CFG.SNR_DB):
        raise ValueError(
            f"Dataset contains {n_snr} SNR points but "
            f"CFG.SNR_DB contains {len(CFG.SNR_DB)} values."
        )

    K_data = az.shape[1]

    if CFG.K is None:
        K = K_data
    else:
        K = int(CFG.K)

    if K != K_data:
        raise ValueError(
            f"CFG.K={K}, but dataset contains K={K_data} targets. "
            f"Set CFG.K=None for automatic detection."
        )

    if K < 1:
        raise ValueError("Number of targets must be at least 1.")

    if K >= M:
        raise ValueError(
            f"MUSIC requires fewer targets than antennas. " f"Got K={K}, M={M}."
        )

    if CFG.SUBARRAY_LEN is None:
        p = M - K + 1

        if p <= K:
            raise ValueError(
                f"FBSS cannot support K={K} coherent targets "
                f"with M={M} antennas using this subarray rule. "
                f"Need approximately 2*K <= M."
            )

    else:
        p = int(CFG.SUBARRAY_LEN)

        if p <= K:
            raise ValueError(f"Need SUBARRAY_LEN > K. " f"Got p={p}, K={K}.")

        if p > M:
            raise ValueError(
                f"SUBARRAY_LEN cannot exceed number of antennas. " f"Got p={p}, M={M}."
            )

    Q = M - p + 1

    if K > 1 and Q < K:
        print(
            f"[warn] Q={Q} spatial subarrays for K={K} targets. "
            f"Full coherent-source rank restoration normally requires Q>=K."
        )

    print(f"[MUSIC] antennas M={M}")

    print(f"[MUSIC] targets K={K}")

    print(f"[MUSIC] subarray length p={p}")

    print(f"[MUSIC] number of forward subarrays Q={Q}")

    idx_tr, idx_va, idx_te = compute_split(N)

    print(f"[split] train setups: {len(idx_tr)}")

    print(f"[split] validation setups: {len(idx_va)}")

    print(f"[split] test setups: {len(idx_te)}")

    angle_grid = np.linspace(-math.pi / 2, math.pi / 2, CFG.GRID_POINTS)

    results = {}

    for snr_idx, snrdb in enumerate(CFG.SNR_DB):
        metric_multi = evaluate_music(
            y, az, idx_te, snr_idx, angle_grid, mode="multi", K=K, p=p
        )

        metric_single = evaluate_music(
            y, az, idx_te, snr_idx, angle_grid, mode="single", K=K, p=p
        )

        results[str(snrdb)] = {
            "unit": "dB",
            "number_targets": K,
            "number_antennas": M,
            "subarray_length": p,
            "number_subarrays": Q,
            "definition": (f"mean_over_examples[" f"10*log10(RMSE_deg_over_{K}_DoAs)]"),
            "avg_10log10_rmse_deg_multi": metric_multi,
            "avg_10log10_rmse_deg_single": metric_single,
        }

        print(
            f"[MUSIC {snrdb:>4} dB] "
            f"single: {metric_single:7.3f}   "
            f"multi(L={y.shape[1]}): {metric_multi:7.3f}"
        )

    out_path = os.path.join(CFG.OUT_DIR, "music_rmse.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[saved] {out_path}")

    plot_comparison(results)

    if len(idx_te) > 0:
        s0 = idx_te[0]

        k0 = min(3, len(CFG.SNR_DB) - 1)

        X_demo = y[:, 0:1, k0, s0].astype(np.complex128)

        est = music_doa(X_demo, K, p, angle_grid, CFG.D_OVER_LAMBDA, CFG.DIAG_LOADING)

        tru = np.sort(az[s0])

        print(f"\n[demo, {CFG.SNR_DB[k0]} dB] " f"truth (rad): {np.round(tru, 4)}")

        print(f"[demo, {CFG.SNR_DB[k0]} dB] " f"MUSIC (rad): {np.round(est, 4)}")

    print(f"[done] {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()
