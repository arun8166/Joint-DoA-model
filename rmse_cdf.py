import csv
import importlib.util
import math
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# ============================================================
# USER SETTINGS
# ============================================================

# Set this to the .py file containing the code you used for training.
# Importing it is safe because your training code uses:
#     if __name__ == "__main__":
#         main()
TRAIN_SCRIPT = r"C:\Users\ugp.DESKTOP-7Q13T9G\Downloads\meanflow_gdm_attention\meanflow_gdm_attention\meanflow_gdm_attention.py"

# Leave as None to use CFG.DATA_PATH from the training script.
DATA_PATH = None

# Leave as None to use os.path.join(CFG.OUT_DIR, CFG.CKPT).
CKPT_PATH = None

# Leave as None to create <CFG.OUT_DIR>\rmse_cdf
OUT_DIR = None

# Posterior inference settings.
# None means: use CFG.TEST_K / TEST_M / TEST_STEPS from the training script.
TEST_K = None
TEST_M = None
TEST_STEPS = None

# Number of snapshots per test setup to use.
# None means use CFG.TEST_SNAPSHOTS, capped by the number available.
# To use every available snapshot, set:
#     N_SNAPSHOTS = "all"
N_SNAPSHOTS = None

# Inference batch size.
# Reduce this if CUDA runs out of memory.
CHUNK = 64

# Save plot also as PDF.
SAVE_PDF = True

# Optional x-axis clipping for display only.
# The CSV always keeps every RMSE value.
CDF_XMAX_DEG = None


# ============================================================
# UTILITIES
# ============================================================

def load_training_module(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Training script not found:\n{path}\n\n"
            "Set TRAIN_SCRIPT at the top of this helper to your original .py file."
        )

    spec = importlib.util.spec_from_file_location("doa_training_code", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def empirical_cdf(values):
    """Return sorted samples x and F_hat(x)=i/N."""
    x = np.sort(np.asarray(values, dtype=np.float64))
    n = len(x)
    if n == 0:
        return x, np.empty(0, dtype=np.float64)
    f = np.arange(1, n + 1, dtype=np.float64) / n
    return x, f


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


# ============================================================
# EVALUATION
# ============================================================

@torch.inference_mode()
def collect_rmse_samples(
    T,
    model,
    dset,
    global_test_indices,
    device,
    snr_db_values,
    snapshots,
    K,
    M,
    steps,
    chunk=64,
):
    """
    Match the prediction rule used in the original evaluate():

        posterior_samples(...)
        -> clamp u
        -> asin(u)
        -> sort target angles
        -> median across posterior samples
        -> RMSE across N_ANGLES

    Returns a list of dictionaries, one row per
    (test setup, SNR, snapshot).
    """

    model.eval()

    items = [
        (s, k, n)
        for s in range(dset.S)
        for k in range(dset.N_SNR)
        for n in snapshots
    ]

    rows = []

    for c0 in tqdm(
        range(0, len(items), chunk),
        desc="CDF evaluation",
        leave=True,
    ):
        sub = items[c0:c0 + chunk]

        xs = []
        pls = []
        truths_rad = []
        meta = []

        for s, k, n in sub:
            idx = np.ravel_multi_index(
                (s, k, n),
                (dset.S, dset.N_SNR, dset.N_SNAP),
            )

            b = dset[int(idx)]

            xs.append(b["x"])
            pls.append(b["pl"])

            # dset returns u = sin(theta).
            # For test mode, u is already sorted, but sort again explicitly.
            u_true = np.sort(b["u"].numpy())
            theta_true_rad = np.arcsin(
                np.clip(u_true, -0.999999, 0.999999)
            )
            truths_rad.append(theta_true_rad)

            meta.append((s, k, n))

        x = torch.stack(xs).to(device, non_blocking=True)
        pl = torch.stack(pls).to(device, non_blocking=True)

        u_post = T.posterior_samples(
            model,
            x,
            pl,
            K,
            M,
            steps,
        ).clamp(-0.999, 0.999)

        # Same aggregation used in the supplied evaluate():
        # sort the four angles inside every posterior draw,
        # then take median across the K*M draws.
        theta_pred_rad = (
            torch.asin(u_post)
            .sort(-1).values
            .median(1).values
            .cpu()
            .numpy()
        )

        for j, (s, k, n) in enumerate(meta):
            err_deg = np.degrees(
                theta_pred_rad[j] - truths_rad[j]
            )

            rmse_deg = float(
                np.sqrt(np.mean(err_deg ** 2))
            )

            # Keep both dB definitions in the CSV.
            # legacy_score_db matches the supplied evaluate() exactly.
            legacy_score_db = 10.0 * math.log10(
                max(rmse_deg, 1e-12)
            )

            # Conventional amplitude/error conversion, included only
            # for reference. The CDF plot uses raw RMSE in degrees.
            rmse_db_20log10 = 20.0 * math.log10(
                max(rmse_deg, 1e-12)
            )

            rows.append(
                {
                    "snr_db": float(snr_db_values[k]),
                    "snr_index": int(k),
                    "setup_local_index": int(s),
                    "setup_global_index": int(global_test_indices[s]),
                    "snapshot_index": int(n),
                    "rmse_deg": rmse_deg,
                    "rmse_10log10_deg": legacy_score_db,
                    "rmse_20log10_deg": rmse_db_20log10,
                }
            )

    return rows


# ============================================================
# CSV SAVING
# ============================================================

def save_raw_csv(rows, path):
    fieldnames = [
        "snr_db",
        "snr_index",
        "setup_local_index",
        "setup_global_index",
        "snapshot_index",
        "rmse_10log10_deg",
        "rmse_deg",
        "rmse_20log10_deg",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_cdf_csv(rows, snr_db_values, path):
    """
    Save empirical CDF coordinates using:

        RMSE_dB = 10*log10(RMSE_deg)

    The saved x-values are ALREADY transformed.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "snr_db",
                "cdf_rank",
                "n_samples",
                "rmse_10log10_deg_sorted",
                "empirical_cdf",
            ]
        )

        for snr in snr_db_values:
            vals = [
                r["rmse_10log10_deg"]
                for r in rows
                if np.isclose(r["snr_db"], snr)
            ]

            x, F = empirical_cdf(vals)

            for i, (xx, ff) in enumerate(zip(x, F), start=1):
                writer.writerow(
                    [
                        float(snr),
                        i,
                        len(x),
                        float(xx),
                        float(ff),
                    ]
                )


def save_summary_csv(rows, snr_db_values, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "snr_db",
                "n_samples",
                "mean_rmse_10log10_deg",
                "median_rmse_10log10_deg",
                "p10_rmse_10log10_deg",
                "p25_rmse_10log10_deg",
                "p75_rmse_10log10_deg",
                "p90_rmse_10log10_deg",
                "p95_rmse_10log10_deg",
            ]
        )

        for snr in snr_db_values:
            vals = np.asarray(
                [
                    r["rmse_10log10_deg"]
                    for r in rows
                    if np.isclose(r["snr_db"], snr)
                ],
                dtype=np.float64,
            )

            writer.writerow(
                [
                    float(snr),
                    len(vals),
                    float(vals.mean()),
                    float(np.median(vals)),
                    percentile(vals, 10),
                    percentile(vals, 25),
                    percentile(vals, 75),
                    percentile(vals, 90),
                    percentile(vals, 95),
                ]
            )


# ============================================================
# CDF PLOT
# ============================================================

def plot_cdf(rows, snr_db_values, out_png, out_pdf=None):
    plt.figure(figsize=(7.2, 5.0))

    for snr in snr_db_values:
        vals = [
            r["rmse_10log10_deg"]
            for r in rows
            if np.isclose(r["snr_db"], snr)
        ]

        x, F = empirical_cdf(vals)

        plt.step(
            x,
            F,
            where="post",
            linewidth=1.8,
            label=f"{snr:g} dB",
        )

    plt.xlabel(r"$10\\log_{10}(\\mathrm{RMSE}_{\\mathrm{deg}})$")
    plt.ylabel("Empirical CDF")
    plt.title(r"CDF of $10\\log_{10}(\\mathrm{RMSE}_{\\mathrm{deg}})$ at each SNR")
    plt.ylim(0.0, 1.0)

    plt.grid(True, alpha=0.3)
    plt.legend(title="SNR")
    plt.tight_layout()

    plt.savefig(out_png, dpi=300, bbox_inches="tight")

    if out_pdf is not None:
        plt.savefig(out_pdf, bbox_inches="tight")

    print(f"[plot] saved: {out_png}")
    if out_pdf is not None:
        print(f"[plot] saved: {out_pdf}")

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    T = load_training_module(TRAIN_SCRIPT)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device = {device}")

    ckpt_path = (
        CKPT_PATH
        if CKPT_PATH is not None
        else os.path.join(T.CFG.OUT_DIR, T.CFG.CKPT)
    )

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{ckpt_path}"
        )

    out_dir = (
        OUT_DIR
        if OUT_DIR is not None
        else os.path.join(T.CFG.OUT_DIR, "rmse_cdf")
    )
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------------
    # Load checkpoint first so the exact saved test split and
    # normalization statistics are reused.
    # --------------------------------------------------------
    ck = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    if "split" not in ck or "test" not in ck["split"]:
        raise KeyError(
            "Checkpoint does not contain ck['split']['test']. "
            "Your supplied training code saves this split, so use the "
            "checkpoint created by that code."
        )

    stats = ck["stats"]
    idx_te = np.asarray(ck["split"]["test"], dtype=np.int64)

    model = T.JointModel(
        mf_scale=stats["mf_scale"]
    ).to(device)

    # Use EMA weights, same as final testing in the supplied code.
    model.load_state_dict(ck["ema"])
    model.eval()

    print(f"[checkpoint] {ckpt_path}")
    print(f"[split] test setups = {len(idx_te)}")

    # --------------------------------------------------------
    # Load dataset and create ONLY the saved test subset.
    # --------------------------------------------------------
    data_path = DATA_PATH if DATA_PATH is not None else T.find_data_path()
    y, yc, az = T.load_matfile(data_path)

    ds_te = T.DoaDataset(
        y,
        yc,
        az,
        idx_te,
        False,
        stats,
    )

    # SNR labels from training script.
    snr_db_values = list(T.SNR_DB)

    if len(snr_db_values) != ds_te.N_SNR:
        raise ValueError(
            f"SNR_DB has {len(snr_db_values)} entries but dataset has "
            f"{ds_te.N_SNR} SNR points.\n"
            "Update SNR_DB in the original training script to match the dataset."
        )

    K = T.CFG.TEST_K if TEST_K is None else int(TEST_K)
    M = T.CFG.TEST_M if TEST_M is None else int(TEST_M)
    steps = T.CFG.TEST_STEPS if TEST_STEPS is None else int(TEST_STEPS)

    if N_SNAPSHOTS == "all":
        n_use = ds_te.N_SNAP
    elif N_SNAPSHOTS is None:
        n_use = min(int(T.CFG.TEST_SNAPSHOTS), ds_te.N_SNAP)
    else:
        n_use = min(int(N_SNAPSHOTS), ds_te.N_SNAP)

    snapshots = tuple(range(n_use))

    print(
        f"[eval] SNR points = {snr_db_values}\n"
        f"[eval] snapshots/setup = {len(snapshots)}\n"
        f"[eval] posterior K={K}, M={M}, K*M={K*M}, DDIM steps={steps}\n"
        f"[eval] expected RMSE samples/SNR = "
        f"{len(idx_te)} x {len(snapshots)} = {len(idx_te)*len(snapshots)}"
    )

    rows = collect_rmse_samples(
        T=T,
        model=model,
        dset=ds_te,
        global_test_indices=idx_te,
        device=device,
        snr_db_values=snr_db_values,
        snapshots=snapshots,
        K=K,
        M=M,
        steps=steps,
        chunk=CHUNK,
    )

    # --------------------------------------------------------
    # Save CSV files.
    # --------------------------------------------------------
    raw_csv = os.path.join(
        out_dir,
        "rmse_cdf_samples.csv",
    )
    cdf_csv = os.path.join(
        out_dir,
        "rmse_cdf_points.csv",
    )
    summary_csv = os.path.join(
        out_dir,
        "rmse_cdf_summary.csv",
    )

    save_raw_csv(rows, raw_csv)
    save_cdf_csv(rows, snr_db_values, cdf_csv)
    save_summary_csv(rows, snr_db_values, summary_csv)

    print(f"[csv] saved: {raw_csv}")
    print(f"[csv] saved: {cdf_csv}")
    print(f"[csv] saved: {summary_csv}")

    # Print useful CDF summary.
    print("\n[CDF summary]")
    for snr in snr_db_values:
        vals = np.asarray(
            [
                r["rmse_deg"]
                for r in rows
                if np.isclose(r["snr_db"], snr)
            ],
            dtype=np.float64,
        )

        print(
            f"  {snr:>5g} dB | "
            f"N={len(vals):5d} | "
            f"median={np.median(vals):8.4f} deg | "
            f"p90={np.percentile(vals,90):8.4f} deg | "
            f"p95={np.percentile(vals,95):8.4f} deg"
        )

    # --------------------------------------------------------
    # Plot CDF.
    # --------------------------------------------------------
    out_png = os.path.join(
        out_dir,
        "rmse_cdf_by_snr.png",
    )
    out_pdf = (
        os.path.join(out_dir, "rmse_cdf_by_snr.pdf")
        if SAVE_PDF
        else None
    )

    plot_cdf(
        rows,
        snr_db_values,
        out_png,
        out_pdf,
    )

    print("\n[done]")


if __name__ == "__main__":
    main()