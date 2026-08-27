import os
import sys
import math
import glob
import importlib.util

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# ============================================================
# USER SETTINGS
# ============================================================

# If None, the helper automatically searches the current folder for the
# original .py file containing "class JointModel" and "def posterior_samples".

SOURCE_SCRIPT = r"C:\Users\ugp.DESKTOP-7Q13T9G\Downloads\meanflow_gdm_attention\meanflow_gdm_attention\meanflow_gdm_attention.py"


SNR_TO_PLOT = [30, 10, -10]

# Same test inference settings as your original script
K_DRAWS = 8
M_DRAWS = 16
INFERENCE_STEPS = 30

# Same number used in your original final test
TEST_SNAPSHOTS = 50

# 18 bins over [-90,90] = 10-degree bins
ANGLE_MIN_DEG = -90.0
ANGLE_MAX_DEG = 90.0
N_BINS = 18

CHUNK = 96
SEED = 0

FIG_NAME = "rmse_vs_actual_angle_histogram.png"
CSV_NAME = "rmse_vs_actual_angle_histogram.csv"
RAW_NAME = "rmse_vs_actual_angle_histogram_raw.npz"


# ============================================================
# FIND ORIGINAL TRAINING SCRIPT
# ============================================================

def find_source_script():
    if SOURCE_SCRIPT is not None:
        p = os.path.abspath(SOURCE_SCRIPT)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"SOURCE_SCRIPT not found:\n{p}")
        return p

    here = os.path.dirname(os.path.abspath(__file__))
    this_file = os.path.abspath(__file__)

    matches = []

    for p in sorted(glob.glob(os.path.join(here, "*.py"))):
        if os.path.abspath(p) == this_file:
            continue

        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
        except OSError:
            continue

        required = [
            "class JointModel",
            "def posterior_samples",
            "class DoaDataset",
            "def load_matfile",
            "class CFG",
        ]

        if all(token in txt for token in required):
            matches.append(p)

    if not matches:
        raise FileNotFoundError(
            "Could not automatically locate the original model script.\n"
            "Set SOURCE_SCRIPT manually near the top of this helper."
        )

    if len(matches) > 1:
        print("[source] Multiple candidate scripts found:")
        for p in matches:
            print("         ", p)
        print("[source] Using:", matches[0])

    return matches[0]


def import_source(path):
    spec = importlib.util.spec_from_file_location("doa_source", path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import:\n{path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ============================================================
# LOAD EMA MODEL + EXACT TEST SPLIT
# ============================================================

def load_model_and_testset(src, device):
    ckpt_path = os.path.join(src.CFG.OUT_DIR, src.CFG.CKPT)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{ckpt_path}"
        )

    ck = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False
    )

    if "stats" not in ck:
        raise KeyError("Checkpoint has no 'stats'.")

    if "split" not in ck or "test" not in ck["split"]:
        raise KeyError(
            "Checkpoint has no saved split['test']; "
            "cannot reconstruct the exact test setups."
        )

    stats = ck["stats"]
    idx_te = np.asarray(ck["split"]["test"], dtype=np.int64)

    model = src.JointModel(
        mf_scale=stats["mf_scale"]
    ).to(device)

    model.load_state_dict(ck["ema"])
    model.eval()

    y, yc, az = src.load_matfile(src.find_data_path())

    ds_te = src.DoaDataset(
        y,
        yc,
        az,
        idx_te,
        False,
        stats,
        getattr(src.CFG, "LABEL_GRID_RAD", 0.0),
    )

    return model, ds_te, ckpt_path


# ============================================================
# COLLECT MATCHED TARGET ERRORS
# ============================================================

@torch.no_grad()
def collect_results(src, model, ds_te, device):
    """
    Uses the SAME matching/aggregation convention as original evaluate():

       truth:
           arcsin(sort(u_true))

       prediction:
           posterior_samples(...)
           -> arcsin
           -> sort each posterior draw
           -> median across posterior draws

    Then ALL FOUR matched targets are flattened/pool together.
    """

    snr_axis = list(src.SNR_DB)

    for snr in SNR_TO_PLOT:
        if snr not in snr_axis:
            raise ValueError(
                f"{snr} dB not found in source SNR_DB={snr_axis}"
            )

    n_snap = min(
        int(TEST_SNAPSHOTS),
        int(ds_te.N_SNAP)
    )

    out = {}

    for snr_db in SNR_TO_PLOT:
        k = snr_axis.index(snr_db)

        items = [
            (setup, snap)
            for setup in range(ds_te.S)
            for snap in range(n_snap)
        ]

        truth_chunks = []
        pred_chunks = []
        setup_chunks = []
        snapshot_chunks = []

        print(
            f"\n[eval] {snr_db:+d} dB | "
            f"{ds_te.S} setups x {n_snap} snapshots "
            f"= {len(items)} received snapshots"
        )

        for c0 in tqdm(
            range(0, len(items), CHUNK),
            desc=f"{snr_db:+d} dB"
        ):
            sub = items[c0:c0 + CHUNK]

            xs = []
            pls = []
            truth_batch = []

            for setup, snap in sub:
                idx = np.ravel_multi_index(
                    (setup, k, snap),
                    (ds_te.S, ds_te.N_SNR, ds_te.N_SNAP)
                )

                b = ds_te[int(idx)]

                xs.append(b["x"])
                pls.append(b["pl"])

                # Original truth convention
                truth_rad = np.arcsin(
                    np.sort(b["u"].numpy())
                )

                truth_batch.append(truth_rad)

            x = torch.stack(xs).to(
                device,
                non_blocking=True
            )

            pl = torch.stack(pls).to(
                device,
                non_blocking=True
            )

            # Shape: [B, K*M, 4]
            u = src.posterior_samples(
                model,
                x,
                pl,
                K_DRAWS,
                M_DRAWS,
                INFERENCE_STEPS
            ).clamp(-0.999, 0.999)

            # Original prediction convention
            pred_rad = (
                torch.asin(u)
                .sort(-1).values
                .median(1).values
                .cpu().numpy()
            )

            truth_rad = np.stack(
                truth_batch,
                axis=0
            )

            truth_deg = np.degrees(truth_rad)
            pred_deg = np.degrees(pred_rad)

            truth_chunks.append(truth_deg)
            pred_chunks.append(pred_deg)

            # 4 target observations per received snapshot
            for setup, snap in sub:
                setup_chunks.extend([setup] * 4)
                snapshot_chunks.extend([snap] * 4)

        # [number_of_received_snapshots, 4]
        truth_matrix = np.concatenate(
            truth_chunks,
            axis=0
        )

        pred_matrix = np.concatenate(
            pred_chunks,
            axis=0
        )

        err_matrix = pred_matrix - truth_matrix

        # ====================================================
        # IMPORTANT:
        # POOL THE FOUR TARGETS HERE
        # ====================================================
        truth_pool = truth_matrix.reshape(-1)
        pred_pool = pred_matrix.reshape(-1)
        err_pool = err_matrix.reshape(-1)

        setup_pool = np.asarray(
            setup_chunks,
            dtype=np.int32
        )

        snapshot_pool = np.asarray(
            snapshot_chunks,
            dtype=np.int32
        )

        out[snr_db] = {
            "truth_deg": truth_pool.astype(np.float32),
            "pred_deg": pred_pool.astype(np.float32),
            "err_deg": err_pool.astype(np.float32),
            "setup": setup_pool,
            "snapshot": snapshot_pool,
        }

        rmse_all = np.sqrt(
            np.mean(err_pool ** 2)
        )

        print(
            f"[eval] {snr_db:+d} dB | "
            f"pooled target observations = {len(err_pool)} | "
            f"overall pooled RMSE = {rmse_all:.6f} deg"
        )

    return out


# ============================================================
# RMSE PER ACTUAL-ANGLE BIN
# ============================================================

def make_histogram_data(results):
    edges = np.linspace(
        ANGLE_MIN_DEG,
        ANGLE_MAX_DEG,
        N_BINS + 1
    )

    centers = 0.5 * (
        edges[:-1] + edges[1:]
    )

    rmse_by_snr = {}
    count_by_snr = {}
    unique_setup_by_snr = {}
    rows = []

    for snr_db in SNR_TO_PLOT:
        truth = results[snr_db]["truth_deg"]
        err = results[snr_db]["err_deg"]
        setup = results[snr_db]["setup"]

        # 0 ... N_BINS-1
        bid = np.digitize(
            truth,
            edges[1:-1],
            right=False
        )

        rmse = np.full(
            N_BINS,
            np.nan,
            dtype=np.float64
        )

        counts = np.zeros(
            N_BINS,
            dtype=np.int64
        )

        unique_setups = np.zeros(
            N_BINS,
            dtype=np.int64
        )

        for b in range(N_BINS):
            mask = (bid == b)

            counts[b] = int(mask.sum())

            if counts[b] > 0:
                unique_setups[b] = len(
                    np.unique(setup[mask])
                )

                # RMSE in DEGREES for all pooled matched targets
                # whose TRUE angle falls inside this bin.
                rmse[b] = np.sqrt(
                    np.mean(
                        err[mask] ** 2
                    )
                )

            rows.append({
                "snr_db": snr_db,
                "bin_index": b,
                "bin_left_deg": float(edges[b]),
                "bin_right_deg": float(edges[b + 1]),
                "actual_angle_bin_center_deg": float(centers[b]),
                "rmse_deg": (
                    float(rmse[b])
                    if np.isfinite(rmse[b])
                    else np.nan
                ),
                "target_observation_count": int(counts[b]),
                "unique_test_setup_count": int(unique_setups[b]),
            })

        rmse_by_snr[snr_db] = rmse
        count_by_snr[snr_db] = counts
        unique_setup_by_snr[snr_db] = unique_setups

    return (
        edges,
        centers,
        rmse_by_snr,
        count_by_snr,
        unique_setup_by_snr,
        rows,
    )


# ============================================================
# SAVE CSV / RAW DATA
# ============================================================

def save_csv(rows, path):
    fields = [
        "snr_db",
        "bin_index",
        "bin_left_deg",
        "bin_right_deg",
        "actual_angle_bin_center_deg",
        "rmse_deg",
        "target_observation_count",
        "unique_test_setup_count",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[save] CSV: {path}")


def save_raw(results, path):
    payload = {}

    for snr_db in SNR_TO_PLOT:
        tag = str(snr_db).replace("-", "m")

        for key in [
            "truth_deg",
            "pred_deg",
            "err_deg",
            "setup",
            "snapshot",
        ]:
            payload[
                f"{key}_{tag}dB"
            ] = results[snr_db][key]

    np.savez_compressed(
        path,
        **payload
    )

    print(f"[save] raw NPZ: {path}")


# ============================================================
# ONE FIGURE ONLY
# ============================================================

def plot_ONE_histogram(
    edges,
    centers,
    rmse_by_snr,
    path
):
    import matplotlib.pyplot as plt

    # ========================================================
    # EXACTLY ONE AXIS / ONE HISTOGRAM
    # ========================================================
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(11.5, 5.8)
    )

    bin_width = float(
        np.mean(np.diff(edges))
    )

    n_series = len(SNR_TO_PLOT)

    total_group_width = 0.82 * bin_width
    bar_width = total_group_width / n_series

    offsets = (
        np.arange(n_series)
        - (n_series - 1) / 2
    ) * bar_width

    for j, snr_db in enumerate(SNR_TO_PLOT):
        y = rmse_by_snr[snr_db]
        valid = np.isfinite(y)

        ax.bar(
            centers[valid] + offsets[j],
            y[valid],
            width=0.95 * bar_width,
            label=f"{snr_db} dB",
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
        )

    ax.set_xlabel(
        "Actual target angle (degrees)"
    )

    ax.set_ylabel(
        "RMSE (degrees)"
    )

    ax.set_title(
        "RMSE vs Actual Target Angle"
    )

    ax.set_xlim(
        ANGLE_MIN_DEG,
        ANGLE_MAX_DEG
    )

    ax.set_xticks(
        np.arange(-80, 81, 20)
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    print(f"[plot] ONE histogram saved: {path}")

    try:
        plt.show()
    except Exception:
        pass


# ============================================================
# PRINT BIN CHECK
# ============================================================

def print_bin_summary(
    centers,
    rmse_by_snr,
    unique_setup_by_snr
):
    print("\n[BIN SUMMARY]")
    print(
        "Angle | "
        + " | ".join(
            f"{snr:+d} dB: RMSE / setups"
            for snr in SNR_TO_PLOT
        )
    )

    for b, c in enumerate(centers):
        parts = []

        for snr in SNR_TO_PLOT:
            r = rmse_by_snr[snr][b]
            ns = unique_setup_by_snr[snr][b]

            if np.isfinite(r):
                parts.append(
                    f"{r:8.3f} / {ns:3d}"
                )
            else:
                parts.append(
                    f"{'NaN':>8} / {ns:3d}"
                )

        print(
            f"{c:6.1f} | "
            + " | ".join(parts)
        )


# ============================================================
# MAIN
# ============================================================

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        device = "cuda"
    else:
        device = "cpu"

    print(f"[setup] device = {device}")

    source_path = find_source_script()
    print(f"[source] {source_path}")

    src = import_source(source_path)

    if int(
        getattr(src.CFG, "N_ANGLES", 4)
    ) != 4:
        raise ValueError(
            "Expected CFG.N_ANGLES = 4."
        )

    model, ds_te, ckpt_path = (
        load_model_and_testset(
            src,
            device
        )
    )

    print(f"[checkpoint] {ckpt_path}")

    print(
        f"[test] setups={ds_te.S}, "
        f"SNR points={ds_te.N_SNR}, "
        f"snapshots={ds_te.N_SNAP}"
    )

    print(
        f"[inference] K={K_DRAWS}, "
        f"M={M_DRAWS}, "
        f"K*M={K_DRAWS*M_DRAWS}, "
        f"steps={INFERENCE_STEPS}"
    )

    results = collect_results(
        src,
        model,
        ds_te,
        device
    )

    (
        edges,
        centers,
        rmse_by_snr,
        count_by_snr,
        unique_setup_by_snr,
        rows,
    ) = make_histogram_data(
        results
    )

    print_bin_summary(
        centers,
        rmse_by_snr,
        unique_setup_by_snr
    )

    out_dir = src.CFG.OUT_DIR
    os.makedirs(
        out_dir,
        exist_ok=True
    )

    fig_path = os.path.join(
        out_dir,
        FIG_NAME
    )

    csv_path = os.path.join(
        out_dir,
        CSV_NAME
    )

    raw_path = os.path.join(
        out_dir,
        RAW_NAME
    )

    save_csv(
        rows,
        csv_path
    )

    save_raw(
        results,
        raw_path
    )

    plot_ONE_histogram(
        edges,
        centers,
        rmse_by_snr,
        fig_path
    )

    print("\n[done]")
    print(f"  Figure : {fig_path}")
    print(f"  CSV    : {csv_path}")
    print(f"  Raw NPZ: {raw_path}")


if __name__ == "__main__":
    main()