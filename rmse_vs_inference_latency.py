r"""
RMSE vs inference-latency sweep for ONE fixed test sample at the highest SNR.

This helper imports the original training/evaluation script so that the model,
dataset class, checkpoint format, normalization, MeanFlow stage, and DDIM angle
sampler remain exactly the same.

Default sweep:
    DDIM steps = 10, 20, ..., 1000

Outputs:
    rmse_vs_inference_latency_highest_snr.csv
    rmse_vs_inference_latency_highest_snr.png

Example:
    python rmse_vs_inference_latency.py --train-script "C:\Users\ugp.DESKTOP-7Q13T9G\Downloads\meanflow_gdm_attention\meanflow_gdm_attention\meanflow_gdm_attention.py"

Optional:
    python rmse_vs_inference_latency.py --train-script "C:\Users\ugp.DESKTOP-7Q13T9G\Downloads\meanflow_gdm_attention\meanflow_gdm_attention\meanflow_gdm_attention.py" --k 8 --m 16
"""

import argparse
import csv
import importlib.util
import math
import os
import time
from pathlib import Path

import numpy as np
import torch




TEST_SETUP_LOCAL_INDEX = 0
SNAPSHOT_INDEX = 0
INFERENCE_SEED = 2026
TIMING_REPEATS = 1
WARMUP_RUNS = 2
EVAL_CHUNK = 32


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Highest-SNR accuracy-latency sweep: single-sample latency, "
            "full-test-set average 10log10(RMSE_deg) accuracy."
        )
    )
    p.add_argument(
        "--train-script",
        required=True,
        help="Path to the original training/evaluation .py file.",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path. Default: base.CFG.OUT_DIR/base.CFG.CKPT.",
    )
    p.add_argument(
        "--data-path",
        default=None,
        help="Optional dataset .mat path. Default: base.find_data_path().",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: base.CFG.OUT_DIR.",
    )
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="MeanFlow clean draws K. Default: base.CFG.TEST_K.",
    )
    p.add_argument(
        "--m",
        type=int,
        default=None,
        help="Angle posterior draws per clean draw M. Default: base.CFG.TEST_M.",
    )
    p.add_argument(
        "--timing-repeats",
        type=int,
        default=TIMING_REPEATS,
        help="Timed repeats for the one latency sample; median latency is reported.",
    )
    p.add_argument(
        "--step-stride",
        type=int,
        default=10,
        help="Step spacing. Default 10 -> 10,20,...,100.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum number of DDIM inference steps. Default: 100.",
    )
    p.add_argument(
        "--setup-index",
        type=int,
        default=TEST_SETUP_LOCAL_INDEX,
        help="Local test-setup index used only for latency measurement.",
    )
    p.add_argument(
        "--snapshot-index",
        type=int,
        default=SNAPSHOT_INDEX,
        help="Snapshot index used only for latency measurement.",
    )
    p.add_argument(
        "--eval-chunk",
        type=int,
        default=EVAL_CHUNK,
        help="Batch size for full highest-SNR accuracy evaluation.",
    )
    return p.parse_args()


def load_base_module(script_path):
    script_path = os.path.abspath(script_path)

    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Training script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("doa_base", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import training script: {script_path}")

    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    return base


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_inference_rng(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def infer_batch(base, model, x, pl, K, M, steps):
    """
    Full inference followed by MEDIAN posterior aggregation.

    Returns
    -------
    theta_hat_rad : torch.Tensor
        Shape [B, N_ANGLES].
    """
    u = base.posterior_samples(
        model=model,
        x=x,
        pl=pl,
        K=K,
        M=M,
        steps=steps,
    ).clamp(-0.999, 0.999)

    # u: [B, K*M, N_ANGLES]
    # Sort the angles within every posterior realization, convert to theta,
    # then take the MEDIAN across K*M posterior realizations.
    theta_hat_rad = (
        torch.asin(u)
        .sort(dim=-1).values
        .median(dim=1).values
    )

    return theta_hat_rad


@torch.no_grad()
def evaluate_all_highest_snr_samples(
    base,
    model,
    dset,
    device,
    snr_index,
    K,
    M,
    steps,
    chunk,
    seed,
):
    """
    Evaluate ALL test setups x ALL snapshots at one SNR.

    This matches the metric definition of the original evaluate():

        sample_rmse_deg
            = sqrt(mean(angle_error_deg^2 over N_ANGLES))

        sample_metric
            = 10*log10(sample_rmse_deg)

        final_metric
            = mean(sample_metric over all selected test samples)
    """
    model.eval()

    items = [
        (s, n)
        for s in range(dset.S)
        for n in range(dset.N_SNAP)
    ]

    all_rmse_deg = []
    all_10log10_rmse = []

    # Reproducible posterior draws for this complete evaluation.
    reset_inference_rng(seed, device)

    for c0 in range(0, len(items), chunk):
        sub = items[c0:c0 + chunk]

        xs = []
        pls = []
        truths = []

        for s, n in sub:
            flat_idx = np.ravel_multi_index(
                (s, snr_index, n),
                (dset.S, dset.N_SNR, dset.N_SNAP),
            )

            b = dset[int(flat_idx)]

            xs.append(b["x"])
            pls.append(b["pl"])

            # train=False makes u sorted, but sort again explicitly.
            true_rad = np.arcsin(np.sort(b["u"].cpu().numpy()))
            truths.append(true_rad)

        x_batch = torch.stack(xs).to(device)
        pl_batch = torch.stack(pls).to(device)

        pred_rad = infer_batch(
            base=base,
            model=model,
            x=x_batch,
            pl=pl_batch,
            K=K,
            M=M,
            steps=steps,
        ).cpu().numpy()

        true_rad = np.stack(truths, axis=0)

        err_deg = np.degrees(pred_rad - true_rad)

        # One RMSE value per received test sample, over its N_ANGLES DoAs.
        rmse_each_deg = np.sqrt(np.mean(err_deg ** 2, axis=1))

        # EXACT requested reporting transform.
        metric_each = 10.0 * np.log10(
            np.maximum(rmse_each_deg, 1e-9)
        )

        all_rmse_deg.extend(rmse_each_deg.astype(float).tolist())
        all_10log10_rmse.extend(metric_each.astype(float).tolist())

    return {
        "avg_10log10_rmse_deg": float(np.mean(all_10log10_rmse)),
        "mean_rmse_deg": float(np.mean(all_rmse_deg)),
        "median_rmse_deg": float(np.median(all_rmse_deg)),
        "num_samples": int(len(all_rmse_deg)),
    }


def main():
    args = parse_args()

    if args.step_stride <= 0:
        raise ValueError("--step-stride must be positive.")
    if args.max_steps < 10:
        raise ValueError("--max-steps must be >= 10.")
    if args.eval_chunk <= 0:
        raise ValueError("--eval-chunk must be positive.")
    if args.timing_repeats <= 0:
        raise ValueError("--timing-repeats must be positive.")

    print("=" * 78)
    
    print("[metric] y = mean_test_samples[10*log10(single-snapshot RMSE_deg)]")
    print("[latency] one fixed highest-SNR test sample")
    print("[posterior aggregation] MEDIAN across K*M posterior realizations")
    print("=" * 78)

    base = load_base_module(args.train_script)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    if device.type == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(device)}")

    # ------------------------------------------------------------------
    # Checkpoint / EMA model
    # ------------------------------------------------------------------
    ckpt_path = (
        args.checkpoint
        if args.checkpoint is not None
        else os.path.join(base.CFG.OUT_DIR, base.CFG.CKPT)
    )

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ck = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    if "ema" not in ck:
        raise KeyError("Checkpoint does not contain 'ema'.")
    if "stats" not in ck:
        raise KeyError("Checkpoint does not contain 'stats'.")
    if "split" not in ck or "test" not in ck["split"]:
        raise KeyError(
            "Checkpoint must contain split['test'] so the original test "
            "set can be reconstructed exactly."
        )

    stats = ck["stats"]

    model = base.JointModel(
        mf_scale=stats["mf_scale"]
    ).to(device)

    model.load_state_dict(ck["ema"])
    model.eval()

    print(f"[checkpoint] {ckpt_path}")

    K = args.k if args.k is not None else base.CFG.TEST_K
    M = args.m if args.m is not None else base.CFG.TEST_M

    print(
        f"[posterior] K={K}, M={M}, "
        f"K*M={K*M} realizations per received sample"
    )

    # ------------------------------------------------------------------
    # Dataset + original checkpoint test split
    # ------------------------------------------------------------------
    if args.data_path is not None:
        data_path = args.data_path
    else:
        data_path = base.find_data_path()

    y, yc, az = base.load_matfile(data_path)

    idx_te = np.asarray(
        ck["split"]["test"],
        dtype=np.int64,
    )

    ds_te = base.DoaDataset(
        y=y,
        yc=yc,
        az_rad=az,
        setup_idx=idx_te,
        train=False,
        stats=stats,
        label_grid_rad=0.0,
    )

    # Determine the true highest SNR index.
    if hasattr(base, "SNR_DB") and len(base.SNR_DB) == ds_te.N_SNR:
        snr_values = np.asarray(base.SNR_DB, dtype=float)
        k_snr = int(np.argmax(snr_values))
        highest_snr_db = float(snr_values[k_snr])
        snr_text = f"{highest_snr_db:g} dB"
    else:
        k_snr = ds_te.N_SNR - 1
        highest_snr_db = None
        snr_text = f"SNR index {k_snr}"

    print(f"[highest SNR] {snr_text}")

    num_accuracy_samples = ds_te.S * ds_te.N_SNAP

    print(
        f"[accuracy set] {ds_te.S} test setups x "
        f"{ds_te.N_SNAP} snapshots = "
        f"{num_accuracy_samples} test samples at {snr_text}"
    )

    # ------------------------------------------------------------------
    # One fixed sample used ONLY for latency measurement
    # ------------------------------------------------------------------
    if not (0 <= args.setup_index < ds_te.S):
        raise IndexError(
            f"--setup-index must be in [0, {ds_te.S - 1}]"
        )

    if not (0 <= args.snapshot_index < ds_te.N_SNAP):
        raise IndexError(
            f"--snapshot-index must be in [0, {ds_te.N_SNAP - 1}]"
        )

    latency_flat_idx = np.ravel_multi_index(
        (args.setup_index, k_snr, args.snapshot_index),
        (ds_te.S, ds_te.N_SNR, ds_te.N_SNAP),
    )

    latency_item = ds_te[int(latency_flat_idx)]

    x_latency = (
        latency_item["x"]
        .unsqueeze(0)
        .to(device)
    )

    pl_latency = (
        latency_item["pl"]
        .reshape(1)
        .to(device)
    )

    global_setup_index = int(
        idx_te[args.setup_index]
    )

    print(
        f"[latency sample] local test setup={args.setup_index}, "
        f"global setup={global_setup_index}, "
        f"snapshot={args.snapshot_index}, "
        f"SNR={snr_text}"
    )

    # ------------------------------------------------------------------
    # GPU warm-up
    # ------------------------------------------------------------------
    warmup_steps = 10

    print(
        f"[warmup] {WARMUP_RUNS} run(s), "
        f"{warmup_steps} steps; excluded from latency"
    )

    for _ in range(WARMUP_RUNS):
        reset_inference_rng(
            INFERENCE_SEED,
            device,
        )

        _ = infer_batch(
            base=base,
            model=model,
            x=x_latency,
            pl=pl_latency,
            K=K,
            M=M,
            steps=warmup_steps,
        )

        sync_if_cuda(device)

    # ------------------------------------------------------------------
    # Step sweep
    # ------------------------------------------------------------------
    step_values = list(
        range(
            10,
            args.max_steps + 1,
            args.step_stride,
        )
    )

    if step_values[-1] != args.max_steps:
        step_values.append(args.max_steps)

    print(
        f"[sweep] {len(step_values)} values: "
        f"{step_values[0]} -> {step_values[-1]} steps"
    )

    rows = []

    for j, steps in enumerate(
        step_values,
        start=1,
    ):
        # --------------------------------------------------------------
        # A) one-sample latency
        # --------------------------------------------------------------
        latency_runs_ms = []

        for _ in range(args.timing_repeats):
            reset_inference_rng(
                INFERENCE_SEED,
                device,
            )

            sync_if_cuda(device)

            t0 = time.perf_counter()

            _ = infer_batch(
                base=base,
                model=model,
                x=x_latency,
                pl=pl_latency,
                K=K,
                M=M,
                steps=steps,
            )

            sync_if_cuda(device)

            elapsed_ms = (
                time.perf_counter() - t0
            ) * 1000.0

            latency_runs_ms.append(
                elapsed_ms
            )

        latency_ms = float(
            np.median(latency_runs_ms)
        )

        # --------------------------------------------------------------
        # B) full highest-SNR test-set accuracy
        # --------------------------------------------------------------
        accuracy = evaluate_all_highest_snr_samples(
            base=base,
            model=model,
            dset=ds_te,
            device=device,
            snr_index=k_snr,
            K=K,
            M=M,
            steps=steps,
            chunk=args.eval_chunk,
            seed=INFERENCE_SEED,
        )

        rows.append(
            {
                "steps": int(steps),
                "latency_ms": latency_ms,
                "avg_10log10_rmse_deg":
                    accuracy["avg_10log10_rmse_deg"],
                "mean_rmse_deg":
                    accuracy["mean_rmse_deg"],
                "median_rmse_deg":
                    accuracy["median_rmse_deg"],
                "num_accuracy_samples":
                    accuracy["num_samples"],
            }
        )

        print(
            f"[{j:03d}/{len(step_values):03d}] "
            f"steps={steps:4d} | "
            f"latency(one sample)={latency_ms:10.3f} ms | "
            f"avg 10log10(RMSE_deg)="
            f"{accuracy['avg_10log10_rmse_deg']:9.5f} | "
            f"mean raw RMSE="
            f"{accuracy['mean_rmse_deg']:8.5f} deg | "
            f"N={accuracy['num_samples']}"
        )

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    out_dir = Path(
        args.out_dir or base.CFG.OUT_DIR
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        out_dir
        / "rmse_vs_inference_latency_highest_snr_FULLTEST_LOGRMSE.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "steps",
                "latency_ms",
                "avg_10log10_rmse_deg",
                "mean_rmse_deg",
                "median_rmse_deg",
                "num_accuracy_samples",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    latency = np.asarray(
        [r["latency_ms"] for r in rows],
        dtype=float,
    )

    accuracy_metric = np.asarray(
        [
            r["avg_10log10_rmse_deg"]
            for r in rows
        ],
        dtype=float,
    )

    steps_arr = np.asarray(
        [r["steps"] for r in rows],
        dtype=int,
    )

    plt.figure(
        figsize=(7.0, 4.8)
    )

    plt.plot(
        latency,
        accuracy_metric,
        "o-",
        markersize=3.5,
        linewidth=1.2,
    )

    requested_labels = {
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
    }

    for xx, yy, ss in zip(
        latency,
        accuracy_metric,
        steps_arr,
    ):
        if int(ss) in requested_labels:
            plt.annotate(
                f"{ss}",
                (xx, yy),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    plt.xlabel(
        "Inference latency for one test sample (ms)"
    )

    plt.ylabel(
        r"Average $10\log_{10}(\mathrm{RMSE}_{\mathrm{deg}})$"
    )

    plt.title(
        f"Accuracy-latency tradeoff at {snr_text}"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    fig_path = (
        out_dir
        / "rmse_vs_inference_latency_highest_snr_FULLTEST_LOGRMSE.png"
    )

    plt.savefig(
        fig_path,
        dpi=200,
        bbox_inches="tight",
    )

        
    

    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()