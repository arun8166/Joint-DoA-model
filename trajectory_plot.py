
import os
import sys
import importlib.util

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt


TRAIN_SCRIPT_PATH = r"C:\Users\sourasis\Downloads\meanflow_gdm_attention\meanflow_gdm_attention.py"

spec = importlib.util.spec_from_file_location("doa_train", TRAIN_SCRIPT_PATH)
doa_train = importlib.util.module_from_spec(spec)
sys.modules["doa_train"] = doa_train
spec.loader.exec_module(doa_train)

CFG            = doa_train.CFG
JointModel     = doa_train.JointModel
DoaDataset     = doa_train.DoaDataset
find_data_path = doa_train.find_data_path
load_matfile   = doa_train.load_matfile
alpha_sigma    = doa_train.alpha_sigma
c2r, r2c       = doa_train.c2r, doa_train.r2c
SNR_DB         = doa_train.SNR_DB

assert not CFG.BASELINE, ("This script visualizes the MeanFlow stage; there's nothing to "
                           "show there when CFG.BASELINE=True.")

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    "mathtext.fontset": "cm", "font.family": "serif", "svg.fonttype": "none",
})
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


SNR_TARGETS_DB = [("Low SNR", -10), ("Mid SNR", 10), ("High SNR", 30)]
SETUP_IDX, SNAPSHOT_IDX = 0, 0     # (s, n) test example to visualize -- change to inspect others
HOPS, MF_DRAWS = 12, 8             # MeanFlow multi-hop trajectory: grid points, ensemble size
K, M, STEPS = CFG.TEST_K, CFG.TEST_M, CFG.TEST_STEPS   # diffusion posterior settings (from your CFG)
D_SHOW = 0                         # which of the 4 (unordered, see note below) angle channels to feature in the fan plot
SEED = 42                           # fixed seed so the figure is reproducible across runs


device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_path = os.path.join(CFG.OUT_DIR, CFG.CKPT)
assert os.path.exists(ckpt_path), (
    f"No checkpoint at {ckpt_path} -- train the model first, or point CFG.OUT_DIR "
    "(inside your training script) at an existing checkpoint.")

ck = torch.load(ckpt_path, map_location=device,weights_only=False)
model = JointModel(mf_scale=ck["stats"]["mf_scale"]).to(device)
model.load_state_dict(ck["ema"])
model.eval()
stats  = ck["stats"]
idx_te = ck["split"]["test"]
print(f"[load] checkpoint epoch={ck['epoch']} step={ck['step']}  |  {len(idx_te)} test setups")

y, yc, az = load_matfile(find_data_path())
ds_te = DoaDataset(y, yc, az, idx_te, train=False, stats=stats)

SNR_TARGETS = [(f"{name} ({db} dB)", SNR_DB.index(db)) for name, db in SNR_TARGETS_DB]


def get_example(ds, s, k, n):
    idx = int(np.ravel_multi_index((s, k, n), (ds.S, 6, 50)))
    b = ds[idx]
    return b["x"].unsqueeze(0), b["xc"].unsqueeze(0), b["pl"].unsqueeze(0), b["u"].numpy()


@torch.no_grad()
def ddim_angles_traj(model, cond, steps):
    B, dev = cond.size(0), cond.device
    z = torch.randn(B, CFG.N_ANGLES, device=dev)
    taus = torch.linspace(1.0, 0.0, steps + 1, device=dev)
    traj = [z.clone()]
    for i in range(steps):
        t = taus[i].expand(B)
        al, si = alpha_sigma(t)
        al = al.unsqueeze(-1)
        si = si.unsqueeze(-1)
        v = model.ang(z, t, cond)
        x0 = al * z - si * v
        ep = si * z + al * v
        aln, sin_ = alpha_sigma(taus[i + 1].expand(B))
        z = (
            aln.unsqueeze(-1) * x0
            + sin_.unsqueeze(-1) * ep
        )
        traj.append(z.clone())
    return torch.stack(traj, dim=0), taus


@torch.no_grad()
def posterior_samples_traj(model, cx, pl, K, M, steps):
    """Same as posterior_samples(), but takes a precomputed cx (so it
    shares the identical noisy-context encoding with the MeanFlow call
    below) and returns the full trajectory instead of only final draws."""
    B = cx.size(0)
    xhat, cxr = model.sample_clean(cx, n_draws=K)
    c_cl = model.ctx_clean(xhat, pl.repeat_interleave(K, dim=0))
    cond = torch.cat([cxr, c_cl], dim=-1).repeat_interleave(M, dim=0)
    traj, taus = ddim_angles_traj(model, cond, steps)
    return traj.view(steps + 1,B,K * M,CFG.N_ANGLES), taus


@torch.no_grad()
def meanflow_hops_traj(model, cx, hops=10, n_draws=1):
    """sample_clean() takes one big jump (t=1 -> r=0). This chains `hops`
    smaller jumps with the same average-velocity network instead, so the
    intermediate points are visualizable. Stays in mf_scale-normalized
    units throughout, matching the space meanflow_loss() trains in."""
    B, dev = cx.size(0), cx.device
    cxr = cx.repeat_interleave(n_draws, dim=0)
    z = torch.randn(B * n_draws, 32, device=dev)
    grid = torch.linspace(1.0, 0.0, hops + 1, device=dev)
    traj = [z.clone()]
    for i in range(hops):
        t = grid[i].expand(B * n_draws)
        r = grid[i + 1].expand(B * n_draws)
        u = model.mf(z, t, r, cxr)
        z = z - u * (t - r).unsqueeze(-1)
        traj.append(z.clone())
    return torch.stack(traj, dim=0), grid                    # (hops+1, B*n_draws, 32)


def meanflow_residual_to_truth(traj_norm, xc_true, mf_scale, n_draws):
    xd_true = (c2r(xc_true) / mf_scale).repeat_interleave(n_draws, dim=0)
    resid = (traj_norm - xd_true.unsqueeze(0)).pow(2).sum(-1).sqrt()
    return resid.cpu().numpy()

@torch.no_grad()
def posterior_from_mf_traj(model, cx, pl, mf_traj, n_mf, M, steps):
    B = cx.size(0)
    # Final MeanFlow points are in normalized real representation
    xhat = r2c(mf_traj[-1] * model.mf_scale)
    cxr = cx.repeat_interleave(n_mf, dim=0)
    plr = pl.repeat_interleave(n_mf, dim=0)
    c_cl = model.ctx_clean(xhat, plr)
    cond = torch.cat([cxr, c_cl], dim=-1)
    cond = cond.repeat_interleave(M, dim=0)
    traj, taus = ddim_angles_traj(model, cond, steps)
    return traj.view(steps + 1,B,n_mf * M,CFG.N_ANGLES), taus

results = []
for label, k in SNR_TARGETS:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
    	torch.cuda.manual_seed_all(SEED)
    x, xc, pl, u_true = get_example(ds_te, SETUP_IDX, k, SNAPSHOT_IDX)
    x, xc, pl = x.to(device), xc.to(device), pl.to(device)

    cx = model.ctx_noisy(x, pl)
    mf_traj, mf_grid = meanflow_hops_traj(model, cx, hops=HOPS, n_draws=MF_DRAWS)
    mf_resid = meanflow_residual_to_truth(mf_traj, xc, model.mf_scale, MF_DRAWS)

    ang_traj, ang_taus = posterior_samples_traj(model,cx,pl,K,M,STEPS)
    ang_traj_np = ang_traj[:, 0].cpu().numpy()                # (STEPS+1, K*M, 4)

    results.append(dict(label=label, mf_grid=mf_grid.cpu().numpy(), mf_resid=mf_resid,
                         ang_taus=ang_taus.cpu().numpy(), ang_traj=ang_traj_np, true_u=u_true))
    print(f"[{label}] true angles (deg): {np.round(np.degrees(np.arcsin(u_true)), 1)}")


fig, axes = plt.subplots(len(results), 2, figsize=(9.6, 3.4 * len(results)))
for row, res in enumerate(results):
    ax = axes[row, 0]
    for s in range(res["mf_resid"].shape[1]):
        ax.plot(res["mf_grid"], res["mf_resid"][:, s], color="#009E73", alpha=0.3, lw=1.2)
    ax.plot(res["mf_grid"], res["mf_resid"].mean(axis=1), color="#009E73", lw=2.2, label="mean")
    ax.invert_xaxis()
    ax.set_ylabel(res["label"] + "\n" + r"$\|\hat z_t - x_{\mathrm{clean}}\|$")
    if row == 0:
        ax.set_title("(a) MeanFlow reconstruction")
        ax.legend(fontsize=8)
    if row == len(results) - 1:
        ax.set_xlabel(r"Flow time $t$  (1 = noise $\rightarrow$ 0 = reconstruction)")

    ax = axes[row, 1]
    single = res["ang_traj"][:, 0, :]                          # one actual sampled path
    for d in range(CFG.N_ANGLES):
        c = COLORS[d % len(COLORS)]
        ax.plot(res["ang_taus"], single[:, d], color=c, lw=1.8, label=f"angle {d+1}")
    for tu in res["true_u"]:
        ax.axhline(tu, color="0.3", ls="--", lw=1.0, alpha=0.6)
    ax.invert_xaxis()
    ax.set_ylabel(r"$u=sin(\theta)$")
    if row == 0:
        ax.set_title("(b) Angle diffusion (one sampled path)")
        ax.legend(fontsize=7, loc="lower left")
    if row == len(results) - 1:
        ax.set_xlabel(r"Diffusion time $\tau$")

fig.suptitle(r"MeanFlow & diffusion trajectories across SNR", y=1.01)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(CFG.OUT_DIR, f"traj_joint_pipeline.{ext}"), bbox_inches="tight")
plt.close(fig)


fig, axes = plt.subplots(1, len(results), figsize=(3.2 * len(results), 3.3), sharey=True)
axes = [axes] if len(results) == 1 else axes
for ax, res in zip(axes, results):
    for s in range(res["ang_traj"].shape[1]):
        ax.plot(res["ang_taus"], res["ang_traj"][:, s, D_SHOW], color="#0072B2", alpha=0.08, lw=1.0)
    for tu in np.sort(res["true_u"]):
        ax.axhline(tu, color="#D55E00", ls="--", lw=1.2, alpha=0.7)
    ax.invert_xaxis()
    ax.set_title(res["label"])
    ax.set_xlabel(r"$\tau$")
axes[0].set_ylabel(fr"$u=sin(\theta$)")
fig.suptitle("Posterior ensemble of angle-diffusion trajectories", y=1.03)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(CFG.OUT_DIR, f"traj_posterior_fan.{ext}"), bbox_inches="tight")
plt.close(fig)

print(f"[done] wrote traj_joint_pipeline.{{png,pdf}} and traj_posterior_fan.{{png,pdf}} to {CFG.OUT_DIR}")