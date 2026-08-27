import copy
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except Exception:

    def tqdm(x, **kw):
        return x


class CFG:
    DATA_PATH = r"C:\Users\ugp.DESKTOP-7Q13T9G\Downloads\meanflow_gdm_attention\meanflow_gdm_attention\5000_data_target_imp_clutter_with_clean_18_07.mat"
    OUT_DIR = r"C:\Users\ugp.DESKTOP-7Q13T9G\Downloads\meanflow_gdm_attention\meanflow_gdm_attention\training_output"
    CKPT = "ckpt_joint_doa.pt"

    EPOCHS = 100
    BATCH = 2048
    LR = 2e-4
    WD = 1e-4
    WARMUP = 250
    WORKERS = 8
    SEED = 0
    LAMBDA_ANG = 1.0
    LABEL_GRID_RAD = 0.0
    COND_DROP = 0.1
    MF_RATIO_RT = 0.75
    MF_ADAPTIVE_P = 0.5
    ORACLE = True
    TF_FINAL = 0.5
    DETACH_XHAT = False
    BASELINE = False
    N_ANGLES = 4
    EVAL_EVERY = 5
    VAL_K, VAL_M, VAL_STEPS = 4, 8, 20
    TEST_K, TEST_M, TEST_STEPS = 8, 16, 30
    TEST_SNAPSHOTS = 5

    TIME_LIMIT_HOURS = 999.0
    QUICK = False


def find_data_path():
    if CFG.DATA_PATH:
        return CFG.DATA_PATH
    cands = sorted(glob.glob("/kaggle/input/**/*.mat", recursive=True),
                   key=os.path.getsize, reverse=True) or \
        sorted(glob.glob("*.mat") + glob.glob("data/*.mat"),
               key=os.path.getsize, reverse=True)
    assert cands, "No .mat file found. Attach your dataset or set CFG.DATA_PATH."
    print(
        f"[data] using {cands[0]} ({os.path.getsize(cands[0]) / 1e6:.0f} MB)")
    return cands[0]


def _to_complex64(a):
    if a.dtype.names and {"real", "imag"} <= set(a.dtype.names):
        a = a["real"].astype(np.float32) + 1j * a["imag"].astype(np.float32)
    return np.ascontiguousarray(a).astype(np.complex64, copy=False)


def load_matfile(path):
    try:
        from scipy.io import loadmat
        d = loadmat(path)
        y, yc, az = d["y_receive"], d["y_receive_clean"], d["target_azimuth"]
    except NotImplementedError:
        import h5py
        with h5py.File(path, "r") as f:
            y = _to_complex64(np.array(f["y_receive"])).transpose()
            yc = _to_complex64(np.array(f["y_receive_clean"])).transpose()
            az = np.array(f["target_azimuth"]).transpose()
    y = _to_complex64(np.asarray(y))
    yc = _to_complex64(np.asarray(yc))
    az = np.squeeze(np.asarray(az, dtype=np.float64))
    if az.ndim == 1:
        az = az[:, None]
    elif az.shape[0] == CFG.N_ANGLES and az.shape[-1] != CFG.N_ANGLES:
        az = az.T
    assert y.shape[
        0] == 16 and y.ndim == 4, f"unexpected y_receive shape {y.shape}"
    assert az.shape[
        1] == CFG.N_ANGLES, f"unexpected target_azimuth shape {az.shape}"

    if np.abs(az).max() > np.pi + 1e-3:
        print(
            "[data] WARNING: |azimuth| > pi -- values look like degrees, converting to radians"
        )
        az = np.radians(az)
    if az.min() >= -1e-6 and az.max() > np.pi / 2 + 1e-3:
        print(
            "[data] azimuths in [0, pi] (endfire) -> shifting to +/- pi/2 broadside"
        )
        az = az - np.pi / 2
    print(f"[data] azimuth range: [{az.min():.4f}, {az.max():.4f}] rad "
          f"= [{np.degrees(az.min()):.1f}, {np.degrees(az.max()):.1f}] deg")
    assert az.min() >= -np.pi / 2 - 1e-3 and az.max() <= np.pi / 2 + 1e-3, \
        "fix the angle convention before training"

    uniq = np.unique(np.round(az.ravel(), 6)).size
    print(
        f"[data] {uniq} unique azimuth values / {az.size} labels "
        f"({'GRIDDED - set CFG.LABEL_GRID_RAD' if uniq < az.size / 4 else 'continuous'})"
    )
    return y, yc, az.astype(np.float32)


class DoaDataset(Dataset):

    def __init__(self,
                 y,
                 yc,
                 az_rad,
                 setup_idx,
                 train,
                 stats,
                 label_grid_rad=0.0):
        self.y = y[..., setup_idx]
        self.yc = yc[..., setup_idx]
        self.az = az_rad[setup_idx]
        self.S = len(setup_idx)
        self.train = train
        self.grid = float(label_grid_rad)
        self.p_mu, self.p_std = stats["p_mu"], stats["p_std"]

    def __len__(self):
        return self.S * 6 * 50

    def __getitem__(self, i):
        s, k, n = np.unravel_index(i, (self.S, 6, 50))
        x = np.array(self.y[:, n, k, s])
        xc = np.array(self.yc[:, n, k, s])
        th = self.az[s].copy()

        if self.train:
            ph = np.exp(1j * np.float32(2 * np.pi * np.random.rand())).astype(
                np.complex64)
            x, xc = x * ph, xc * ph
            if self.grid > 0:
                th = th + np.random.uniform(-self.grid / 2, self.grid / 2,
                                            th.shape).astype(np.float32)

        u = np.sin(th).astype(np.float32)
        u = np.random.permutation(u) if self.train else np.sort(u)
        nx = float(np.linalg.norm(x)) + 1e-12
        x, xc = x / nx, xc / nx
        pl = (math.log(nx * nx) - self.p_mu) / self.p_std
        return dict(x=torch.from_numpy(x),
                    xc=torch.from_numpy(xc),
                    u=torch.from_numpy(u),
                    pl=torch.tensor(pl, dtype=torch.float32))


def compute_stats(y, yc, setup_idx):
    yt, yct = y[..., setup_idx], yc[..., setup_idx]
    norms = np.linalg.norm(yt, axis=0)
    logp = np.log(norms**2 + 1e-12)
    ratio = yct / (norms[None] + 1e-12)
    comp = np.concatenate([ratio.real.ravel()[::7], ratio.imag.ravel()[::7]])
    stats = dict(p_mu=float(logp.mean()),
                 p_std=float(logp.std() + 1e-6),
                 mf_scale=float(comp.std() + 1e-8))
    print(
        f"[stats] log-power mu/std = {stats['p_mu']:.3f}/{stats['p_std']:.3f}, "
        f"mf_scale = {stats['mf_scale']:.4f}")
    return stats


class CLinear(nn.Module):

    def __init__(self, d_in, d_out, bias=True):
        super().__init__()
        s = 1.0 / math.sqrt(2 * d_in)
        self.Wr = nn.Parameter(torch.randn(d_out, d_in) * s)
        self.Wi = nn.Parameter(torch.randn(d_out, d_in) * s)
        self.br = nn.Parameter(torch.zeros(d_out)) if bias else None
        self.bi = nn.Parameter(torch.zeros(d_out)) if bias else None

    def forward(self, z):
        r = F.linear(z.real, self.Wr) - F.linear(z.imag, self.Wi)
        i = F.linear(z.real, self.Wi) + F.linear(z.imag, self.Wr)
        if self.br is not None:
            r, i = r + self.br, i + self.bi
        return torch.complex(r, i)


class ModReLU(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, z):
        m = z.abs()
        return z * (F.relu(m + self.b) / (m + 1e-6))


class CLayerNorm(nn.Module):

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.grr = nn.Parameter(torch.ones(d))
        self.gii = nn.Parameter(torch.ones(d))
        self.gri = nn.Parameter(torch.zeros(d))
        self.gir = nn.Parameter(torch.zeros(d))
        self.br = nn.Parameter(torch.zeros(d))
        self.bi = nn.Parameter(torch.zeros(d))

    def forward(self, z):
        zr, zi = z.real, z.imag
        zr = zr - zr.mean(-1, keepdim=True)
        zi = zi - zi.mean(-1, keepdim=True)
        Vrr = (zr * zr).mean(-1, keepdim=True) + self.eps
        Vii = (zi * zi).mean(-1, keepdim=True) + self.eps
        Vri = (zr * zi).mean(-1, keepdim=True)
        s = torch.sqrt(torch.clamp(Vrr * Vii - Vri * Vri, min=1e-10))
        t = torch.sqrt(Vrr + Vii + 2 * s)
        inv = 1.0 / (s * t + 1e-10)
        Wrr, Wii, Wri = (Vii + s) * inv, (Vrr + s) * inv, -Vri * inv
        xr = Wrr * zr + Wri * zi
        xi = Wri * zr + Wii * zi
        return torch.complex(self.grr * xr + self.gri * xi + self.br,
                             self.gir * xr + self.gii * xi + self.bi)


class CDropout(nn.Module):

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, z):
        if not self.training or self.p <= 0:
            return z
        keep = (torch.rand(z.shape, device=z.device)
                > self.p).float() / (1 - self.p)
        return z * keep


class CAttention(nn.Module):

    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.q = CLinear(d, d, bias=False)
        self.k = CLinear(d, d, bias=False)
        self.v = CLinear(d, d, bias=False)
        self.o = CLinear(d, d)

    def _split(self, z, B, T):
        return z.view(B, T, self.h, self.dh).transpose(1, 2)

    def forward(self, z):
        B, T, _ = z.shape
        q, k, v = (self._split(f(z), B, T) for f in (self.q, self.k, self.v))
        att = torch.einsum("bhid,bhjd->bhij", q, k.conj()).real / math.sqrt(
            self.dh)
        w = att.softmax(-1)
        outr = torch.einsum("bhij,bhjd->bhid", w, v.real)
        outi = torch.einsum("bhij,bhjd->bhid", w, v.imag)
        out = torch.complex(outr,
                            outi).transpose(1,
                                            2).reshape(B, T, self.h * self.dh)
        return self.o(out)


class CBlock(nn.Module):

    def __init__(self, d, heads, drop=0.1):
        super().__init__()
        self.ln1, self.ln2 = CLayerNorm(d), CLayerNorm(d)
        self.attn = CAttention(d, heads)
        self.ffn = nn.ModuleList(
            [CLinear(d, 4 * d),
             ModReLU(4 * d),
             CLinear(4 * d, d)])
        self.drop = CDropout(drop)

    def forward(self, z):
        z = z + self.drop(self.attn(self.ln1(z)))
        h = self.ln2(z)
        for m in self.ffn:
            h = m(h)
        return z + self.drop(h)


class ComplexEncoder(nn.Module):

    def __init__(self, n_ant=16, d=128, layers=5, heads=4, ctx=256, drop=0.1):
        super().__init__()
        self.d = d
        self.inp = CLinear(n_ant + 1, d)
        self.pos = nn.Parameter(torch.randn(n_ant, d, 2) * 0.02)
        self.blocks = nn.ModuleList(
            [CBlock(d, heads, drop) for _ in range(layers)])
        self.lnf = CLayerNorm(d)
        self.pool_q = nn.Parameter(torch.randn(d, 2) * 0.02)
        self.mlp = nn.Sequential(nn.Linear(2 * d + 3, ctx), nn.GELU(),
                                 nn.Linear(ctx, ctx))

    def forward(self, z, extra, role):
        n2 = (z.real**2 + z.imag**2).sum(-1, keepdim=True)
        zt = z / torch.sqrt(n2 + 1e-12)
        X = zt.unsqueeze(-1) * zt.conj().unsqueeze(-2)
        tok = torch.cat([zt.unsqueeze(-1), X], dim=-1)
        h = self.inp(tok) + torch.view_as_complex(self.pos)[None]
        for b in self.blocks:
            h = b(h)
        h = self.lnf(h)
        q = torch.view_as_complex(self.pool_q)
        w = (torch.einsum("btd,d->bt", h, q.conj()).real /
             math.sqrt(self.d)).softmax(-1)
        gr = torch.einsum("bt,btd->bd", w, h.real)
        gi = torch.einsum("bt,btd->bd", w, h.imag)
        feat = torch.cat([
            gr, gi,
            torch.log(n2 + 1e-8),
            extra.unsqueeze(-1),
            role.unsqueeze(-1)
        ],
                         dim=-1)
        return self.mlp(feat)


class TimeEmb(nn.Module):

    def __init__(self, dim=64):
        super().__init__()
        self.register_buffer(
            "freqs", torch.exp(torch.linspace(0.0, math.log(200.0), dim // 2)))

    def forward(self, t):
        a = t.unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class AdaBlock(nn.Module):

    def __init__(self, h):
        super().__init__()
        self.ln = nn.LayerNorm(h, elementwise_affine=False)
        self.mod = nn.Linear(h, 3 * h)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)
        self.net = nn.Sequential(nn.Linear(h, 2 * h), nn.SiLU(),
                                 nn.Linear(2 * h, h))

    def forward(self, x, c):
        s, b, g = self.mod(c).chunk(3, dim=-1)
        return x + g * self.net(self.ln(x) * (1 + s) + b)


class MeanFlowNet(nn.Module):

    def __init__(self, xdim=32, h=256, blocks=6, cdim=256):
        super().__init__()
        self.temb, self.remb = TimeEmb(64), TimeEmb(64)
        self.cmap = nn.Sequential(nn.Linear(cdim + 128, h), nn.SiLU(),
                                  nn.Linear(h, h))
        self.inp = nn.Linear(xdim, h)
        self.blocks = nn.ModuleList([AdaBlock(h) for _ in range(blocks)])
        self.out_ln = nn.LayerNorm(h, elementwise_affine=False)
        self.out = nn.Linear(h, xdim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z, t, r, c):
        cond = self.cmap(torch.cat([c, self.temb(t), self.remb(r)], dim=-1))
        x = self.inp(z)
        for b in self.blocks:
            x = b(x, cond)
        return self.out(self.out_ln(x))


class DiTBlock(nn.Module):

    def __init__(self, h, heads, cdim):
        super().__init__()
        self.ln1 = nn.LayerNorm(h, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(h, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(h, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(h, 4 * h), nn.SiLU(),
                                 nn.Linear(4 * h, h))
        self.mod = nn.Linear(cdim, 6 * h)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, c):
        s1, b1, g1, s2, b2, g2 = self.mod(c).unsqueeze(1).chunk(6, dim=-1)
        y = self.ln1(x) * (1 + s1) + b1
        a, _ = self.attn(y, y, y, need_weights=False)
        x = x + g1 * a
        y = self.ln2(x) * (1 + s2) + b2
        return x + g2 * self.mlp(y)


class AngleDenoiser(nn.Module):

    def __init__(self, h=128, blocks=4, heads=4, cond_in=512, cdim=256):
        super().__init__()
        self.inp = nn.Linear(1, h)
        self.temb = TimeEmb(64)
        self.cmap = nn.Sequential(nn.Linear(cond_in + 64, cdim), nn.SiLU(),
                                  nn.Linear(cdim, cdim))
        self.blocks = nn.ModuleList(
            [DiTBlock(h, heads, cdim) for _ in range(blocks)])
        self.out_ln = nn.LayerNorm(h, elementwise_affine=False)
        self.out = nn.Linear(h, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, ut, tau, cond):
        x = self.inp(ut.unsqueeze(-1))
        c = self.cmap(torch.cat([cond, self.temb(tau)], dim=-1))
        for b in self.blocks:
            x = b(x, c)
        return self.out(self.out_ln(x)).squeeze(-1)


def c2r(z):
    return torch.view_as_real(z).flatten(-2)


def r2c(v):
    return torch.view_as_complex(v.view(*v.shape[:-1], 16, 2).contiguous())


def alpha_sigma(tau):
    return torch.cos(0.5 * math.pi * tau), torch.sin(0.5 * math.pi * tau)


def cwhere(mask, a, b):
    return torch.complex(torch.where(mask, a.real, b.real),
                         torch.where(mask, a.imag, b.imag))


class JointModel(nn.Module):

    def __init__(self, ctx=256, mf_scale=1.0):
        super().__init__()
        self.encoder = ComplexEncoder(ctx=ctx)
        self.mf = MeanFlowNet(cdim=ctx)
        self.ang = AngleDenoiser(cond_in=2 * ctx)
        self.c_null = nn.Parameter(torch.zeros(ctx))
        self.register_buffer("mf_scale", torch.tensor(float(mf_scale)))

    def ctx_noisy(self, x, pl):
        return self.encoder(x, pl, torch.zeros_like(pl))

    def ctx_clean(self, z, pl):
        return self.encoder(z, pl, torch.ones_like(pl))

    def sample_clean(self, cx, n_draws=1):
        B, dev = cx.size(0), cx.device
        cxr = cx.repeat_interleave(n_draws, dim=0)
        z1 = torch.randn(B * n_draws, 32, device=dev)
        t1 = torch.ones(B * n_draws, device=dev)
        u = self.mf(z1, t1, torch.zeros_like(t1), cxr)
        return r2c((z1 - u) * self.mf_scale), cxr


def meanflow_loss(model, cx, xc):
    B, dev = xc.size(0), xc.device
    xd = c2r(xc) / model.mf_scale
    a, b = torch.rand(B, device=dev), torch.rand(B, device=dev)
    t, r = torch.maximum(a, b), torch.minimum(a, b)
    r = torch.where(torch.rand(B, device=dev) < CFG.MF_RATIO_RT, t, r)
    eps = torch.randn_like(xd)
    tt = t.unsqueeze(-1)
    zt = (1 - tt) * xd + tt * eps
    v = eps - xd
    u, dudt = torch.func.jvp(
        lambda z_, t_, r_, c_: model.mf(z_, t_, r_, c_),
        (zt, t, r, cx),
        (v, torch.ones_like(t), torch.zeros_like(r), torch.zeros_like(cx)),
    )
    u_tgt = (v - (t - r).unsqueeze(-1) * dudt).detach()
    err = ((u - u_tgt)**2).sum(-1)
    return (err / (err.detach() + 1e-3).pow(CFG.MF_ADAPTIVE_P)).mean()


def angle_diffusion_loss(model, cond, u0):
    B, dev = u0.size(0), u0.device
    tau = torch.rand(B, device=dev)
    al, si = alpha_sigma(tau)
    al, si = al.unsqueeze(-1), si.unsqueeze(-1)
    eps = torch.randn_like(u0)
    return F.mse_loss(model.ang(al * u0 + si * eps, tau, cond),
                      al * eps - si * u0)


def training_step(model, batch, p_tf):
    x, xc, u0, pl = batch["x"], batch["xc"], batch["u"], batch["pl"]
    B, dev = x.size(0), x.device
    cx = model.ctx_noisy(x, pl)

    if CFG.BASELINE:
        mf_l = torch.zeros((), device=dev)
        c_cl = model.c_null.unsqueeze(0).expand(B, -1)
    else:
        mf_l = meanflow_loss(model, cx, xc)
        xhat, _ = model.sample_clean(cx, n_draws=1)
        if CFG.DETACH_XHAT:
            xhat = xhat.detach()
        tf = (torch.rand(B, device=dev) < p_tf).unsqueeze(-1)
        c_cl = model.ctx_clean(cwhere(tf, xc, xhat), pl)
        drop = (torch.rand(B, device=dev) < CFG.COND_DROP).unsqueeze(-1)
        c_cl = torch.where(drop, model.c_null.unsqueeze(0).expand(B, -1), c_cl)

    ang_l = angle_diffusion_loss(model, torch.cat([cx, c_cl], dim=-1), u0)
    return mf_l + CFG.LAMBDA_ANG * ang_l, mf_l.detach(), ang_l.detach()


@torch.no_grad()
def ddim_angles(model, cond, steps):
    B, dev = cond.size(0), cond.device
    z = torch.randn(B, CFG.N_ANGLES, device=dev)
    taus = torch.linspace(1.0, 0.0, steps + 1, device=dev)
    for i in range(steps):
        t = taus[i].expand(B)
        al, si = alpha_sigma(t)
        al, si = al.unsqueeze(-1), si.unsqueeze(-1)
        v = model.ang(z, t, cond)
        x0, ep = al * z - si * v, si * z + al * v
        aln, sin_ = alpha_sigma(taus[i + 1].expand(B))
        z = aln.unsqueeze(-1) * x0 + sin_.unsqueeze(-1) * ep
    return z


@torch.no_grad()
def posterior_samples(model, x, pl, K, M, steps, xc=None, oracle=False):

    B = x.size(0)

    # Noisy context is retained in both normal and oracle models
    cx = model.ctx_noisy(x, pl)

    # ============================================================
    # ORACLE
    # Use ACTUAL clean snapshot instead of MeanFlow-generated xhat
    # ============================================================
    if oracle:

        if xc is None:
            raise ValueError("Oracle inference requires xc.")

        # Encode true clean snapshot
        c_cl = model.ctx_clean(xc, pl)

        # Same conditioning structure as joint model:
        # noisy context + clean context
        cond = torch.cat([cx, c_cl], dim=-1)

        # Keep the same total number K*M of DoA posterior samples
        cond = cond.repeat_interleave(K * M, dim=0)

    # ============================================================
    # DIFFUSION-ONLY BASELINE
    # ============================================================
    elif CFG.BASELINE:

        cond = torch.cat([cx, model.c_null.unsqueeze(0).expand(B, -1)], dim=-1)

        cond = cond.repeat_interleave(K * M, dim=0)

    # ============================================================
    # NORMAL JOINT MODEL
    # ============================================================
    else:

        # K MeanFlow-generated clean snapshots
        xhat, cxr = model.sample_clean(cx, n_draws=K)

        # Encode generated clean snapshots
        c_cl = model.ctx_clean(xhat, pl.repeat_interleave(K, dim=0))

        cond = torch.cat([cxr, c_cl], dim=-1)

        # M DoA diffusion samples per generated clean snapshot
        cond = cond.repeat_interleave(M, dim=0)

    return ddim_angles(model, cond, steps).view(B, K * M, CFG.N_ANGLES)


@torch.no_grad()
def predict_single(model,
                   stats,
                   x_np,
                   K=None,
                   M=None,
                   steps=None,
                   device=None):
    model.eval()
    device = device or next(model.parameters()).device
    K, M, steps = K or CFG.TEST_K, M or CFG.TEST_M, steps or CFG.TEST_STEPS
    v = np.asarray(x_np).reshape(16).astype(np.complex64)
    n = np.linalg.norm(v) + 1e-12
    x = torch.from_numpy(v / n).unsqueeze(0).to(device)
    pl = torch.tensor([(math.log(n * n) - stats["p_mu"]) / stats["p_std"]],
                      dtype=torch.float32,
                      device=device)
    u = posterior_samples(model, x, pl, K, M,
                          steps).reshape(-1,
                                         CFG.N_ANGLES).clamp(-0.999, 0.999)
    th = torch.asin(u).sort(-1).values.cpu().numpy()
    return np.median(th, axis=0), th


@torch.no_grad()
def evaluate(model,
             dset,
             device,
             snapshots=(0, ),
             K=4,
             M=8,
             steps=20,
             chunk=192,
             oracle=False):
    model.eval()
    db_scores = {k: [] for k in range(6)}
    items = [(s, k, n) for s in range(dset.S) for k in range(6)
             for n in snapshots]

    for c0 in tqdm(range(0, len(items), chunk), desc="eval", leave=False):
        sub = items[c0:c0 + chunk]
        xs, xcs, pls, trus, ks = [], [], [], [], []

        for (s, k, n) in sub:
            b = dset[int(np.ravel_multi_index((s, k, n), (dset.S, 6, 50)))]
            xs.append(b["x"])
            pls.append(b["pl"])
            xcs.append(b["xc"])
            ks.append(k)
            trus.append(np.arcsin(np.sort(b["u"].numpy())))

        x = torch.stack(xs).to(device)
        pl = torch.stack(pls).to(device)
        xc = torch.stack(xcs).to(device)
        u = posterior_samples(model, x, pl, K, M, steps, xc=xc,
                              oracle=oracle).clamp(-0.999, 0.999)
        th_pred_rad = torch.asin(u).sort(-1).values.median(
            1).values.cpu().numpy()

        for j, k in enumerate(ks):
            err_deg = np.degrees(th_pred_rad[j] - trus[j])
            rmse_one_snapshot_deg = np.sqrt(np.mean(err_deg**2))
            score_db = 10.0 * np.log10(max(rmse_one_snapshot_deg, 1e-9))
            db_scores[k].append(float(score_db))

    return {k: float(np.mean(v)) for k, v in db_scores.items()}


SNR_DB = [-20, -10, 0, 10, 20, 30]


def rmse_to_db(r):
    return 20.0 * math.log10(max(float(r), 1e-9))


def plot_rmse_db(metric_db, out_path, label):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.4, 4.2))
    plt.plot(SNR_DB, [metric_db[k] for k in range(6)], "o-", label=label)

    other = os.path.join(
        CFG.OUT_DIR,
        "test_metrics.json" if CFG.BASELINE else "test_metrics_baseline.json")

    if os.path.exists(other):
        try:
            with open(other) as f:
                o = json.load(f)["avg_10log10_rmse_deg"]

            olab = ("joint (MeanFlow + diffusion)"
                    if CFG.BASELINE else "baseline (no MeanFlow)")
            plt.plot(SNR_DB, [o[f"{s}dB"] for s in SNR_DB], "s--", label=olab)
        except Exception as e:
            print(f"[plot] could not overlay {other}: {e}")

    plt.xlabel("SNR (dB)")
    plt.ylabel("Average 10*log10(single-snapshot RMSE in degrees)")
    plt.title("DoA test metric vs SNR (single-snapshot input)")
    plt.xticks(SNR_DB)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[plot] saved {out_path}")

    try:
        plt.show()
    except Exception:
        pass


class EMA:

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point or v.dtype.is_complex:
                self.shadow[k].mul_(self.decay).add_(v.detach(),
                                                     alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def build_eval_model(self, model):
        m = copy.deepcopy(model)
        m.load_state_dict(self.shadow)
        m.eval()
        return m


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = JointModel(mf_scale=ck["stats"]["mf_scale"]).to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    return model, ck["stats"]


def main():
    torch.manual_seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected")
    if CFG.QUICK:
        CFG.EPOCHS, CFG.EVAL_EVERY, CFG.TEST_SNAPSHOTS = 1, 1, 1
    print(f"[setup] device = {device}")

    y, yc, az = load_matfile(find_data_path())
    N = y.shape[-1]
    perm = np.random.RandomState(123).permutation(N)
    n_tr, n_va = int(0.8 * N), int(0.1 * N)
    idx_tr, idx_va, idx_te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr +
                                                                       n_va:]
    print(
        f"[split] setups: train {len(idx_tr)} / val {len(idx_va)} / test {len(idx_te)} "
        f"(SPLIT IS BY SETUP - never split by row)")

    stats = compute_stats(y, yc, idx_tr)
    ds_tr = DoaDataset(y, yc, az, idx_tr, True, stats, CFG.LABEL_GRID_RAD)
    ds_va = DoaDataset(y, yc, az, idx_va, False, stats)
    ds_te = DoaDataset(y, yc, az, idx_te, False, stats)
    dl = DataLoader(ds_tr,
                    batch_size=CFG.BATCH,
                    shuffle=True,
                    num_workers=CFG.WORKERS,
                    drop_last=True,
                    pin_memory=(device == "cuda"),
                    persistent_workers=(CFG.WORKERS > 0))

    model = JointModel(mf_scale=stats["mf_scale"]).to(device)
    print(
        f"[model] {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters"
    )
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WD)
    total_steps = max(1, CFG.EPOCHS * len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, CFG.WARMUP)) * 0.5 *
        (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    ema = EMA(model)

    ckpt_path = os.path.join(CFG.OUT_DIR, CFG.CKPT)
    step, ep0 = 0, 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, ep0 = ck["step"], ck["epoch"]
        print(f"[resume] from {ckpt_path} at epoch {ep0}, step {step}")

    def save(ep):
        torch.save(
            dict(ema=ema.shadow,
                 model=model.state_dict(),
                 opt=opt.state_dict(),
                 sched=sched.state_dict(),
                 step=step,
                 epoch=ep,
                 stats=stats,
                 split=dict(train=idx_tr, val=idx_va, test=idx_te)), ckpt_path)

    t_start = time.time()
    anneal = max(1, int(0.3 * total_steps))
    out_of_time = False

    for ep in range(ep0, CFG.EPOCHS):
        model.train()
        mf_m = an_m = 0.0
        nb = 0
        for batch in tqdm(dl, desc=f"epoch {ep + 1}/{CFG.EPOCHS}",
                          leave=False):
            batch = {
                k: v.to(device, non_blocking=True)
                for k, v in batch.items()
            }
            p_tf = 1.0 - (1.0 - CFG.TF_FINAL) * min(1.0, step / anneal)
            loss, mf_l, an_l = training_step(model, batch, p_tf)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.update(model)
            step += 1
            mf_m += mf_l.item()
            an_m += an_l.item()
            nb += 1
            if CFG.QUICK and nb >= 30:
                break
            if (time.time() - t_start) > CFG.TIME_LIMIT_HOURS * 3600 - 900:
                out_of_time = True
                break
        print(
            f"[ep {ep + 1:03d}] mf={mf_m / max(nb, 1):.4f} ang={an_m / max(nb, 1):.4f} "
            f"p_tf={p_tf:.2f} lr={sched.get_last_lr()[0]:.2e}")

        if (ep + 1
            ) % CFG.EVAL_EVERY == 0 or ep + 1 == CFG.EPOCHS or out_of_time:
            em = ema.build_eval_model(model)
            rmse = evaluate(em,
                            ds_va,
                            device,
                            K=CFG.VAL_K,
                            M=CFG.VAL_M,
                            steps=CFG.VAL_STEPS)
            print("        val avg 10log10(RMSE_deg): " +
                  "  ".join(f"{SNR_DB[k]}dB:{rmse[k]:.4f}" for k in range(6)))
            save(ep + 1)
            del em
        if out_of_time:
            print(
                "[time] approaching session limit -> checkpoint saved, stopping. "
                "Re-run the notebook to resume.")
            break

    em = ema.build_eval_model(model)
    snaps = tuple(range(CFG.TEST_SNAPSHOTS))
    metric_te = evaluate(em,
                         ds_te,
                         device,
                         snapshots=snaps,
                         K=CFG.TEST_K,
                         M=CFG.TEST_M,
                         steps=CFG.TEST_STEPS,
                         oracle=True)

    print("[TEST] average 10*log10(single-snapshot RMSE in degrees): " +
          "  ".join(f"{SNR_DB[k]}dB:{metric_te[k]:.4f}" for k in range(6)))

    tag = "test_metrics_baseline.json" if CFG.BASELINE else "test_metrics.json"
    with open(os.path.join(CFG.OUT_DIR, tag), "w") as f:
        json.dump(
            dict(
                unit="dB",
                definition=
                f"mean_over_single_snapshots[10*log10(RMSE_deg_over_{CFG.N_ANGLES}_DoAs)]",
                avg_10log10_rmse_deg={
                    f"{SNR_DB[k]}dB": metric_te[k]
                    for k in range(6)
                },
            ),
            f,
            indent=2,
        )

    lab = "baseline (no MeanFlow)" if CFG.BASELINE else "joint (MeanFlow + diffusion)"
    plot_rmse_db(metric_te, os.path.join(CFG.OUT_DIR, "rmse_vs_snr.png"), lab)

    s, k, n = 0, 3, 0
    b = ds_te[int(np.ravel_multi_index((s, k, n), (ds_te.S, 6, 50)))]
    x_raw = (b["x"].numpy() * 1.0)
    est, _ = predict_single(em, stats, x_raw, device=device)
    tru = np.arcsin(np.sort(b["u"].numpy()))
    print(
        f"[demo] truth (rad): {np.round(tru, 3)}  |  estimate (rad): {np.round(est, 3)}"
    )
    print(f"[done] checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
