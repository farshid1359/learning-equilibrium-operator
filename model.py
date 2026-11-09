"""Physics-informed mixture density network for the equilibrium operator."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from functions import GRID, A_DOM, B_DOM, STAB_THR

H_GRID = GRID[1] - GRID[0]
XG = torch.tensor(GRID, dtype=torch.float32)
MAXFP = 10


def features(Y):
    """Scale-robust four-channel encoding. Y: (B, n) samples f(x_i)."""
    G = Y - XG
    s = torch.clamp(G.abs().median(1, keepdim=True).values, min=1e-8)
    return torch.stack([torch.tanh(G / (3 * s)), G / (1 + G.abs()),
                        XG.expand_as(Y) / 10, Y / (1 + Y.abs())], 1), s


class EqOperator(nn.Module):
    def __init__(self, K=3, encoder="both", point=False):
        super().__init__()
        self.K, self.encoder, self.point = K, encoder, point
        if encoder in ("both", "conv"):
            self.conv = nn.Sequential(nn.Conv1d(4, 32, 5, padding=2), nn.ReLU(),
                                      nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                                      nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        if encoder in ("both", "lstm"):
            self.lstm = nn.LSTM(64 if encoder == "both" else 4, 64, batch_first=True, bidirectional=True)
        d = 64 if encoder == "conv" else 128
        self.mlp = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.1))
        self.mu = nn.Linear(128, K)
        self.logvar = nn.Linear(128, K)
        self.mix = nn.Linear(128, K)
        self.stab = nn.Sequential(nn.Linear(130, 64), nn.ReLU(), nn.Linear(64, 1))
        self.mult = nn.Sequential(nn.Linear(130, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, Y):
        z, s = features(Y)
        if self.encoder == "both":
            hseq = self.lstm(self.conv(z).transpose(1, 2))[0]
        elif self.encoder == "conv":
            hseq = self.conv(z).transpose(1, 2)
        else:
            hseq = self.lstm(z.transpose(1, 2))[0]
        h = self.mlp(hseq.mean(1))
        mu = 10 * torch.tanh(self.mu(h))
        logvar = torch.clamp(self.logvar(h), -10, 4)
        if self.point:
            logvar = torch.zeros_like(logvar)
        pi = torch.softmax(self.mix(h), -1)
        hk = torch.cat([h[:, None].expand(-1, self.K, -1), mu[..., None] / 10, logvar[..., None]], -1)
        stab = torch.sigmoid(self.stab(hk)[..., 0])
        mult = self.mult(hk)[..., 0]           # predicts asinh(f'(x*))
        return dict(mu=mu, logvar=logvar, pi=pi, stab=stab, mult=mult, scale=s)


def interp(arr, mu):
    """Differentiable linear interpolation of arr (B, n) on GRID at mu (B, K)."""
    p = torch.clamp((mu - A_DOM) / H_GRID, 0, len(GRID) - 1 - 1e-6)
    i0 = p.floor().long()
    w = p - i0
    a0 = torch.gather(arr, 1, i0)
    a1 = torch.gather(arr, 1, i0 + 1)
    return a0 * (1 - w) + a1 * w


def losses(out, Y, R, M, S, D, lam_fp, lam_stab, lam_ent=1e-3):
    """R: roots (B, MAXFP); M: mask; S: stability labels; D: asinh(f'(x*))."""
    mu, lv, pi = out["mu"], out["logvar"], out["pi"]
    var = lv.exp()
    diff = R[:, :, None] - mu[:, None, :]                        # (B, J, K)
    logN = -0.5 * diff ** 2 / var[:, None] - 0.5 * lv[:, None] - 0.5 * np.log(2 * np.pi)
    lp = torch.logsumexp(torch.log(pi[:, None] + 1e-12) + logN, -1)   # (B, J)
    nll = -(lp * M).sum(1) / M.sum(1)
    k = diff.abs().argmin(-1)                                      # nearest component
    s_k = torch.gather(out["stab"], 1, k)
    m_k = torch.gather(out["mult"], 1, k)
    bce = (Fn.binary_cross_entropy(s_k.clamp(1e-6, 1 - 1e-6), S, reduction="none") * M).sum(1) / M.sum(1)
    hub = (Fn.smooth_l1_loss(m_k, D, reduction="none") * M).sum(1) / M.sum(1)
    # physics terms (scale-normalized residual of the sampled function)
    G = (Y - XG) / (3 * out["scale"])
    dF = torch.zeros_like(Y)
    dF[:, 1:-1] = (Y[:, 2:] - Y[:, :-2]) / (2 * H_GRID)
    dF[:, 0], dF[:, -1] = dF[:, 1], dF[:, -2]
    g_mu = interp(G, mu)
    dg_mu = interp((dF - 1) / (3 * out["scale"]), mu)
    fp = (pi * (torch.tanh(g_mu) ** 2 + torch.tanh(var.sqrt() * dg_mu.abs()) ** 2)).sum(1)
    af = interp(dF, mu).abs().clamp(max=3.0)
    st = out["stab"]
    stab = (pi * (st * Fn.relu(af - STAB_THR) + (1 - st) * Fn.relu(STAB_THR - af))).sum(1)
    ent = (pi * torch.log(pi + 1e-12)).sum(1)                      # = -entropy
    total = nll + bce + hub + lam_fp * fp + lam_stab * stab + lam_ent * ent
    return total.mean(), dict(nll=nll.mean().item(), bce=bce.mean().item(), hub=hub.mean().item(),
                              fp=fp.mean().item(), stab=stab.mean().item())


def point_losses(out, Y, R, M, S, D):
    """Point-regression baseline: squared error to the nearest true fixed point."""
    mu = out["mu"][:, :1]
    d2 = (R - mu) ** 2 + 1e6 * (1 - M)
    k = d2.argmin(1, keepdim=True)
    mse = d2.min(1).values
    s_k = out["stab"][:, 0]
    bce = Fn.binary_cross_entropy(s_k.clamp(1e-6, 1 - 1e-6), torch.gather(S, 1, k)[:, 0], reduction="none")
    hub = Fn.smooth_l1_loss(out["mult"][:, 0], torch.gather(D, 1, k)[:, 0], reduction="none")
    total = mse + bce + hub
    return total.mean(), dict(nll=mse.mean().item(), bce=bce.mean().item(), hub=hub.mean().item(), fp=0, stab=0)
