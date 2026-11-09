"""Evaluate trained models: portraits, calibration, certificates, baselines, sweep.

usage: python evaluate.py           (full evaluation of the main model + ablations)"""
import json
import math
import pickle
import sys
import time
import numpy as np
import sympy as sp
import torch
from multiprocessing import Pool

import functions as F
from model import EqOperator
from train import tensors
from cert1d import Certifier1D

torch.set_num_threads(2)
PI_MIN, MERGE, MATCH = 0.15, 0.1, 0.5


def load_model(name):
    ck = torch.load(f"results/{name}.pt", weights_only=False)
    a = ck["args"]
    net = EqOperator(K=ck["K"], encoder=a["encoder"], point=a["point"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net


@torch.no_grad()
def predict(net, items):
    Y = torch.tensor(np.stack([it["samples"] for it in items]).astype(np.float32))
    outs = []
    for i in range(0, len(Y), 1000):
        o = net(Y[i:i + 1000])
        outs.append({k: v.numpy() for k, v in o.items()})
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def components(out, i):
    """Significant components of function i after merging near-duplicates."""
    mu, sd, pi = out["mu"][i], np.exp(0.5 * out["logvar"][i]), out["pi"][i]
    st, mu_mult = out["stab"][i], out["mult"][i]
    order = np.argsort(-pi)
    comps = []
    for k in order:
        if pi[k] < PI_MIN:
            continue
        for c in comps:
            if abs(c["mu"] - mu[k]) < MERGE:
                c["pi"] += pi[k]
                break
        else:
            comps.append(dict(mu=float(mu[k]), sd=float(sd[k]), pi=float(pi[k]), s=float(st[k]),
                              mult=float(np.sinh(mu_mult[k]))))
    return comps


def portrait_metrics(items, out):
    rec = []
    for i, it in enumerate(items):
        comps = components(out, i)
        mus = np.array([c["mu"] for c in comps])
        roots = it["roots"]
        d = np.abs(roots[:, None] - mus[None])
        near = d.argmin(1)
        ae = d.min(1)
        covered = ae <= MATCH
        pure = (d.min(0) <= MATCH)
        stab_ok = [(comps[near[j]]["s"] > 0.5) == bool(it["stable"][j]) for j in range(len(roots)) if covered[j]]
        mult_err = [abs(np.arcsinh(comps[near[j]]["mult"]) - np.arcsinh(it["deriv"][j])) for j in range(len(roots)) if covered[j]]
        probs = [(comps[near[j]]["s"], int(it["stable"][j])) for j in range(len(roots)) if covered[j]]
        rec.append(dict(n=len(roots), cov=covered.tolist(), pure=pure.tolist(), ae=ae.tolist(),
                        stab=stab_ok, mult=mult_err, probs=probs, ncomp=len(comps)))
    return rec


def summarize(rec, subset=None):
    R = rec if subset is None else [r for r in rec if subset(r)]
    cov = np.concatenate([r["cov"] for r in R]) if R else np.array([])
    pure = np.concatenate([r["pure"] for r in R]) if R else np.array([])
    ae = np.concatenate([r["ae"] for r in R]) if R else np.array([])
    st = np.concatenate([r["stab"] for r in R if r["stab"]]) if R else np.array([])
    mu = np.concatenate([r["mult"] for r in R if r["mult"]]) if R else np.array([])
    return dict(functions=len(R), points=int(len(cov)), coverage=float(cov.mean()), purity=float(pure.mean()),
                median_ae=float(np.median(ae)), mean_ae=float(ae.mean()), stab_acc=float(st.mean()),
                mult_mae=float(np.mean(mu)) if len(mu) else float("nan"))


def ece(probs, nb=10):
    p = np.array([a for a, _ in probs]); y = np.array([b for _, b in probs])
    tot = 0.0
    edges = np.linspace(0, 1, nb + 1)
    for b in range(nb):
        m = (p >= edges[b]) & ((p < edges[b + 1]) if b < nb - 1 else (p <= 1))
        if m.any():
            tot += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return float(tot)


def reliability(probs, nb=10):
    p = np.array([a for a, _ in probs]); y = np.array([b for _, b in probs])
    edges = np.linspace(0, 1, nb + 1)
    out = []
    for b in range(nb):
        m = (p >= edges[b]) & ((p < edges[b + 1]) if b < nb - 1 else (p <= 1))
        if m.sum() > 0:
            out.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return out


# ----------------------------------------------------------------------------
# conformal calibration (simultaneous coverage of all fixed points of a function)
# ----------------------------------------------------------------------------
def conf_scores(items, out):
    per_fn, per_pt = [], []
    for i, it in enumerate(items):
        comps = components(out, i)
        mus = np.array([c["mu"] for c in comps]); sds = np.array([c["sd"] for c in comps])
        s = np.min(np.abs(it["roots"][:, None] - mus[None]) / sds[None], 1)
        per_pt += list(s)
        per_fn.append(float(s.max()))
    return np.array(per_fn), np.array(per_pt)


def union_width(comps, q):
    iv_ = sorted([(c["mu"] - q * c["sd"], c["mu"] + q * c["sd"]) for c in comps])
    tot, cur = 0.0, None
    for a, b in iv_:
        if cur is None or a > cur[1]:
            if cur:
                tot += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    return tot + (cur[1] - cur[0])


def conformal(val, oval, test, otest, alphas=(0.5, 0.2, 0.1, 0.05, 0.01)):
    sv, _ = conf_scores(val, oval)
    st, spt = conf_scores(test, otest)
    n = len(sv)
    res = []
    for a in alphas:
        qh = float(np.quantile(sv, min(1.0, math.ceil((n + 1) * (1 - a)) / n), method="higher"))
        z = float(abs(np.sqrt(2) * __import__("scipy.special", fromlist=["erfinv"]).erfinv(1 - a)))
        widths_c = [union_width(components(otest, i), qh) for i in range(len(test))]
        widths_n = [union_width(components(otest, i), z) for i in range(len(test))]
        res.append(dict(alpha=a, qhat=qh, z=z, cov_fn=float(np.mean(st <= qh)), cov_pt=float(np.mean(spt <= qh)),
                        naive_fn=float(np.mean(st <= z)), naive_pt=float(np.mean(spt <= z)),
                        width_c=float(np.mean(widths_c)), width_n=float(np.mean(widths_n))))
    return res


# ----------------------------------------------------------------------------
# certification (predict-then-verify)
# ----------------------------------------------------------------------------
def newton_refine(f, df, x, maxit=30, tol=1e-12):
    ev = 0
    for _ in range(maxit):
        fx = f(np.array([x]))[0]; d = df(np.array([x]))[0] - 1; ev += 2
        if not np.isfinite(fx) or not np.isfinite(d) or d == 0:
            return x, ev, False
        step = (fx - x) / d
        x = x - step
        if abs(step) < tol or abs(fx - x) < tol:
            return x, ev, abs(x) <= 10
        if abs(x) > 1e3:
            return x, ev, False
    return x, ev, False


def cert_one(args):
    it, comps = args
    expr = eval(it["expr"], vars(sp))
    g_iv, dg_iv = F.interval_funcs(expr)
    f, df = F.numpy_funcs(expr)
    out = []
    certs = []
    for c in comps:
        C = Certifier1D(g_iv, dg_iv)
        r0 = C.certify(c["mu"], slope=c["mult"])
        rec = dict(mu=c["mu"], sd=c["sd"], s_pred=c["s"], direct=r0["ok"], banach=r0.get("banach", False),
                   n_point=C.n_point, n_int=C.n_int, n_newton=0)
        res = r0
        if not r0["ok"]:
            x, ev, conv = newton_refine(f, df, c["mu"])
            rec["n_newton"] = ev
            if conv:
                res = C.certify(x, slope=float(df(np.array([x]))[0]))
                rec["n_point"], rec["n_int"] = C.n_point, C.n_int
        rec["ok"] = res["ok"]
        if res["ok"]:
            rec.update(center=res["mu"], rho=res["rho"], r=res["r"], stab=res["stab"],
                       banach_after=res.get("banach", False))
        out.append(rec)
    # deduplicate certified enclosures and check soundness against the ground truth
    roots = it["roots"]
    uniq = []
    for rec in out:
        if not rec["ok"]:
            continue
        if any(abs(rec["center"] - u["center"]) <= u["r"] for u in uniq):
            rec["dup"] = True
            continue
        rec["dup"] = False
        inside = np.sum(np.abs(roots - rec["center"]) <= rec["rho"] + 1e-12)
        inball = np.sum(np.abs(roots - rec["center"]) <= rec["r"])
        rec["sound"] = bool(inside >= 1 and inball <= 1)
        j = int(np.argmin(np.abs(roots - rec["center"])))
        rec["true_stab"] = int(abs(it["deriv"][j]) < 1)
        # certified basin: Banach ball is forward invariant -> must lie in the true basin
        if rec.get("banach_after") or rec.get("banach"):
            gpts = np.where(np.abs(F.GRID - rec["center"]) <= rec["r"])[0]
            rec["basin_sound"] = bool(np.all(it["basin"][gpts] == j)) if len(gpts) else True
        uniq.append(rec)
    found = [any(abs(r - u["center"]) <= u["rho"] + 1e-12 for u in uniq) for r in roots]
    return dict(comps=out, found=found, n_roots=len(roots))


# ----------------------------------------------------------------------------
# classical baselines
# ----------------------------------------------------------------------------
def classical(it):
    expr = eval(it["expr"], vars(sp))
    fs = F.scalar_func(expr)
    if fs is None:
        f, _ = F.numpy_funcs(expr)
        fs = lambda t: float(f(np.array([t]))[0])
    roots = it["roots"]
    res = {}
    cnt = [0]

    def fe(x):
        cnt[0] += 1
        return fs(x)

    def finish(name, x, ok, t0):
        ok = ok and x is not None and np.isfinite(x) and abs(x) <= 10 and abs(fs(x) - x) < 1e-6
        res[name] = dict(ok=bool(ok), err=float(np.min(np.abs(roots - x))) if ok else None, evals=cnt[0],
                         time=time.perf_counter() - t0)
    # Newton with central differences
    cnt[0] = 0; t0 = time.perf_counter(); x = 0.0; ok = False; h = 1e-6
    try:
        for _ in range(100):
            gx = fe(x) - x; d = (fe(x + h) - fe(x - h)) / (2 * h) - 1
            if d == 0 or not np.isfinite(d):
                break
            xn = x - gx / d
            if abs(xn - x) < 1e-8:
                x = xn; ok = True; break
            x = xn
            if abs(x) > 1e6:
                break
    except (OverflowError, ValueError, ZeroDivisionError):
        ok = False
    finish("newton", x, ok, t0)
    # bisection on the first bracket of the shared grid (samples are free)
    cnt[0] = 0; t0 = time.perf_counter()
    g = it["samples"] - F.GRID
    br = np.where(g[:-1] * g[1:] <= 0)[0]
    if len(br):
        a, b = F.GRID[br[0]], F.GRID[br[0] + 1]; ga = g[br[0]]
        while b - a > 1e-8:
            m = 0.5 * (a + b); gm = fe(m) - m
            if ga * gm <= 0:
                b = m
            else:
                a, ga = m, gm
        finish("bisection", 0.5 * (a + b), True, t0)
    else:
        finish("bisection", None, False, t0)
    # fixed-point iteration
    cnt[0] = 0; t0 = time.perf_counter(); x = 0.0; ok = False
    try:
        for _ in range(100):
            xn = fe(x)
            if abs(xn - x) < 1e-8:
                x = xn; ok = True; break
            x = xn
            if abs(x) > 1e6 or not np.isfinite(x):
                break
    except (OverflowError, ValueError):
        ok = False
    finish("fpi", x, ok, t0)
    # secant
    cnt[0] = 0; t0 = time.perf_counter(); x0, x1 = 0.0, 0.1; ok = False
    try:
        g0, g1 = fe(x0) - x0, fe(x1) - x1
        for _ in range(100):
            if g1 == g0:
                break
            x2 = x1 - g1 * (x1 - x0) / (g1 - g0)
            x0, g0, x1 = x1, g1, x2
            g1 = fe(x1) - x1
            if abs(x1 - x0) < 1e-8:
                ok = True; break
            if abs(x1) > 1e6:
                break
    except (OverflowError, ValueError, ZeroDivisionError):
        ok = False
    finish("secant", x1, ok, t0)
    # classical full portrait: brackets on the shared grid + Brent + stability by central differences
    from scipy.optimize import brentq
    cnt[0] = 0; t0 = time.perf_counter(); found = []
    for i in np.where(g[:-1] * g[1:] < 0)[0]:
        r = brentq(lambda s: fe(s) - s, F.GRID[i], F.GRID[i + 1], xtol=1e-12)
        fe(r + 1e-6); fe(r - 1e-6)
        found.append(r)
    for i in np.where(g == 0)[0]:
        found.append(F.GRID[i])
    rec = [bool(len(found) and np.min(np.abs(np.array(found) - r)) < 1e-6) for r in roots]
    res["portrait"] = dict(recall=float(np.mean(rec)), evals=cnt[0], time=time.perf_counter() - t0, nrec=int(sum(rec)))
    return res


def auroc(score, label):
    score, label = np.asarray(score), np.asarray(label).astype(bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ----------------------------------------------------------------------------
# amortized bifurcation sweep of the logistic map (conjugated to [-10, 10])
# ----------------------------------------------------------------------------
def logistic_conj(r, X):
    """F = h o f_r o h^{-1} with h(x) = -10 + 20 x."""
    x = (X + 10) / 20
    return -10 + 20 * r * x * (1 - x)


def bifurcation(net, nr=2000):
    from scipy.optimize import brentq
    rs = np.linspace(0.5, 4.0, nr)
    Y = np.stack([logistic_conj(r, F.GRID) for r in rs]).astype(np.float32)
    with torch.no_grad():
        net(torch.tensor(Y[:10]))
        t0 = time.perf_counter()
        o = net(torch.tensor(Y))
        t_net = time.perf_counter() - t0
    o = {k: v.numpy() for k, v in o.items()}
    # classical portrait: brackets on the same grid + Brent + derivative-based stability
    t0 = time.perf_counter()
    nev = 0
    for r in rs:
        g = logistic_conj(r, F.GRID) - F.GRID
        for i in np.where(g[:-1] * g[1:] < 0)[0]:
            cnt = [0]
            def gg(x):
                cnt[0] += 1
                return logistic_conj(r, x) - x
            z = brentq(gg, F.GRID[i], F.GRID[i + 1], xtol=1e-12)
            nev += cnt[0] + 2
            (logistic_conj(r, z + 1e-6) - logistic_conj(r, z - 1e-6)) / 2e-6
    t_cls = time.perf_counter() - t0
    rows = []
    for i, r in enumerate(rs):
        comps = components(o, i)
        rows.append(dict(r=float(r), comps=[(c["mu"], c["s"], c["mult"]) for c in comps]))
    # stability transitions of the nontrivial fixed point X* = -10 + 20 (r-1)/r
    def s_near(row, target):
        best = min(row["comps"], key=lambda c: abs(c[0] - target)) if row["comps"] else None
        return best[1] if best and abs(best[0] - target) < 0.5 else np.nan
    sn = np.array([s_near(row, -10 + 20 * (row["r"] - 1) / row["r"]) for row in rows])
    so = np.array([s_near(row, -10.0) for row in rows])
    def crossing(mask_rs, vals, lo, hi, down):
        sel = (rs >= lo) & (rs <= hi) & np.isfinite(vals)
        rr, vv = rs[sel], vals[sel]
        idx = np.where((vv[:-1] > 0.5) & (vv[1:] <= 0.5))[0] if down else np.where((vv[:-1] <= 0.5) & (vv[1:] > 0.5))[0]
        return [float(rr[j]) for j in idx]
    res = dict(n=nr, t_net=t_net, t_cls=t_cls, evals_cls=nev / nr, rows=rows,
               flip=crossing(None, sn, 2.0, 4.0, True), trans_nontriv=crossing(None, sn, 0.5, 2.0, False),
               trans_zero=crossing(None, so, 0.5, 2.0, True))
    return res


def main():
    data = pickle.load(open("data/benchmark.pkl", "rb"))
    names = [n for n in ["full", "nophys", "conv", "lstm", "K1", "point"]]
    R = dict(gen_time=data["gen_time"])
    test, val = data["test"], data["val"]
    allfp = sum(len(it["roots"]) for k in ("train", "val", "test") for it in data[k])
    nfp = np.array([len(it["roots"]) for k in ("train", "val", "test") for it in data[k]])
    stab = np.concatenate([it["stable"] for k in ("train", "val", "test") for it in data[k]])
    der = np.concatenate([it["deriv"] for k in ("train", "val", "test") for it in data[k]])
    R["dataset"] = dict(n=int(len(nfp)), fp=int(allfp), one=float(np.mean(nfp == 1)), two=float(np.mean(nfp == 2)),
                        three=float(np.mean(nfp >= 3)), maxfp=int(nfp.max()), stable=float(stab.mean()),
                        marginal=int(np.sum((np.abs(der) >= 0.95) & (np.abs(der) <= 1.05))),
                        test_fp=int(sum(len(it["roots"]) for it in test)))
    for name in names:
        try:
            net = load_model(name)
        except FileNotFoundError:
            continue
        out = predict(net, test)
        rec = portrait_metrics(test, out)
        R[name] = dict(all=summarize(rec), s1=summarize(rec, lambda r: r["n"] == 1),
                       s2=summarize(rec, lambda r: r["n"] == 2), s3=summarize(rec, lambda r: r["n"] >= 3),
                       train_time=json.load(open(f"results/{name}_hist.json"))["train_time"])
        print(name, R[name]["all"], flush=True)
        if name in ("full", "nophys"):
            jobs = [(it, components(out, i)) for i, it in enumerate(test)]
            t0 = time.time()
            with Pool(4) as pool:
                cres = pool.map(cert_one, jobs, chunksize=10)
            R[name]["cert_time"] = time.time() - t0
            pickle.dump(cres, open(f"results/cert_{name}.pkl", "wb"))
        if name != "full":
            continue
        probs = sum([r["probs"] for r in rec], [])
        R["full"]["ece"] = ece(probs)
        R["full"]["reliability"] = reliability(probs)
        R["full"]["matched"] = len(probs)
        oval = predict(net, val)
        R["full"]["conformal"] = conformal(val, oval, test, out)
        # timing of batched inference
        Y = torch.tensor(np.stack([it["samples"] for it in test]).astype(np.float32))
        with torch.no_grad():
            net(Y[:100]); t0 = time.perf_counter(); net(Y); tb = time.perf_counter() - t0
            t0 = time.perf_counter()
            for i in range(200):
                net(Y[i:i + 1])
            t1 = (time.perf_counter() - t0) / 200
        R["full"]["time_batch_ms"] = 1000 * tb / len(test)
        R["full"]["time_single_ms"] = 1000 * t1
        # zero-shot suites
        for suite in ("OOD", "DYNSYS"):
            R[suite] = {}
            for fam, items in data[suite].items():
                o = predict(net, items)
                R[suite][fam] = summarize(portrait_metrics(items, o))
            allit = sum(data[suite].values(), [])
            R[suite]["Overall"] = summarize(portrait_metrics(allit, predict(net, allit)))
        R["APPS"] = {}
        for fam, items in data["APPS"].items():
            R["APPS"][fam] = summarize(portrait_metrics(items, predict(net, items)))
        allit = sum(data["APPS"].values(), [])
        R["APPS"]["Overall"] = summarize(portrait_metrics(allit, predict(net, allit)))
        # example for the multimodal figure
        ex = [i for i, it in enumerate(test) if len(it["roots"]) == 3 and it["family"] == "trigonometric"
              and all(rec[i]["cov"]) and rec[i]["ncomp"] == 3]
        R["full"]["example"] = ex[:5]
        np.savez("results/test_pred.npz", **out)
        R["bifurcation"] = bifurcation(net)
        # classical
        with Pool(4) as pool:
            cl = pool.map(classical, test, chunksize=20)
        pickle.dump(cl, open("results/classical.pkl", "wb"))
    json.dump(R, open("results/eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
