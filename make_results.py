"""Regenerate results_macros.tex, tables/*.tex, figures/*.pdf and sec_ablation_text.tex."""
import json
import os
import pickle
import sys
import numpy as np

from vizstyle import plt, BLUE, ORANGE, AQUA, GRAY, LIGHTGRAY, INK, INK2  # noqa: E402

import functions as F  # noqa: E402

OUT = ".."
MAC = {}


def mac(k, v):
    MAC[k] = v


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\%"


def sci(x, d=1):
    if x is None or not np.isfinite(x):
        return "--"
    e = int(np.floor(np.log10(abs(x)))) if x != 0 else 0
    m = x / 10 ** e
    if f"{m:.{d}f}" == "10." + "0" * d:
        m, e = 1.0, e + 1
    return f"${m:.{d}f}\\!\\times\\!10^{{{e}}}$"


def num(x, d=3):
    return f"{x:.{d}f}"


def summ_row(name, s):
    return f"{name} & {s['functions']:,} & {pct(s['coverage'])} & {pct(s['purity'])} & {num(s['median_ae'])} & {pct(s['stab_acc'])}\\\\".replace(",", "{,}")


def main():
    R = json.load(open("results/eval.json"))
    data = pickle.load(open("data/benchmark.pkl", "rb"))
    os.makedirs(f"{OUT}/tables", exist_ok=True); os.makedirs(f"{OUT}/figures", exist_ok=True)
    ds = R["dataset"]
    mac("DSn", f"{ds['n']:,}".replace(",", "{,}")); mac("DSfp", f"{ds['fp']:,}".replace(",", "{,}"))
    mac("DSone", pct(ds["one"])); mac("DStwo", pct(ds["two"])); mac("DSthree", pct(ds["three"]))
    mac("DSmaxfp", str(ds["maxfp"])); mac("DSstable", pct(ds["stable"])); mac("DSmarginal", str(ds["marginal"]))
    mac("DStestfp", f"{ds['test_fp']:,}".replace(",", "{,}")); mac("DStestn", f"{len(data['test']):,}".replace(",", "{,}"))
    mac("DSgenmin", f"{R['gen_time']/60:.0f}")
    full = R["full"]; a = full["all"]
    mac("Cov", pct(a["coverage"])); mac("Pur", pct(a["purity"])); mac("StabAcc", pct(a["stab_acc"]))
    mac("MedAE", num(a["median_ae"])); mac("MeanAE", num(a["mean_ae"])); mac("MedAEgrid", f"{0.2 / a['median_ae']:.0f}")
    mac("MultMAE", num(a["mult_mae"], 2))
    mac("TimeBatch", f"{full['time_batch_ms']:.2f}"); mac("TimeSingle", f"{full['time_single_ms']:.1f}")
    mac("TrainMin", f"{full['train_time']/60:.0f}")
    mac("ECE", num(full["ece"])); mac("Matched", f"{full['matched']:,}".replace(",", "{,}"))
    hist = json.load(open("results/full_hist.json"))
    E = hist["args"]["epochs"]
    mac("TrainEpochs", str(E)); mac("TrainWarm", str(round(E / 6))); mac("TrainRamp", str(round(E / 3)))
    # conformal
    conf = {c["alpha"]: c for c in full["conformal"]}
    c = conf[0.05]
    mac("ConfQhat", num(c["qhat"])); mac("ConfCovFn", pct(c["cov_fn"])); mac("ConfCovPt", pct(c["cov_pt"]))
    mac("ConfWidth", num(c["width_c"], 2)); mac("NaiveCovFn", pct(c["naive_fn"])); mac("NaiveCovPt", pct(c["naive_pt"]))
    mac("NaiveWidth", num(c["width_n"], 2))
    rows = [f"{1-c['alpha']:.2f} & {pct(c['naive_fn'])} & {pct(c['naive_pt'])} & {num(c['width_n'],2)} & {num(c['qhat'],3)} & {pct(c['cov_fn'])} & {pct(c['cov_pt'])} & {num(c['width_c'],2)}\\\\"
            for c in full["conformal"]]
    open(f"{OUT}/tables/conformal.tex", "w").write(
        "\\begin{tabular}{lrrrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{naive $\\mu_k\\pm z\\sigma_k$} & \\multicolumn{4}{c}{conformal $\\mu_k\\pm\\hat q\\sigma_k$}\\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}\nlevel $1-\\alpha$ & simultaneous & per point & width & $\\hat q$ & simultaneous & per point & width\\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    # stratified
    open(f"{OUT}/tables/stratified.tex", "w").write(
        "\\begin{tabular}{lccccc}\n\\toprule\n\\textbf{Subset} & \\textbf{Functions} & \\textbf{Coverage} & \\textbf{Purity} & \\textbf{Median AE} & \\textbf{Stab.\\ acc.}\\\\\n\\midrule\n"
        + summ_row("1 fixed point", full["s1"]) + "\n" + summ_row("2 fixed points", full["s2"]) + "\n" + summ_row("3+ fixed points", full["s3"])
        + "\n\\midrule\n" + summ_row("all", a) + "\n\\bottomrule\n\\end{tabular}")
    # OOD / dynsys / apps
    def suite_table(suite, fname):
        rows = [summ_row(fam, s) for fam, s in R[suite].items() if fam != "Overall"]
        rows.append("\\midrule\n" + summ_row("\\textbf{Overall}", R[suite]["Overall"]))
        open(f"{OUT}/tables/{fname}.tex", "w").write(
            "\\begin{tabular}{lccccc}\n\\toprule\n\\textbf{Family} & \\textbf{Functions} & \\textbf{Coverage} & \\textbf{Purity} & \\textbf{Median AE} & \\textbf{Stab.\\ acc.}\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    suite_table("OOD", "ood")
    mg = json.load(open("results/margin.json"))
    rows = []
    for fam, sv in R["DYNSYS"].items():
        m = mg[fam]
        lab = r"\textbf{Overall}" if fam == "Overall" else fam
        pre = "\midrule\n" if fam == "Overall" else ""
        rows.append(pre + f"{lab} & {pct(sv['coverage'])} & {pct(sv['purity'])} & {num(sv['median_ae'])} & "
                    f"{pct(m['coverage'])} & {pct(m['purity'])} & {num(m['median_ae'])} & {pct(m['stab_acc'])}" + r"\\")
    head = (r"\begin{tabular}{lccccccc}" + "\n" + r"\toprule" + "\n"
            + r" & \multicolumn{3}{c}{natural interval $\to[-10,10]$} & \multicolumn{4}{c}{natural interval $\to[-8,8]$ (margin)}\\" + "\n"
            + r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}" + "\n"
            + r"\textbf{System} & Coverage & Purity & Median AE & Coverage & Purity & Median AE & Stab.\ acc.\\" + "\n" + r"\midrule" + "\n")
    open(f"{OUT}/tables/dynsys.tex", "w").write(head + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}")
    mac("MARcov", pct(mg["Overall"]["coverage"]))
    mac("OODcov", pct(R["OOD"]["Overall"]["coverage"])); mac("DYNcov", pct(R["DYNSYS"]["Overall"]["coverage"]))
    ap = R["APPS"]
    mac("APPcov", pct(ap["Overall"]["coverage"])); mac("APPpur", pct(ap["Overall"]["purity"])); mac("APPstab", pct(ap["Overall"]["stab_acc"]))
    mac("APPecocov", pct(ap["Economics"]["coverage"])); mac("APPctlcov", pct(ap["Control"]["coverage"])); mac("APPchmcov", pct(ap["Chemical kinetics"]["coverage"]))
    # certification
    cert = pickle.load(open("results/cert_full.pkl", "rb"))
    comps = [c for r in cert for c in r["comps"]]
    mac("CertCands", f"{len(comps):,}".replace(",", "{,}"))
    mac("CertDirectPct", pct(np.mean([c["direct"] for c in comps])))
    mac("CertFinalPct", pct(np.mean([c["ok"] for c in comps])))
    uniq = [c for c in comps if c["ok"] and not c.get("dup", False)]
    mac("CertSoundPct", pct(np.mean([c["sound"] for c in uniq])))
    mac("CertNuniq", str(len(uniq)))
    allfound = np.concatenate([r["found"] for r in cert])
    mac("CertRecallPct", pct(np.mean(allfound)))
    direct = [c for c in comps if c["direct"]]
    mac("CertMedRho", sci(np.median([c["rho"] for c in direct])))
    mac("CertBanachPct", pct(np.mean([c["banach"] for c in direct])))
    dec = [c for c in uniq if c["stab"] in (0, 1)]
    mac("CertRepelPct", pct(np.mean([c["stab"] == 0 for c in uniq])))
    mac("CertStabAgreePct", pct(np.mean([c["stab"] == c["true_stab"] for c in dec])))
    mac("CertStabUndetPct", pct(np.mean([c["stab"] == -1 for c in uniq])))
    bs = [c["basin_sound"] for c in uniq if "basin_sound" in c]
    mac("CertBasinSoundPct", pct(np.mean(bs)) if bs else "--"); mac("CertBasinN", str(len(bs)))
    from evaluate import auroc
    mac("CertAUROC", num(auroc([-c["sd"] for c in comps], [c["direct"] for c in comps])))
    ev_fn = [sum(c["n_point"] + c["n_int"] + c["n_newton"] for c in r["comps"]) for r in cert]
    mac("CertEvalsFn", f"{np.mean(ev_fn):.1f}")
    cl = pickle.load(open("results/classical.pkl", "rb"))
    mac("PortraitEvals", f"{np.mean([c['portrait']['evals'] for c in cl]):.1f}")
    mac("PortraitRecall", pct(np.sum([c["portrait"]["nrec"] for c in cl]) / sum(len(it["roots"]) for it in data["test"])))
    # main table
    rows = []
    for key, name in (("newton", "Newton (central differences)"), ("bisection", "Bisection"), ("fpi", "Fixed-point iteration"), ("secant", "Secant")):
        ok = [c[key]["ok"] for c in cl]
        errs = [c[key]["err"] for c in cl if c[key]["ok"]]
        rows.append(f"{name} & {pct(1-np.mean(ok))} & 1 & no & {sci(np.mean(errs))} & {np.mean([c[key]['evals'] for c in cl]):.1f} & {1000*np.mean([c[key]['time'] for c in cl]):.2f}\\\\")
    rows.append(f"Classical portrait (grid + Brent) & -- & all bracketed & yes & recall {pct(np.sum([c['portrait']['nrec'] for c in cl]) / sum(len(it['roots']) for it in data['test']))} & "
                f"{np.mean([c['portrait']['evals'] for c in cl]):.1f} & {1000*np.mean([c['portrait']['time'] for c in cl]):.2f}\\\\")
    rows.append("\\midrule")
    rows.append(f"Operator: forward pass & -- & all ({pct(a['coverage'])}) & {pct(a['stab_acc'])} & median {num(a['median_ae'])}, mean {num(a['mean_ae'])} & 0 & {full['time_batch_ms']:.2f}\\\\")
    rows.append(f"Operator: predict-then-verify & -- & certified {pct(np.mean(allfound))} & certified & median $\\rho$ {sci(np.median([c['rho'] for c in uniq]))} & {np.mean(ev_fn):.1f} & --\\\\")
    open(f"{OUT}/tables/main_results.tex", "w").write(
        "\\begin{tabular}{lcccccc}\n\\toprule\n\\textbf{Method} & \\textbf{Failure} & \\textbf{Roots/run} & \\textbf{Stability} & \\textbf{Error} & \\textbf{Evals} & \\textbf{Time [ms]}\\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    # ablations
    def abl(names):
        out = []
        for n, lab in names:
            if n not in R:
                continue
            s = R[n]["all"]
            out.append(f"{lab} & {pct(s['coverage'])} & {pct(s['purity'])} & {num(s['median_ae'])} & {num(s['mean_ae'])} & {pct(s['stab_acc'])}\\\\")
        return ("\\begin{tabular}{lccccc}\n\\toprule\n\\textbf{Variant} & \\textbf{Coverage} & \\textbf{Purity} & \\textbf{Median AE} & \\textbf{Mean AE} & \\textbf{Stab.\\ acc.}\\\\\n\\midrule\n"
                + "\n".join(out) + "\n\\bottomrule\n\\end{tabular}")
    open(f"{OUT}/tables/arch_ablation.tex", "w").write(abl([("full", "Full (Conv + BiLSTM)"), ("conv", "Conv only"), ("lstm", "BiLSTM only")]))
    open(f"{OUT}/tables/mixture_ablation.tex", "w").write(abl([("full", "$K=3$ (ours)"), ("K1", "$K=1$"), ("point", "Point regression")]))
    for k, n in (("Point", "point"), ("Kone", "K1"), ("Conv", "conv"), ("Lstm", "lstm"), ("Nophys", "nophys")):
        if n in R:
            mac(f"{k}Cov", pct(R[n]["all"]["coverage"])); mac(f"{k}Pur", pct(R[n]["all"]["purity"]))
            mac(f"{k}MedAE", num(R[n]["all"]["median_ae"])); mac(f"{k}Stab", pct(R[n]["all"]["stab_acc"]))
    lrows = []
    cm = {}
    for n, lab in (("full", "Full (with physics)"), ("nophys", "No physics (likelihood only)")):
        cp = pickle.load(open(f"results/cert_{n}.pkl", "rb"))
        cc = [c for r in cp for c in r["comps"]]
        dr = [c for c in cc if c["direct"]]
        cm[n] = (np.mean([c["direct"] for c in cc]), np.median([c["rho"] for c in dr]), auroc([-c["sd"] for c in cc], [c["direct"] for c in cc]),
                 np.mean(np.concatenate([r["found"] for r in cp])))
        s = R[n]["all"]
        lrows.append(f"{lab} & {pct(s['coverage'])} & {num(s['median_ae'])} & {pct(s['stab_acc'])} & {pct(cm[n][0])} & {sci(cm[n][1], 2)} & {num(cm[n][2])} & {pct(cm[n][3])}\\\\")
    open(f"{OUT}/tables/loss_ablation.tex", "w").write(
        "\\begin{tabular}{lccccccc}\n\\toprule\n\\textbf{Variant} & \\textbf{Coverage} & \\textbf{Median AE} & \\textbf{Stab.\\ acc.} & \\textbf{Direct cert.} & \\textbf{Median $\\rho$} & \\textbf{AUROC($\\sigma$)} & \\textbf{Cert.\\ recall}\\\\\n\\midrule\n"
        + "\n".join(lrows) + "\n\\bottomrule\n\\end{tabular}")
    # data-driven ablation text
    f_, np_ = R["full"]["all"], R["nophys"]["all"]
    txt = []
    if "conv" in R and "lstm" in R:
        c_, l_ = R["conv"]["all"], R["lstm"]["all"]
        txt.append(f"\\paragraph{{Encoder.}} Removing the recurrent branch changes coverage from {pct(f_['coverage'])} to {pct(c_['coverage'])} and purity from {pct(f_['purity'])} to {pct(c_['purity'])}; removing the convolutional branch gives {pct(l_['coverage'])} coverage and {pct(l_['purity'])} purity. The median localization error is {num(f_['median_ae'])} for the fused encoder, against {num(c_['median_ae'])} (convolution only) and {num(l_['median_ae'])} (BiLSTM only).")
    if "K1" in R and "point" in R:
        k_, p_ = R["K1"]["all"], R["point"]["all"]
        txt.append(f"\\paragraph{{Output.}} Collapsing the output to a single component reduces coverage from {pct(f_['coverage'])} to {pct(k_['coverage'])}, and point regression to {pct(p_['coverage'])}; their median errors are {num(k_['median_ae'])} and {num(p_['median_ae'])}, against {num(f_['median_ae'])}. A single output forced to explain functions with several fixed points compromises among them; this is the largest effect in the ablation study.")
    d_cov = 100 * (f_["coverage"] - np_["coverage"])
    txt.append(f"\\paragraph{{Physics terms.}} We report this ablation as measured. Without the physics terms the model reaches {pct(np_['coverage'])} coverage (full model: {pct(f_['coverage'])}; difference {d_cov:+.1f} points), median error {num(np_['median_ae'])} ({num(f_['median_ae'])}), and stability accuracy {pct(np_['stab_acc'])} ({pct(f_['stab_acc'])}). "
               f"On the certification metrics, the full model certifies {pct(cm['full'][0])} of its kept components directly (no physics: {pct(cm['nophys'][0])}), with median radius {sci(cm['full'][1],2)} ({sci(cm['nophys'][1],2)}), and the AUROC of $\\sigma_k$ is {num(cm['full'][2])} ({num(cm['nophys'][2])}). "
               "Because the training targets themselves satisfy the fixed point equation to $10^{-10}$, the likelihood term already carries most of the structure that the residual term re-asserts, so large differences are not to be expected; we therefore make no claim beyond the measured numbers, which come from a single seed.")
    open(f"{OUT}/sec_ablation_text.tex", "w").write("\n\n".join(txt) + "\n")
    # bifurcation
    b = R["bifurcation"]
    mac("BifN", f"{b['n']:,}".replace(",", "{,}")); mac("BifTnet", f"{b['t_net']:.2f}"); mac("BifTnetms", f"{1000*b['t_net']/b['n']:.2f}")
    mac("BifTcls", f"{b['t_cls']:.2f}"); mac("BifEvals", f"{b['evals_cls']:.0f}")
    mac("BifCrossUs", f"{1e6 * b['t_net'] / b['n'] / b['evals_cls']:.0f}")
    mac("BifFlip", ", ".join(f"{x:.3f}" for x in b["flip"]) if b["flip"] else "-- (no crossing)")
    tr = b["trans_nontriv"] + b["trans_zero"]
    mac("BifTrans", ", ".join(f"{x:.3f}" for x in sorted(tr)) if tr else "-- (no crossing)")
    figures(R, data, hist, cert, b)
    with open(f"{OUT}/results_macros.tex", "w") as fh:
        fh.write("% generated by code/make_results.py -- do not edit\n")
        for k in sorted(MAC):
            fh.write(f"\\newcommand{{\\{k}}}{{{MAC[k]}}}\n")
    print(len(MAC), "macros")


def figures(R, data, hist, cert, b):
    test = data["test"]
    out = dict(np.load("results/test_pred.npz"))
    from evaluate import components
    # training dynamics
    h = hist["hist"]
    ep = [x["epoch"] for x in h]
    E = hist["args"]["epochs"]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.axvspan(0.5, round(E / 6) + 0.5, color="#eef3fb", zorder=0)
    ax.axvspan(round(E / 6) + 0.5, round(E / 3) + 0.5, color="#fdf0e9", zorder=0)
    ax.plot(ep, [x["train"] for x in h], color=BLUE, label="training loss (full objective)")
    ax.plot(ep, [x["sel"] for x in h], color=ORANGE, label="validation loss (selection)")
    ax.text(1, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 1, "warm-up", fontsize=7, color=INK2)
    ax.text(round(E / 6) + 1, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 1, "physics ramp", fontsize=7, color=INK2)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(loc="upper right")
    fig.savefig(f"{OUT}/figures/training_dynamics.pdf"); plt.close(fig)
    # prediction quality
    tr, pr, st, ae = [], [], [], []
    for i, it in enumerate(test):
        cs = components(out, i)
        mus = np.array([c["mu"] for c in cs])
        for r, s in zip(it["roots"], it["stable"]):
            j = np.argmin(np.abs(mus - r)); tr.append(r); pr.append(mus[j]); st.append(s); ae.append(abs(mus[j] - r))
    tr, pr, st, ae = map(np.array, (tr, pr, st, ae))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    ax = axes[0]
    ax.plot([-10, 10], [-10, 10], color=LIGHTGRAY, lw=1)
    ax.scatter(tr[st == 0], pr[st == 0], s=3, color=ORANGE, alpha=0.5, label="unstable", lw=0)
    ax.scatter(tr[st == 1], pr[st == 1], s=3, color=BLUE, alpha=0.5, label="stable", lw=0)
    ax.set_xlabel("true fixed point"); ax.set_ylabel("nearest predicted component"); ax.legend(loc="upper left", markerscale=3)
    ax.set_title("(a)", loc="left")
    ax = axes[1]
    bins = np.logspace(-5, 1.3, 50)
    ax.hist(np.maximum(ae, 1e-5), bins=bins, color=BLUE)
    ax.axvline(0.2, color=INK2, ls=":", lw=1.2); ax.text(0.22, ax.get_ylim()[1] * 0.85, "$h=0.2$", color=INK2, fontsize=8)
    ax.axvline(np.median(ae), color=ORANGE, lw=1.5); ax.text(np.median(ae) * 0.8, ax.get_ylim()[1] * 0.95, "median", color=INK2, fontsize=8, ha="right")
    ax.set_xscale("log"); ax.set_xlabel("absolute error"); ax.set_ylabel("fixed points"); ax.set_title("(b)", loc="left")
    fig.savefig(f"{OUT}/figures/prediction_quality.pdf"); plt.close(fig)
    # calibration
    full = R["full"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    ax = axes[0]
    rel = full["reliability"]
    ax.plot([0, 1], [0, 1], color=LIGHTGRAY, lw=1)
    ax.scatter([r[0] for r in rel], [r[1] for r in rel], s=[max(8, 300 * r[2] / max(x[2] for x in rel)) for r in rel], color=BLUE, zorder=3)
    ax.plot([r[0] for r in rel], [r[1] for r in rel], color=BLUE, lw=1)
    ax.set_xlabel("predicted stability probability"); ax.set_ylabel("observed frequency of stable")
    ax.set_title(f"(a) ECE = {full['ece']:.3f}", loc="left")
    ax = axes[1]
    lv = [1 - c["alpha"] for c in full["conformal"]]
    ax.plot([0.4, 1], [0.4, 1], color=LIGHTGRAY, lw=1)
    ax.plot(lv, [c["naive_fn"] for c in full["conformal"]], "s-", color=ORANGE, ms=4, label="naive Gaussian")
    ax.plot(lv, [c["cov_fn"] for c in full["conformal"]], "o-", color=BLUE, ms=4, label="conformal")
    ax.set_xlabel("nominal level $1-\\alpha$"); ax.set_ylabel("simultaneous coverage"); ax.legend(loc="lower right")
    ax.set_title("(b)", loc="left")
    fig.savefig(f"{OUT}/figures/calibration.pdf"); plt.close(fig)
    # multimodal example
    ex = None
    for i, it in enumerate(test):
        if len(it["roots"]) == 3 and np.min(np.diff(np.sort(it["roots"]))) > 3.0:
            cs = components(out, i)
            mus = np.array([c["mu"] for c in cs])
            if len(cs) == 3 and np.all(np.min(np.abs(it["roots"][:, None] - mus[None]), 1) < 0.2) and all(c["ok"] for c in cert[i]["comps"]):
                ex = i; break
    ex = ex if ex is not None else (full["example"][0] if full["example"] else 0)
    it = test[ex]
    cs = components(out, ex)
    expr = eval(it["expr"], vars(__import__("sympy")))
    f, _ = F.numpy_funcs(expr)
    xs = np.linspace(-10, 10, 800)
    cres = cert[ex]["comps"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))
    ax = axes[0]
    ax.plot(xs, f(xs), color=BLUE, label="$f(x)$"); ax.plot(xs, xs, color=GRAY, lw=1.2, label="$y=x$")
    for r in it["roots"]:
        ax.plot(r, r, "o", color=INK, ms=4)
    ax.set_xlabel("$x$"); ax.legend(loc="upper left"); ax.set_title("(a)", loc="left")
    ax = axes[1]
    dens = sum(c["pi"] * np.exp(-0.5 * ((xs - c["mu"]) / c["sd"]) ** 2) / (c["sd"] * np.sqrt(2 * np.pi)) for c in cs)
    ax.plot(xs, dens, color=BLUE)
    for r in it["roots"]:
        ax.axvline(r, color=INK2, ls=":", lw=1)
    lab = {1: "attracting", 0: "repelling", -1: "undecided"}
    for c, cr in zip(cs, cres):
        y = c["pi"] / (c["sd"] * np.sqrt(2 * np.pi))
        txt = f"$\\pi$={c['pi']:.2f}, $s$={c['s']:.2f}\n" + (f"cert.: {lab[cr['stab']]}" if cr.get("ok") else "not certified")
        ax.annotate(txt, (c["mu"], min(y, dens.max())), xytext=(0, 6), textcoords="offset points",
                    fontsize=6.5, ha="center", va="bottom", color=INK)
    ax.set_yscale("log"); ax.set_ylim(1e-3, dens.max() * 60)
    ax.set_xlabel("$x$"); ax.set_ylabel("predicted density"); ax.set_title("(b)", loc="left")
    fig.savefig(f"{OUT}/figures/multimodal_example.pdf"); plt.close(fig)
    # certification
    comps = [c for r in cert for c in r["comps"]]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))
    ax = axes[0]
    bins = np.logspace(-17, 0, 40)
    d = [c["rho"] for c in comps if c["direct"]]
    rf = [c["rho"] for c in comps if c["ok"] and not c["direct"]]
    ax.hist(np.maximum(d, 1e-17), bins=bins, color=BLUE, label="direct")
    ax.hist(np.maximum(rf, 1e-17), bins=bins, color=ORANGE, label="after Newton refinement", alpha=0.85)
    ax.set_xscale("log"); ax.set_xlabel("certified radius $\\rho$"); ax.set_ylabel("certificates"); ax.legend(loc="upper left")
    ax.set_title("(a)", loc="left")
    ax = axes[1]
    sd_d = [c["sd"] for c in comps if c["direct"]]; sd_n = [c["sd"] for c in comps if not c["direct"]]
    bins = np.logspace(np.log10(max(1e-3, min(sd_d + sd_n))), np.log10(max(sd_d + sd_n)), 40)
    ax.hist(sd_d, bins=bins, color=BLUE, label="certified directly", density=True)
    ax.hist(sd_n, bins=bins, color=ORANGE, label="routed to refinement", density=True, alpha=0.75)
    ax.set_xscale("log"); ax.set_xlabel("predicted $\\sigma_k$"); ax.set_ylabel("density"); ax.legend(loc="upper right")
    ax.set_title("(b)", loc="left")
    fig.savefig(f"{OUT}/figures/certification.pdf"); plt.close(fig)
    # generalization
    suites = [("Test (in-dist.)", R["full"]["all"])] + [(k, v) for k, v in R["OOD"].items() if k != "Overall"] + \
             [(k.replace("\\'e", "é"), v) for k, v in R["DYNSYS"].items() if k != "Overall"] + [(k, v) for k, v in R["APPS"].items() if k != "Overall"]
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    xs = np.arange(len(suites))
    ax.bar(xs - 0.2, [s["coverage"] for _, s in suites], width=0.38, color=BLUE, label="coverage")
    ax.bar(xs + 0.2, [s["purity"] for _, s in suites], width=0.38, color=AQUA, label="purity")
    ax.set_xticks(xs); ax.set_xticklabels([n.replace("--", "–") for n, _ in suites], rotation=40, ha="right", fontsize=7)
    ax.set_ylim(0, 1.18); ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    for x0 in (0.5, 4.5, 9.5):
        ax.axvline(x0, color=LIGHTGRAY, lw=1)
    fig.savefig(f"{OUT}/figures/generalization.pdf"); plt.close(fig)
    # bifurcation
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    rr, mm, ss = [], [], []
    for row in b["rows"]:
        for mu, s, _ in row["comps"]:
            rr.append(row["r"]); mm.append(mu); ss.append(s)
    sc = ax.scatter(rr, mm, c=ss, cmap=__import__("matplotlib.colors", fromlist=["x"]).LinearSegmentedColormap.from_list("st", [ORANGE, "#d9d8d4", BLUE]),
                    vmin=0, vmax=1, s=6, lw=0, zorder=3)
    r = np.linspace(0.5, 4, 400)
    ax.plot(r, -10 + 20 * (r - 1) / r, color=GRAY, lw=0.8, ls="--", zorder=1, label="true fixed points"); ax.plot(r, -10 * np.ones_like(r), color=GRAY, lw=0.8, ls="--", zorder=1)
    ax.legend(loc="upper left")
    ax.axvline(1, color=INK2, ls=":", lw=1); ax.axvline(3, color=INK2, ls=":", lw=1)
    cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label("predicted stability probability")
    ax.set_xlabel("$r$"); ax.set_ylabel("fixed point (conjugated to $[-10,10]$)"); ax.set_ylim(-10.8, 10.5)
    fig.savefig(f"{OUT}/figures/bifurcation.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
