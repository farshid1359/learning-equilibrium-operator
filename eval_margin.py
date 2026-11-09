"""Control experiment: the classical dynamical systems conjugated so that the natural interval
maps to [-8, 8] (a 20% margin) instead of [-10, 10], so no fixed point lies on the edge of the
sampling window. Same random parameters as the main suite (same seeds)."""
import json
import numpy as np
import sympy as sp

import functions as F
from evaluate import load_model, predict, portrait_metrics, summarize

X = F.X


def conj_margin(expr_nat, p, q, frac=0.125):
    p, q = sp.nsimplify(p), sp.nsimplify(q)
    w = q - p
    return F.conj(expr_nat, p - sp.nsimplify(frac) * w, q + sp.nsimplify(frac) * w)


def make(rng, fam):
    if fam == "Logistic map":
        return conj_margin(F.q(rng.uniform(0.5, 4)) * X * (1 - X), 0, 1)
    if fam == "Beverton--Holt":
        R = F.q(rng.uniform(0.5, 5)); return conj_margin(R * X / (1 + (R - 1) * X), 0, 2)
    if fam == "H\'enon slice":
        return conj_margin(1 - F.q(rng.uniform(0.1, 1.4)) * X ** 2, -1.5, 1.5)
    if fam == "Sine map":
        return conj_margin(F.q(rng.uniform(0.3, 1.0)) * sp.sin(sp.pi * X), 0, 1)
    return conj_margin(X * sp.exp(F.q(rng.uniform(0.5, 3)) * (1 - X)), 0, 4)


def main():
    net = load_model("full")
    out, allit = {}, []
    for i, fam in enumerate(F.DYNSYS):
        rng = np.random.default_rng(300 + i)
        items = []
        while len(items) < 100:
            a = F.analyze(make(rng, fam), reject=False)
            if a is not None:
                items.append(a)
        out[fam] = summarize(portrait_metrics(items, predict(net, items)))
        allit += items
        print(fam, out[fam], flush=True)
    out["Overall"] = summarize(portrait_metrics(allit, predict(net, allit)))
    json.dump(out, open("results/margin.json", "w"), indent=1)


if __name__ == "__main__":
    main()
