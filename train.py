"""Train the equilibrium operator and its ablations.

usage: python train.py NAME [--K 3] [--encoder both|conv|lstm] [--nophys] [--point] [--epochs E]"""
import argparse
import json
import math
import pickle
import time
import numpy as np
import torch

from model import EqOperator, losses, point_losses, MAXFP

torch.set_num_threads(1)


def tensors(items):
    B = len(items)
    Y = np.stack([it["samples"] for it in items]).astype(np.float32)
    R = np.zeros((B, MAXFP), np.float32); M = np.zeros_like(R); S = np.zeros_like(R); D = np.zeros_like(R)
    for i, it in enumerate(items):
        n = len(it["roots"])
        R[i, :n] = it["roots"]; M[i, :n] = 1; S[i, :n] = it["stable"]; D[i, :n] = np.arcsinh(it["deriv"])
    return [torch.tensor(a) for a in (Y, R, M, S, D)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--encoder", default="both"); ap.add_argument("--nophys", action="store_true")
    ap.add_argument("--point", action="store_true"); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    data = pickle.load(open("data/benchmark.pkl", "rb"))
    tr, va = tensors(data["train"]), tensors(data["val"])
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    K = 1 if a.point else a.K
    net = EqOperator(K=K, encoder=a.encoder, point=a.point)
    for p in net.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    E = a.epochs
    warm, ramp_end = round(E / 6), round(E / 3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, E)
    n = len(tr[0])
    hist, best, bad, best_state = [], math.inf, 0, None
    t0 = time.time()
    for ep in range(E):
        if a.nophys or ep < warm:
            r = 0.0
        else:
            r = min(1.0, (ep - warm + 1) / max(1, ramp_end - warm))
        lam_fp, lam_st = 0.5 * r, 0.3 * r
        net.train()
        perm = torch.randperm(n)
        tl, nb = 0.0, 0
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            batch = [t[idx] for t in tr]
            out = net(batch[0])
            if a.point:
                loss, _ = point_losses(out, *batch)
            else:
                loss, _ = losses(out, *batch, lam_fp=lam_fp, lam_stab=lam_st)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tl += loss.item(); nb += 1
        sched.step()
        net.eval()
        with torch.no_grad():
            vl, parts = 0.0, {}
            for i in range(0, len(va[0]), 1000):
                batch = [t[i:i + 1000] for t in va]
                out = net(batch[0])
                l, p = point_losses(out, *batch) if a.point else losses(out, *batch, lam_fp=lam_fp, lam_stab=lam_st)
                w = len(batch[0]) / len(va[0])
                vl += l.item() * w
                for k2, v2 in p.items():
                    parts[k2] = parts.get(k2, 0) + v2 * w
            # model selection on the physics-free part of the validation loss
            sel = parts["nll"] + parts["bce"] + parts["hub"]
        hist.append(dict(epoch=ep + 1, train=tl / nb, val=vl, sel=sel, lam=r, time=time.time() - t0, **parts))
        print(f"{a.name} ep{ep+1} train {tl/nb:.4f} val {vl:.4f} sel {sel:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if sel < best - 1e-4:
            best, bad = sel, 0
            best_state = {k2: v2.clone() for k2, v2 in net.state_dict().items()}
        else:
            bad += 1
        # checkpoint every epoch (best weights so far + history)
        torch.save(dict(state=best_state, args=vars(a), K=K, epoch=ep + 1), f"results/{a.name}.pt")
        json.dump(dict(hist=hist, args=vars(a), train_time=time.time() - t0, finished=False),
                  open(f"results/{a.name}_hist.json", "w"))
        if bad >= 12 and ep >= ramp_end:
            break
    json.dump(dict(hist=hist, args=vars(a), train_time=time.time() - t0, finished=True),
              open(f"results/{a.name}_hist.json", "w"))


if __name__ == "__main__":
    main()
