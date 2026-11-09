"""Generate the verified benchmark and all zero-shot evaluation suites."""
import pickle
import sys
import time
import numpy as np
import sympy as sp
from multiprocessing import Pool

import functions as F


def gen_family(args):
    fam, n, seed, table = args
    gen = getattr(F, table)[fam]
    if isinstance(gen, tuple):
        gen = gen[0]
    rng = np.random.default_rng(seed)
    out, tries = [], 0
    reject = table == "FAMILIES"
    while len(out) < n:
        tries += 1
        e = gen(rng)
        if e is None:
            continue
        a = F.analyze(e, reject=reject)
        if a is None:
            continue
        a["family"] = fam
        a["expr"] = sp.srepr(e)
        out.append(a)
    return fam, out, tries


def main():
    t0 = time.time()
    jobs = [(fam, 4000, 100 + i, "FAMILIES") for i, fam in enumerate(F.FAMILIES)]
    jobs += [(fam, 250, 200 + i, "OOD") for i, fam in enumerate(F.OOD)]
    jobs += [(fam, 100, 300 + i, "DYNSYS") for i, fam in enumerate(F.DYNSYS)]
    jobs += [(fam, F.APPS[fam][1], 400 + i, "APPS") for i, fam in enumerate(F.APPS)]
    with Pool(4) as p:
        res = p.map(gen_family, jobs, chunksize=1)
    data = {"train_families": [], "OOD": {}, "DYNSYS": {}, "APPS": {}}
    tries = {}
    for (fam, out, tr), job in zip(res, jobs):
        tries[fam] = tr
        if job[3] == "FAMILIES":
            data["train_families"] += out
        else:
            data[job[3]][fam] = out
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(data["train_families"]))
    allf = [data["train_families"][i] for i in perm]
    n = len(allf)
    data["train"], data["val"], data["test"] = allf[:int(.7 * n)], allf[int(.7 * n):int(.85 * n)], allf[int(.85 * n):]
    del data["train_families"]
    data["gen_time"] = time.time() - t0
    data["tries"] = tries
    pickle.dump(data, open("data/benchmark.pkl", "wb"))
    print("done", data["gen_time"], {k: len(data[k]) for k in ("train", "val", "test")}, tries)


if __name__ == "__main__":
    main()
