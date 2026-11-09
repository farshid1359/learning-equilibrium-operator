"""Re-measure wall-clock timings on an otherwise idle machine and update results/eval.json."""
import json
import pickle
import time
import numpy as np
import torch

from evaluate import load_model, bifurcation

torch.set_num_threads(1)
R = json.load(open("results/eval.json"))
data = pickle.load(open("data/benchmark.pkl", "rb"))
net = load_model("full")
Y = torch.tensor(np.stack([it["samples"] for it in data["test"]]).astype(np.float32))
with torch.no_grad():
    net(Y[:200])
    tb = min(_t for _t in [(lambda t0: (net(Y), time.perf_counter() - t0)[1])(time.perf_counter()) for _ in range(3)])
    ts = []
    for i in range(300):
        t0 = time.perf_counter(); net(Y[i:i + 1]); ts.append(time.perf_counter() - t0)
R["full"]["time_batch_ms"] = 1000 * tb / len(Y)
R["full"]["time_single_ms"] = 1000 * float(np.median(ts))
b = bifurcation(net)
R["bifurcation"].update(t_net=b["t_net"], t_cls=b["t_cls"], evals_cls=b["evals_cls"])
R["timing_note"] = "re-measured on an idle machine, one CPU thread"
json.dump(R, open("results/eval.json", "w"), indent=1)
print(R["full"]["time_batch_ms"], R["full"]["time_single_ms"], b["t_net"], b["t_cls"])
