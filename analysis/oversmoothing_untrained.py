"""Over-smoothing without training: apply A_hat repeatedly to random features.

H_k = A_hat^k H_0   (no weights, no ReLU), H_0 ~ N(0, 1) with 16 dimensions.
We measure the mean pairwise cosine similarity of 1,000 random nodes,
over the whole graph and inside its largest connected component, and
compute lambda_2 = max_{i>=2} |mu_i| of A_hat on the largest component.

Run from the repository root:  python analysis/oversmoothing_untrained.py
"""
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

D = "data/elliptic_bitcoin_dataset"
feat = pd.read_csv(os.path.join(D, "elliptic_txs_features.csv"), header=None, usecols=[0])
edges = pd.read_csv(os.path.join(D, "elliptic_txs_edgelist.csv"))
ids = feat[0].values
N = len(ids)
pos = pd.Series(np.arange(N), index=ids)
src, dst = pos[edges.txId1].values, pos[edges.txId2].values

# A_hat = D~^-1/2 (A + I) D~^-1/2   (same construction as in the notebook)
A = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(N, N)).tocsr()
A = ((A + A.T) > 0).astype(np.float64)
At = A + sp.eye(N)
d = np.asarray(At.sum(1)).ravel()
Dm = sp.diags(d ** -0.5)
A_hat = (Dm @ At @ Dm).tocsr()

n_comp, comp = connected_components(A, directed=False)
largest = np.argmax(np.bincount(comp))
in_lc = np.where(comp == largest)[0]
print(f"components: {n_comp}, largest component: {len(in_lc):,} nodes")

rng = np.random.default_rng(0)
H = rng.standard_normal((N, 16))
idx_all = rng.choice(N, 1000, replace=False)
idx_lc = rng.choice(in_lc, 1000, replace=False)

def mean_cos(H, idx):
    Z = H[idx] / np.linalg.norm(H[idx], axis=1, keepdims=True)
    S = Z @ Z.T
    n = len(idx)
    return (S.sum() - n) / (n * (n - 1))

rows, k_done = [], 0
for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    while k_done < k:
        H = A_hat @ H
        k_done += 1
    rows.append((k, mean_cos(H, idx_all), mean_cos(H, idx_lc)))
    print(f"k={k:4d}  all nodes {rows[-1][1]:.4f}   largest component {rows[-1][2]:.4f}")

# second-largest |eigenvalue| of A_hat on the largest component
A_lc = A_hat[in_lc][:, in_lc]
vals = eigsh(A_lc, k=3, which="LM", return_eigenvectors=False)
vals = sorted(np.abs(vals), reverse=True)
print(f"largest |eigenvalues| on largest component: {[round(v, 5) for v in vals]}")
print(f"lambda_2 = {vals[1]:.5f}")

os.makedirs("results", exist_ok=True)
pd.DataFrame(rows, columns=["k", "cos_all_nodes", "cos_largest_component"]).to_csv(
    "results/oversmoothing_untrained.csv", index=False)

# figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
df = pd.DataFrame(rows, columns=["k", "all", "lc"])
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "font.size": 10})
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(df.k, df.lc, marker="o", lw=2, color="#2a78d6", label=f"largest component ({len(in_lc):,} nodes)")
ax.plot(df.k, df["all"], marker="o", lw=2, color="#eb6834", label="all nodes (49 components)")
ax.axhline(1, color="grey", ls=":", lw=1); ax.text(1, 0.95, "complete collapse (similarity = 1)", color="grey", fontsize=8)
ax.set_xscale("log", base=2); ax.set_xticks(df.k, [str(k) for k in df.k])
ax.set(xlabel="number of propagation steps k", ylabel="mean cosine similarity", ylim=(-0.05, 1.05),
       title="Over-smoothing without training: $H_k = \\hat{A}^k H_0$")
ax.legend(frameon=False, loc="center left")
plt.tight_layout(); plt.savefig("results/figures/oversmoothing_untrained.png", dpi=200)
print("figure saved")
