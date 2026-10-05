"""Implementation check: does our GCN class compute exactly the GCN formula?

Uses the 5-transaction example from the maths sheet. We build A_hat with the
same build_norm_adj function as the notebook, run the notebook's GCN class,
and compare with the formula written out by hand:

    Z = A_hat . ReLU(A_hat . X . W0 + b0) . W1 + b1

Run from the repository root:  python analysis/implementation_check.py
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# take build_norm_adj and the GCN classes straight from the notebook, so the check uses the real code
nb = json.load(open("gcn_elliptic.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
ns = {"torch": torch, "nn": nn, "F": F}
exec(cells[7].split("A_hat = build_norm_adj")[0], ns)   # build_norm_adj, identity_adj
exec(cells[8].split("m = GCN(")[0], ns)                  # GCNLayer, GCN
build_norm_adj, GCN = ns["build_norm_adj"], ns["GCN"]

# the 5-transaction graph and features from the maths sheet
src = torch.tensor([0, 1, 1, 2, 3])
dst = torch.tensor([1, 2, 3, 3, 4])
X = torch.tensor([[1., 0.], [1., 1.], [1., 0.], [0., 1.], [0., 1.]])
A_hat = build_norm_adj(src, dst, 5)

# 1. A_hat matches the hand-computed matrix
s2, s3 = 2 ** 0.5, 3 ** 0.5
A_hand = torch.tensor([
    [1/2, 1/(2*s2), 0, 0, 0],
    [1/(2*s2), 1/4, 1/(2*s3), 1/4, 0],
    [0, 1/(2*s3), 1/3, 1/(2*s3), 0],
    [0, 1/4, 1/(2*s3), 1/4, 1/(2*s2)],
    [0, 0, 0, 1/(2*s2), 1/2]])
print("max |A_hat - hand calculation| =", (A_hat.to_dense() - A_hand).abs().max().item())
print("A_hat symmetric:", torch.equal(A_hat.to_dense(), A_hat.to_dense().T))

# 2. the model output matches the formula, for random weights
torch.manual_seed(0)
model = GCN(2, 4, 2).eval()
with torch.no_grad():
    for p in model.parameters():
        p.normal_()
    Z_model = model(A_hat, X)
    Z_model_pre = model(A_hat, X, AX=torch.sparse.mm(A_hat, X))   # the pre-computed A_hat X path
    W0, b0 = model.layers[0].W, model.layers[0].b
    W1, b1 = model.layers[1].W, model.layers[1].b
    A = A_hat.to_dense()
    Z_formula = A @ torch.relu(A @ X @ W0 + b0) @ W1 + b1
print("max |model - formula| =", (Z_model - Z_formula).abs().max().item())
print("max |model (pre-computed A_hat X) - formula| =", (Z_model_pre - Z_formula).abs().max().item())

# 3. with the weights of the worked example, the probabilities match the maths sheet
with torch.no_grad():
    model2 = GCN(2, 2, 2).eval()
    model2.layers[0].W.copy_(torch.eye(2)); model2.layers[0].b.zero_()
    model2.layers[1].W.copy_(torch.tensor([[-1., 1.], [1., -1.]])); model2.layers[1].b.zero_()
    P = model2(A_hat, X).softmax(1)
print("probabilities from our model:\n", P.numpy().round(3))
print("node 0 illicit prob =", round(P[0, 1].item(), 3), " (maths sheet: 0.685)")
print("node 4 licit prob   =", round(P[4, 0].item(), 3), " (maths sheet: 0.746)")
