# Catching Bitcoin Fraud with a Graph Convolutional Network

A 2-layer **Graph Convolutional Network (GCN)**, written from scratch in PyTorch, that classifies Bitcoin
transactions in the **Elliptic dataset** as *licit* or *illicit*.

**The idea:** a transaction is known by the company it keeps. A GCN looks at each transaction **and** the
transactions it sends money to or receives money from, instead of judging each one alone.

```
Problem: spot suspicious transactions
   → Graph: transactions (nodes) and money flows (edges)
   → GCN: learns from each transaction + its neighbours
   → Output: licit / illicit
```

## Results

The models are trained on time steps 1–34 and tested on time steps 35–49 (future transactions). The scores are
averaged over 5 runs and use all 165 features.

| Model | Precision | Recall | Illicit F1 | Ranking (AP) |
|---|---|---|---|---|
| Logistic Regression | 0.31 | 0.68 | 0.43 | 0.32 |
| MLP (same network, no graph) | 0.56 | 0.54 | 0.55 | 0.44 |
| **GCN (ours)** | **0.79** | 0.44 | 0.56 | **0.60** |
| Skip-GCN | 0.76 | 0.47 | 0.58 | 0.60 |
| Random Forest | 0.92 | 0.72 | **0.81** | 0.78 |

* **The graph makes alarms more trustworthy:** precision rises from 0.56 to 0.79 compared with the same network
  without the graph.
* **The Random Forest still wins,** matching the original Elliptic paper (Weber et al., 2019).
* **Every model fails after a dark-market shutdown at time step 43.** Criminals changed their behaviour, which is
  called concept drift.
* **Two layers work best.** Deeper GCNs over-smooth: all transactions start to look the same.

## What's in this repository

```
gcn_elliptic.ipynb              The model: data loading, GCN from scratch, baselines, experiments (already run)
requirements.txt
data/
  unpack_data.py                Rebuilds the large features file (run this first)
  elliptic_bitcoin_dataset/
    elliptic_txs_classes.csv    Labels: 1 = illicit, 2 = licit, unknown
    elliptic_txs_edgelist.csv   Money flows between transactions
    elliptic_txs_features.csv.gz.part00-03   Features, compressed and split (GitHub's 100 MB limit)
results/
  figures/                      All plots from the notebook (+ oversmoothing_untrained.png)
  final_results.csv             Summary table
  all_runs.csv                  Every model, every seed
  oversmoothing_untrained.csv   Similarity after k averaging steps, no training
analysis/
  implementation_check.py       Checks the GCN code against the formula and the hand calculation
  oversmoothing_untrained.py    Over-smoothing without training + lambda_2 of A_hat
report/
  gcn_elliptic_report.pdf             Report (simple, easy-to-read version)
  gcn_elliptic_report_detailed.pdf    Report (detailed, more technical version)
  latex/                              LaTeX source of both reports + figures
maths/
  gcn_maths_3page.pdf           Mathematical modelling + worked example (3 pages)
  gcn_maths_4page.pdf           Mathematical modelling + worked example (4 pages)
  gcn_maths_6page.pdf           Same, with more steps (6 pages)
  gcn_maths_detailed.pdf        Full derivations incl. backpropagation and over-smoothing (11 pages)
  latex/                        LaTeX source of the maths sheets
  verify_worked_example.py      Python check of every number in the worked example
```

## How to run

```bash
pip install -r requirements.txt
python data/unpack_data.py        # rebuilds elliptic_txs_features.csv (~370 MB)
jupyter notebook gcn_elliptic.ipynb
```

The full notebook takes about 2 hours on a 2-core CPU, and much less on a laptop or Colab. The notebook already
contains all outputs, so you can read the results without running it.

## The model in one line

```
Z = softmax( Â · ReLU( Â · X · W⁰ ) · W¹ ),   Â = D̃^(-1/2) (A + I) D̃^(-1/2)
```

The normalised adjacency matrix Â (203,769 × 203,769, only 672,479 non-zeros) is built by hand as a sparse
tensor. Each layer is a sparse matrix product; no library graph layer such as `GCNConv` is used.

## Dataset

The Elliptic Bitcoin dataset: 203,769 transactions, 234,355 edges, 165 features and 49 time steps. Of the
transactions, 4,545 are labelled illicit, 42,019 licit and 157,205 unknown.
The original source is Kaggle: <https://www.kaggle.com/datasets/ellipticco/elliptic-data-set>.

## References

1. T. N. Kipf and M. Welling. *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR 2017.
2. M. Weber et al. *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for
   Financial Forensics.* KDD Workshop on Anomaly Detection in Finance, 2019.
