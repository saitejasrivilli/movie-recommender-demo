"""
Benchmark SASRec vs SVD (matrix factorization) on MovieLens 100K.

Compares:
  - SVD: scipy TruncatedSVD on user-item interaction matrix (baseline)
  - SASRec: self-attentive sequential recommendation model

Evaluation: leave-one-out, 100-item candidate pool (1 pos + 99 negs)
Metrics: HR@10, NDCG@10
"""
import math
import random
import sys
import os
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_sasrec import (
    load_movielens_100k,
    prepare_sequences,
    remap_items,
    evaluate as sasrec_evaluate,
    train as sasrec_train,
    SASRecDataset,
)
from models.sasrec import SASRec


# ── SVD baseline ──────────────────────────────────────────────────────────────

def build_interaction_matrix(user_seqs: dict, n_users: int, n_items: int) -> np.ndarray:
    """Build binary user-item interaction matrix (n_users x n_items)."""
    uid_map = {u: i for i, u in enumerate(sorted(user_seqs.keys()))}
    mat = np.zeros((n_users, n_items), dtype=np.float32)
    for u, seq in user_seqs.items():
        for item in seq:
            if 1 <= item <= n_items:
                mat[uid_map[u], item - 1] = 1.0
    return mat, uid_map


def train_svd(train_seqs: dict, n_items: int, n_components: int = 64):
    """Fit TruncatedSVD on training interactions. Returns user/item embeddings."""
    from sklearn.decomposition import TruncatedSVD

    n_users = len(train_seqs)
    mat, uid_map = build_interaction_matrix(train_seqs, n_users, n_items)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_emb = svd.fit_transform(mat)                    # (n_users, k)
    item_emb = svd.components_.T                         # (n_items, k)

    # L2 normalise for cosine scoring
    user_emb = user_emb / (np.linalg.norm(user_emb, axis=1, keepdims=True) + 1e-8)
    item_emb = item_emb / (np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-8)
    return user_emb, item_emb, uid_map


def evaluate_svd(user_emb: np.ndarray, item_emb: np.ndarray,
                 uid_map: dict, test_seqs: dict, n_items: int, k: int = 10) -> dict:
    """Leave-one-out evaluation for SVD."""
    hits, ndcgs = [], []
    for uid, seq in test_seqs.items():
        if len(seq) < 2 or uid not in uid_map:
            continue
        target = seq[-1]
        history = set(seq[:-1])

        uidx = uid_map[uid]
        u_vec = user_emb[uidx]

        neg_set = history | {target}
        negs = []
        while len(negs) < 99:
            n = random.randint(1, n_items)
            if n not in neg_set:
                negs.append(n)
                neg_set.add(n)

        candidates = [target] + negs  # 0-indexed for item_emb
        cand_embs = item_emb[[c - 1 for c in candidates]]
        scores = cand_embs @ u_vec  # (100,)

        rank = (scores > scores[0]).sum()

        hits.append(1 if rank < k else 0)
        ndcgs.append(1.0 / math.log2(rank + 2) if rank < k else 0.0)

    return {
        f"hr@{k}": float(np.mean(hits)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
    }


# ── Shared data loading ───────────────────────────────────────────────────────

def load_data():
    df = load_movielens_100k()
    user_seqs = prepare_sequences(df, min_interactions=5, max_seq_len=50)
    user_seqs, n_items = remap_items(user_seqs)
    test_seqs = {u: seq for u, seq in user_seqs.items() if len(seq) >= 2}
    train_seqs = {u: seq[:-1] for u, seq in test_seqs.items()}
    return train_seqs, test_seqs, n_items


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_svd_benchmark(train_seqs, test_seqs, n_items, n_components=64):
    print("Running SVD benchmark...")
    t0 = time.time()
    user_emb, item_emb, uid_map = train_svd(train_seqs, n_items, n_components)
    train_time = time.time() - t0
    metrics = evaluate_svd(user_emb, item_emb, uid_map, test_seqs, n_items, k=10)
    return metrics, train_time


def run_sasrec_benchmark(train_seqs, test_seqs, n_items, epochs=20):
    print("Running SASRec benchmark...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SASRec(
        n_items=n_items, hidden_dim=64, n_heads=2, n_layers=2,
        max_seq_len=50, dropout=0.2,
    ).to(device)

    t0 = time.time()
    history = sasrec_train(model, train_seqs, n_items, epochs=epochs, lr=1e-3, batch_size=256)
    train_time = time.time() - t0

    metrics = sasrec_evaluate(model, test_seqs, n_items, k=10)
    return metrics, train_time, history


def print_comparison_table(svd_metrics, svd_time, sasrec_metrics, sasrec_time):
    hr_svd = svd_metrics["hr@10"]
    ndcg_svd = svd_metrics["ndcg@10"]
    hr_sar = sasrec_metrics["hr@10"]
    ndcg_sar = sasrec_metrics["ndcg@10"]

    hr_gain = (hr_sar - hr_svd) / max(hr_svd, 1e-9) * 100
    ndcg_gain = (ndcg_sar - ndcg_svd) / max(ndcg_svd, 1e-9) * 100

    header = f"{'Model':<10} | {'HR@10':>7} | {'NDCG@10':>8} | {'Train Time':>12} | Notes"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    print(f"{'SVD':<10} | {hr_svd:>7.3f} | {ndcg_svd:>8.3f} | {svd_time:>10.0f}s | Matrix factorization baseline")
    print(f"{'SASRec':<10} | {hr_sar:>7.3f} | {ndcg_sar:>8.3f} | {sasrec_time:>10.0f}s | "
          f"Sequential model, +{hr_gain:.0f}% HR@10")
    print(sep)
    print(f"\nSASRec vs SVD: HR@10 +{hr_gain:.1f}%, NDCG@10 +{ndcg_gain:.1f}%")


def main():
    print("=" * 60)
    print("SASRec vs SVD Benchmark on MovieLens 100K")
    print("=" * 60)
    print()

    train_seqs, test_seqs, n_items = load_data()
    print(f"Dataset: {len(train_seqs)} users, {n_items} items\n")

    svd_metrics, svd_time = run_svd_benchmark(train_seqs, test_seqs, n_items)
    print(f"SVD   HR@10={svd_metrics['hr@10']:.4f}  NDCG@10={svd_metrics['ndcg@10']:.4f}  "
          f"time={svd_time:.1f}s\n")

    sasrec_metrics, sasrec_time, _ = run_sasrec_benchmark(
        train_seqs, test_seqs, n_items, epochs=20
    )
    print(f"\nSASRec HR@10={sasrec_metrics['hr@10']:.4f}  "
          f"NDCG@10={sasrec_metrics['ndcg@10']:.4f}  time={sasrec_time:.1f}s")

    print_comparison_table(svd_metrics, svd_time, sasrec_metrics, sasrec_time)


if __name__ == "__main__":
    main()
