"""
Train SASRec on MovieLens 100K.

Data: downloaded via HuggingFace datasets or direct URL fallback.
Feedback: implicit (any rating = interaction).
Sequences: sorted by timestamp per user.
Loss: BCE with 1 positive + 99 random negatives per step.
Eval: leave-one-out — last item = test, rest = history.
Metrics: HR@10 (Hit Rate), NDCG@10.
"""
import math
import random
import time
import sys
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.sasrec import SASRec


# ── Data loading ──────────────────────────────────────────────────────────────

def load_movielens_100k():
    """Load MovieLens 100K ratings. Returns a DataFrame with columns: user_id, item_id, timestamp."""
    try:
        from datasets import load_dataset
        ds = load_dataset("nateraw/movielens-100k", split="train", trust_remote_code=True)
        import pandas as pd
        df = ds.to_pandas()
        # column name normalisation
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if "user" in cl:
                col_map[c] = "user_id"
            elif "item" in cl or "movie" in cl:
                col_map[c] = "item_id"
            elif "time" in cl or "stamp" in cl:
                col_map[c] = "timestamp"
            elif "rating" in cl:
                col_map[c] = "rating"
        df = df.rename(columns=col_map)
        if "user_id" not in df.columns or "item_id" not in df.columns:
            raise ValueError("column mapping failed")
        if "timestamp" not in df.columns:
            df["timestamp"] = range(len(df))
        return df[["user_id", "item_id", "timestamp"]]
    except Exception:
        pass

    # fallback: download u.data directly
    import urllib.request
    import io
    import pandas as pd
    url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            raw = r.read().decode()
        df = pd.read_csv(
            io.StringIO(raw),
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
        )
        return df[["user_id", "item_id", "timestamp"]]
    except Exception:
        pass

    # final fallback: generate synthetic data that mirrors ML-100K statistics
    print("WARNING: could not fetch MovieLens 100K; generating synthetic data.")
    import pandas as pd
    rng = np.random.default_rng(42)
    n_users, n_items, n_rows = 943, 1682, 100_000
    df = pd.DataFrame({
        "user_id": rng.integers(1, n_users + 1, n_rows),
        "item_id": rng.integers(1, n_items + 1, n_rows),
        "timestamp": rng.integers(880_000_000, 900_000_000, n_rows),
    })
    return df


def prepare_sequences(ratings_df, min_interactions: int = 5, max_seq_len: int = 50) -> dict:
    """Returns {user_id: [item_id_1, item_id_2, ...]} sorted by timestamp."""
    ratings_df = ratings_df.copy()
    ratings_df = ratings_df.sort_values("timestamp")
    user_seqs = {}
    for uid, grp in ratings_df.groupby("user_id"):
        items = grp["item_id"].tolist()
        if len(items) >= min_interactions:
            user_seqs[uid] = items[-max_seq_len:]  # keep most recent
    return user_seqs


# ── Item id remapping ─────────────────────────────────────────────────────────

def remap_items(user_seqs: dict):
    """Remap item ids to 1-indexed contiguous integers. Returns new seqs + n_items."""
    all_items = sorted({item for seq in user_seqs.values() for item in seq})
    item2idx = {item: idx + 1 for idx, item in enumerate(all_items)}
    remapped = {u: [item2idx[i] for i in seq] for u, seq in user_seqs.items()}
    return remapped, len(all_items)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SASRecDataset(Dataset):
    """
    Each sample: input sequence (L-1 items) → target = last item.
    Negative sampling: 99 random items per positive.
    """
    def __init__(self, user_seqs: dict, n_items: int, max_seq_len: int = 50, n_neg: int = 99):
        self.samples = []
        self.n_items = n_items
        self.n_neg = n_neg
        self.max_seq_len = max_seq_len

        for uid, seq in user_seqs.items():
            if len(seq) < 2:
                continue
            # use all but last item as input, predict each next item
            for t in range(1, len(seq)):
                inp = seq[max(0, t - max_seq_len):t]
                pos = seq[t]
                self.samples.append((inp, pos))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, pos = self.samples[idx]
        L = self.max_seq_len
        # left-pad with 0
        padded = [0] * (L - len(inp)) + inp
        padded = padded[:L]
        # negative sampling
        negs = []
        item_set = set(inp) | {pos}
        while len(negs) < self.n_neg:
            neg = random.randint(1, self.n_items)
            if neg not in item_set:
                negs.append(neg)
        seq_t = torch.tensor(padded, dtype=torch.long)
        pos_t = torch.tensor(pos, dtype=torch.long)
        neg_t = torch.tensor(negs, dtype=torch.long)
        return seq_t, pos_t, neg_t


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: SASRec, test_seqs: dict, n_items: int, k: int = 10) -> dict:
    """
    Leave-one-out: last item = test, rest = history.
    For each test user sample 99 random negatives + the positive → rank among 100.
    """
    model.eval()
    device = next(model.parameters()).device
    max_seq_len = model.max_seq_len
    hits, ndcgs = [], []

    with torch.no_grad():
        for uid, seq in test_seqs.items():
            if len(seq) < 2:
                continue
            history = seq[:-1]
            target = seq[-1]

            inp = history[-max_seq_len:]
            padded = [0] * (max_seq_len - len(inp)) + inp
            seq_t = torch.tensor([padded], dtype=torch.long, device=device)

            # 99 negatives
            neg_set = set(history)
            neg_set.add(target)
            negs = []
            while len(negs) < 99:
                n = random.randint(1, n_items)
                if n not in neg_set:
                    negs.append(n)
                    neg_set.add(n)

            candidates = torch.tensor([target] + negs, dtype=torch.long, device=device)
            scores = model.predict(seq_t, candidates)[0]  # (100,)
            rank = (scores > scores[0]).sum().item()  # how many scored higher than positive

            hits.append(1 if rank < k else 0)
            if rank < k:
                ndcgs.append(1.0 / math.log2(rank + 2))
            else:
                ndcgs.append(0.0)

    return {
        f"hr@{k}": float(np.mean(hits)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train(model: SASRec, train_seqs: dict, n_items: int,
          epochs: int = 20, lr: float = 1e-3, batch_size: int = 256) -> list:
    """Returns list of {epoch, loss, hr@10, ndcg@10}."""
    device = next(model.parameters()).device
    dataset = SASRecDataset(train_seqs, n_items, max_seq_len=model.max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for seq_batch, pos_batch, neg_batch in loader:
            seq_batch = seq_batch.to(device)
            pos_batch = pos_batch.to(device)
            neg_batch = neg_batch.to(device)

            # forward — use last position representation
            h = model(seq_batch)[:, -1, :]  # (B, D)

            # positive scores
            pos_emb = model.item_emb(pos_batch)  # (B, D)
            pos_scores = (h * pos_emb).sum(-1)  # (B,)

            # negative scores: average over all negatives
            neg_emb = model.item_emb(neg_batch)  # (B, N, D)
            neg_scores = torch.bmm(neg_emb, h.unsqueeze(-1)).squeeze(-1)  # (B, N)

            pos_labels = torch.ones_like(pos_scores)
            neg_labels = torch.zeros_like(neg_scores)

            loss = bce(pos_scores, pos_labels) + bce(neg_scores, neg_labels)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        metrics = evaluate(model, train_seqs, n_items, k=10)
        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "loss": round(avg_loss, 4),
            "hr@10": round(metrics["hr@10"], 4),
            "ndcg@10": round(metrics["ndcg@10"], 4),
            "time_s": round(elapsed, 1),
        }
        history.append(row)
        print(f"Epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}  "
              f"HR@10={metrics['hr@10']:.4f}  NDCG@10={metrics['ndcg@10']:.4f}  "
              f"({elapsed:.1f}s)")

    return history


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading MovieLens 100K...")
    df = load_movielens_100k()
    print(f"Loaded {len(df):,} interactions from {df['user_id'].nunique()} users")

    user_seqs = prepare_sequences(df, min_interactions=5, max_seq_len=50)
    user_seqs, n_items = remap_items(user_seqs)
    print(f"Sequences: {len(user_seqs)} users, {n_items} items")

    # train/test split: last item of each user = test
    test_seqs = {u: seq for u, seq in user_seqs.items() if len(seq) >= 2}
    train_seqs = {u: seq[:-1] for u, seq in test_seqs.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = SASRec(
        n_items=n_items,
        hidden_dim=64,
        n_heads=2,
        n_layers=2,
        max_seq_len=50,
        dropout=0.2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"SASRec parameters: {n_params:,}")

    print("\nTraining for 20 epochs...")
    history = train(model, train_seqs, n_items, epochs=20, lr=1e-3, batch_size=256)

    print("\nEvaluating on held-out last item...")
    final_metrics = evaluate(model, test_seqs, n_items, k=10)
    print(f"\nFinal Test Metrics:")
    print(f"  HR@10  = {final_metrics['hr@10']:.4f}")
    print(f"  NDCG@10 = {final_metrics['ndcg@10']:.4f}")

    return model, history, final_metrics


if __name__ == "__main__":
    main()
