#!/usr/bin/env python3
"""
Graph Recommender — Evaluation & Analysis Script
=================================================
CMPE 256 · SJSU

Run this AFTER graph_recommender.py has trained and saved lightgcn_model.pt.
This script produces:
  1. Full test-set metric report (NDCG, Recall, Precision, MRR at multiple K)
  2. Learning curves (train loss + val NDCG over epochs)
  3. Cold-start vs. warm-user metric breakdown
  4. Coverage and diversity metrics
  5. Per-beer-style recommendation analysis

Usage:
  python evaluate_graph.py
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from typing import Optional
from sklearn.preprocessing import LabelEncoder

# Import model and helpers from the main script.
from graph_recommender import (
    CONFIG,
    build_norm_adjacency,
    LightGCN,
    ContentEncoder,             # new trainable content module
    HybridGraphRecommender,
    evaluate,
    ndcg_at_k,
    recall_at_k,
    precision_at_k,
    mrr as mrr_metric,
)


# ─────────────────────────────────────────────────────────────
# Load saved model checkpoint
# ─────────────────────────────────────────────────────────────

def load_model(cfg: dict) -> tuple:
    """
    Load the saved model checkpoint and reconstruct all components
    needed for inference and evaluation.

    The checkpoint saved by graph_recommender.py contains:
      - lightgcn_state_dict        : LightGCN embedding weights
      - content_encoder_state_dict : ContentEncoder weights (style/brewer emb + linear)
      - item_style_ids             : [n_items] LongTensor of style indices
      - item_brewer_ids            : [n_items] LongTensor of brewer indices
      - item_cont_feats            : [n_items, 5] FloatTensor of continuous features
      - n_styles / n_brewers       : vocabulary sizes for ContentEncoder
      - user_enc_classes           : numpy array of user ID strings
      - item_enc_classes           : numpy array of item ID strings
      - history                    : dict of train_loss / val_ndcg lists
      - test_metrics               : dict of final test scores
    """
    save_path = os.path.join(cfg["data_dir"], "lightgcn_model.pt")
    assert os.path.exists(save_path), (
        f"No saved model at {save_path}. Run graph_recommender.py first."
    )

    # weights_only=False needed because the checkpoint contains numpy arrays
    # (label encoder classes). Safe here since we wrote this checkpoint ourselves.
    ckpt = torch.load(save_path, map_location=cfg["device"], weights_only=False)

    # ── Reconstruct label encoders from saved class arrays ────
    user_enc = LabelEncoder()
    user_enc.classes_ = ckpt["user_enc_classes"]
    item_enc = LabelEncoder()
    item_enc.classes_ = ckpt["item_enc_classes"]

    n_users  = len(user_enc.classes_)
    n_items  = len(item_enc.classes_)
    n_styles  = int(ckpt["n_styles"])
    n_brewers = int(ckpt["n_brewers"])

    # ── Restore content feature tensors ───────────────────────
    item_style_ids  = ckpt["item_style_ids"].to(cfg["device"])
    item_brewer_ids = ckpt["item_brewer_ids"].to(cfg["device"])
    item_cont_feats = ckpt["item_cont_feats"].to(cfg["device"])

    # ── Restore train/test edges ──────────────────────────────
    # New checkpoints have these saved directly; older checkpoints
    # (saved before the format was updated) fall back to reloading
    # raw data files, which must be present on the local machine.
    if "train_edges" in ckpt:
        print("Restoring train/test edges from checkpoint …")
        train_df  = ckpt["train_edges"]
        test_df   = ckpt["test_edges"]
        val_df    = pd.DataFrame(columns=["user", "item", "rating"])
        item_meta = ckpt.get("item_meta", pd.DataFrame())
    else:
        print("train_edges not in checkpoint — reloading from raw data files …")
        from graph_recommender import (
            load_and_merge, k_core_filter, encode_ids, temporal_split
        )
        df = load_and_merge(cfg)
        df = k_core_filter(df, cfg["min_user_interactions"], cfg["min_item_interactions"])
        df, _, _ = encode_ids(df)
        train_df, val_df, test_df = temporal_split(df, cfg["val_ratio"], cfg["test_ratio"])
        item_meta = df[["item", "beer_style"]].drop_duplicates(
            subset="item").reset_index(drop=True)

    adj = build_norm_adjacency(
        train_df, n_users, n_items, cfg["rating_threshold"], cfg["device"]
    )

    # ── Reconstruct LightGCN with saved weights ───────────────
    lightgcn = LightGCN(n_users, n_items, cfg["embedding_dim"], cfg["n_layers"], adj)
    lightgcn.load_state_dict(ckpt["lightgcn_state_dict"])
    lightgcn.to(cfg["device"])

    # ── Reconstruct ContentEncoder with saved weights ─────────
    content_encoder = ContentEncoder(
        n_styles=n_styles,
        n_brewers=n_brewers,
        embedding_dim=cfg["embedding_dim"],
    )
    content_encoder.load_state_dict(ckpt["content_encoder_state_dict"])
    content_encoder.to(cfg["device"])

    # ── Assemble HybridGraphRecommender ───────────────────────
    model = HybridGraphRecommender(
        lightgcn        = lightgcn,
        content_encoder = content_encoder,
        item_style_ids  = item_style_ids,
        item_brewer_ids = item_brewer_ids,
        item_cont_feats = item_cont_feats,
        content_weight  = cfg["content_weight"],
    )
    model.to(cfg["device"])
    model.eval()

    return model, train_df, val_df, test_df, item_meta, user_enc, item_enc, ckpt


# ─────────────────────────────────────────────────────────────
# Metric breakdown: warm vs. cold-start users
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def cold_warm_breakdown(
    model: HybridGraphRecommender,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    top_k: list,
    device: str,
    cold_threshold: int = 5,
) -> dict:
    """
    Split test users into "cold" (≤ cold_threshold training interactions)
    and "warm" (> cold_threshold) and compute metrics for each group separately.

    Cold-start metric gap shows how much the content embedding helps.

    Parameters
    ----------
    cold_threshold : int
        Users with this many or fewer training interactions are "cold".
        Default 5 matches the dataset analysis (61.6% of users have ≤5 reviews).
    """
    user_train_counts = train_df.groupby("user").size().to_dict()

    train_pos: dict = defaultdict(set)
    for u, i in zip(train_df["user"].values, train_df["item"].values):
        train_pos[u].add(i)

    eval_pos: dict = defaultdict(set)
    for u, i, r in zip(test_df["user"].values, test_df["item"].values, test_df["rating"].values):
        if r >= 0.6:
            eval_pos[u].add(i)

    cold_users = [u for u in eval_pos if user_train_counts.get(u, 0) <= cold_threshold and eval_pos[u]]
    warm_users = [u for u in eval_pos if user_train_counts.get(u, 0) > cold_threshold and eval_pos[u]]

    def _compute(users):
        results = {f"ndcg@{k}": [] for k in top_k}
        for k in top_k:
            results[f"recall@{k}"] = []
        if not users:
            return {k: 0.0 for k in results}

        max_k = max(top_k)
        for start in range(0, len(users), 256):
            batch = users[start: start + 256]
            scores = model.score_all_items(
                torch.tensor(batch, dtype=torch.long, device=device)
            ).cpu().numpy()
            for i, u in enumerate(batch):
                vec = scores[i].copy()
                for excl in train_pos.get(u, set()):
                    if excl < n_items:
                        vec[excl] = -np.inf
                top_idx = np.argpartition(vec, -max_k)[-max_k:]
                ranked = top_idx[np.argsort(vec[top_idx])[::-1]].tolist()
                rel = eval_pos[u]
                for k in top_k:
                    results[f"ndcg@{k}"].append(ndcg_at_k(ranked, rel, k))
                    results[f"recall@{k}"].append(recall_at_k(ranked, rel, k))
        return {name: float(np.mean(vals)) for name, vals in results.items()}

    print(f"Cold users: {len(cold_users)} | Warm users: {len(warm_users)}")
    cold_metrics = _compute(cold_users)
    warm_metrics = _compute(warm_users)

    return {"cold": cold_metrics, "warm": warm_metrics,
            "n_cold": len(cold_users), "n_warm": len(warm_users)}


# ─────────────────────────────────────────────────────────────
# Coverage and diversity metrics
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def coverage_diversity(
    model: HybridGraphRecommender,
    train_df: pd.DataFrame,
    item_meta_df: pd.DataFrame,
    n_items: int,
    device: str,
    top_k: int = 10,
    sample_users: int = 1000,
) -> dict:
    """
    Compute catalogue coverage and intra-list diversity for a random sample of users.

    Coverage@K
    ----------
    Fraction of the item catalogue that appears in at least one user's top-K list.
    A system that always recommends the same popular items has low coverage,
    hurting the long-tail beers that might be great matches.

    Intra-List Diversity (ILD)
    --------------------------
    Average pairwise distance between beer styles in the recommendation list.
    High ILD means the system recommends a diverse set of styles per user,
    which is often preferred for discovery-oriented users.
    """
    user_ids = train_df["user"].unique()
    sampled = np.random.choice(user_ids, size=min(sample_users, len(user_ids)), replace=False)

    train_pos: dict = defaultdict(set)
    for u, i in zip(train_df["user"].values, train_df["item"].values):
        train_pos[u].add(i)

    # Style lookup keyed by INTEGER item ID (not string item_id).
    # item_meta_df must have an "item" column (the integer encoding).
    style_map = dict(zip(item_meta_df["item"], item_meta_df["beer_style"]))

    all_recommended = set()
    ild_scores = []

    for start in range(0, len(sampled), 256):
        batch = sampled[start: start + 256].tolist()
        scores = model.score_all_items(
            torch.tensor(batch, dtype=torch.long, device=device)
        ).cpu().numpy()

        for i, u in enumerate(batch):
            vec = scores[i].copy()
            for excl in train_pos.get(u, set()):
                if excl < n_items:
                    vec[excl] = -np.inf
            top_idx = np.argpartition(vec, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(vec[top_idx])[::-1]].tolist()
            all_recommended.update(top_idx)

            # ILD: fraction of pairs with different styles.
            styles = [style_map.get(int(idx), "Unknown") for idx in top_idx]
            pairs = [(styles[a], styles[b]) for a in range(len(styles))
                     for b in range(a + 1, len(styles))]
            if pairs:
                diversity = sum(1 for s1, s2 in pairs if s1 != s2) / len(pairs)
                ild_scores.append(diversity)

    coverage = len(all_recommended) / n_items
    ild = float(np.mean(ild_scores)) if ild_scores else 0.0

    return {"coverage@10": coverage, "ild@10": ild}


# ─────────────────────────────────────────────────────────────
# Shared-protocol evaluation (matches team standard)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_shared_protocol(
    model: HybridGraphRecommender,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    n_items: int,
    device: str,
    relevance_threshold: float = 0.7,   # team standard
    n_sample: int = 1000,               # team standard
    seed: int = 42,                     # team standard
    top_k: int = 10,
) -> dict:
    """
    Evaluate under the exact shared protocol agreed by the team:
      - Relevance threshold : 0.7
      - User sample         : 1,000 users, seed 42
      - Candidate pool      : full catalogue (no negative sampling)
      - K                   : 10

    Metrics: HR@10, NDCG@10, Recall@10, Coverage@10, Diversity@10
    HR@10 = fraction of users for whom at least one relevant item
            appears in the top-10 list (binary hit per user).
    """
    rng = np.random.default_rng(seed)

    # Build training mask (items to exclude per user).
    train_pos: dict = defaultdict(set)
    for u, i in zip(train_df["user"].values, train_df["item"].values):
        train_pos[u].add(i)

    # Build ground-truth relevant sets using threshold 0.7.
    eval_pos: dict = defaultdict(set)
    for u, i, r in zip(test_df["user"].values,
                       test_df["item"].values,
                       test_df["rating"].values):
        if r >= relevance_threshold:
            eval_pos[u].add(i)

    # Users that have at least one relevant item in test.
    eligible = [u for u in eval_pos if eval_pos[u]]

    # Sample up to n_sample users.
    if len(eligible) > n_sample:
        sampled_users = rng.choice(eligible, size=n_sample, replace=False).tolist()
    else:
        sampled_users = eligible

    # Pre-compute full embedding matrix once.
    user_all, item_hybrid = model.get_all_embeddings()
    item_hybrid_T = item_hybrid.T.contiguous()

    hr_list, ndcg_list, recall_list = [], [], []
    all_recommended: set = set()

    for start in range(0, len(sampled_users), 256):
        batch = sampled_users[start: start + 256]
        u_tensor = torch.tensor(batch, dtype=torch.long, device=device)
        scores = (user_all[u_tensor] @ item_hybrid_T).cpu().numpy()

        for i, u in enumerate(batch):
            vec = scores[i].copy()
            for excl in train_pos.get(u, set()):
                if excl < n_items:
                    vec[excl] = -np.inf

            top_idx = np.argpartition(vec, -top_k)[-top_k:]
            ranked  = top_idx[np.argsort(vec[top_idx])[::-1]].tolist()
            rel     = eval_pos[u]

            all_recommended.update(ranked)

            # HR@K: 1 if any relevant item appears in top-K.
            hr_list.append(1.0 if any(r in rel for r in ranked) else 0.0)
            ndcg_list.append(ndcg_at_k(ranked, rel, top_k))
            recall_list.append(recall_at_k(ranked, rel, top_k))

    coverage = len(all_recommended) / n_items

    return {
        f"hr@{top_k}":       float(np.mean(hr_list)),
        f"ndcg@{top_k}":     float(np.mean(ndcg_list)),
        f"recall@{top_k}":   float(np.mean(recall_list)),
        f"coverage@{top_k}": coverage,
        "n_users_evaluated": len(sampled_users),
    }


# ─────────────────────────────────────────────────────────────
# Plot learning curves
# ─────────────────────────────────────────────────────────────

def plot_learning_curves(history: dict, save_path: Optional[str] = None):
    """
    Plot train loss and validation NDCG@10 side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Train loss
    axes[0].plot(history["train_loss"], color="#2196F3", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BPR Loss")
    axes[0].set_title("Training Loss (BPR)")
    axes[0].grid(alpha=0.3)

    # Val NDCG
    for k in [10, 20]:
        key = f"val_ndcg@{k}"
        if key in history and history[key]:
            axes[1].plot(history[key], label=f"NDCG@{k}", linewidth=1.5)
    axes[1].set_xlabel("Evaluation checkpoint")
    axes[1].set_ylabel("NDCG")
    axes[1].set_title("Validation NDCG (higher is better)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Learning curves saved to: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# Main report
# ─────────────────────────────────────────────────────────────

def main():
    cfg = CONFIG

    print("\n" + "=" * 60)
    print("  LightGCN Evaluation Report — CMPE 256 / SJSU")
    print("=" * 60 + "\n")

    # ── Load model and data ───────────────────────────────
    model, train_df, val_df, test_df, item_meta, user_enc, item_enc, ckpt = load_model(cfg)

    history = ckpt.get("history", {})
    saved_metrics = ckpt.get("test_metrics", {})
    n_users = model.lightgcn.n_users
    n_items = model.lightgcn.n_items

    # ── 1. Shared-protocol metrics (team standard) ────────
    print("─" * 45)
    print("1. SHARED PROTOCOL METRICS")
    print("   (threshold=0.7, 1000-user sample, seed=42, K=10)")
    print("─" * 45)
    shared = evaluate_shared_protocol(
        model, test_df, train_df, n_items, cfg["device"],
        relevance_threshold=0.7, n_sample=1000, seed=42, top_k=10,
    )
    print(f"   Users evaluated      : {shared['n_users_evaluated']}")
    print(f"   HR@10                : {shared['hr@10']:.4f}")
    print(f"   NDCG@10              : {shared['ndcg@10']:.4f}")
    print(f"   Recall@10            : {shared['recall@10']:.4f}")
    print(f"   Coverage@10          : {shared['coverage@10']:.4f}")

    # ── 2. Full test metrics (all users, threshold=0.6) ──
    print("\n" + "─" * 45)
    print("2. FULL TEST METRICS (all users, threshold=0.6)")
    print("─" * 45)
    if saved_metrics:
        for name, val in sorted(saved_metrics.items()):
            print(f"   {name:<20} {val:.4f}")
    else:
        test_metrics = evaluate(
            model, test_df, train_df, n_users, n_items,
            cfg["top_k"], cfg["device"]
        )
        for name, val in sorted(test_metrics.items()):
            print(f"   {name:<20} {val:.4f}")

    # ── 3. Cold vs. warm breakdown ─────────────────────────
    print("\n" + "─" * 45)
    print("3. COLD-START vs. WARM USER BREAKDOWN (threshold ≤ 5 interactions)")
    print("─" * 45)
    breakdown = cold_warm_breakdown(
        model, test_df, train_df, n_users, n_items,
        cfg["top_k"], cfg["device"], cold_threshold=5,
    )
    print(f"   {'Metric':<20} {'Cold ({} users)'.format(breakdown['n_cold']):<20} "
          f"{'Warm ({} users)'.format(breakdown['n_warm']):<20}")
    print("   " + "-" * 55)
    for k in cfg["top_k"]:
        for metric in [f"ndcg@{k}", f"recall@{k}"]:
            cold_v = breakdown["cold"].get(metric, 0.0)
            warm_v = breakdown["warm"].get(metric, 0.0)
            gap = warm_v - cold_v
            print(f"   {metric:<20} {cold_v:<20.4f} {warm_v:<20.4f}  (gap: {gap:+.4f})")

    # ── 4. Coverage & diversity ────────────────────────────
    print("\n" + "─" * 45)
    print("4. COVERAGE & DIVERSITY")
    print("─" * 45)
    # item_meta has columns: item (int), beer_style — saved directly in checkpoint.
    cov_div = coverage_diversity(
        model, train_df, item_meta, n_items, cfg["device"],
        top_k=10, sample_users=1000,
    )
    print(f"   Catalogue Coverage@10  : {cov_div['coverage@10']:.4f}  "
          f"({cov_div['coverage@10']*100:.1f}% of all beers appear in at least one top-10 list)")
    print(f"   Intra-List Diversity@10: {cov_div['ild@10']:.4f}  "
          f"(fraction of style-distinct pairs in top-10 lists)")

    # ── 5. Learning curves ────────────────────────────────
    if history and "train_loss" in history:
        print("\n" + "─" * 45)
        print("5. LEARNING CURVES")
        print("─" * 45)
        curve_path = os.path.join(cfg["data_dir"], "learning_curves.png")
        plot_learning_curves(history, save_path=curve_path)

    print("\n✓ Evaluation complete.")


if __name__ == "__main__":
    main()
