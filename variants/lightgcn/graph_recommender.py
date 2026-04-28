#!/usr/bin/env python3
"""
Graph-Based Beer Recommender System — LightGCN + Content Features
==================================================================
CMPE 256 · Recommender Systems · SJSU
Variant: Graph-Based (Vishnu Peram)

Architecture Overview
---------------------
  1. DATA        : Load + merge BeerAdvocate and RateBeer datasets
  2. K-CORE      : Remove noisy low-interaction users/items from training graph
  3. SPLIT       : Temporal train/val/test split (no data leakage)
  4. CONTENT ENC : Encode beer style, brewer, and ABV as item features
  5. GRAPH       : Build symmetric normalized user-item bipartite adjacency matrix
  6. LightGCN    : Multi-layer graph convolution for collaborative embeddings
  7. HYBRID      : Blend collaborative + content embeddings for cold-start support
  8. BPR LOSS    : Bayesian Personalized Ranking — treats recsys as ranking problem
  9. EVALUATION  : NDCG@K, Recall@K, Precision@K, MRR

Why LightGCN over standard GCN for Collaborative Filtering?
  - Standard GCN applies feature transformation (W) + nonlinear activation at each layer,
    both of which add parameters that overfit and complicate CF where the only input
    features are IDs (no rich node attributes).
  - LightGCN (He et al., SIGIR 2020) removes both, keeping only neighbourhood aggregation:
      E^(k) = D^{-1/2} · A · D^{-1/2} · E^(k-1)
    This dramatically reduces overfitting and trains 2-3× faster.
  - Final embeddings are the mean across all layers (0 through K), capturing information
    from both immediate neighbours (k=1) and high-order neighbours (k=3+).

Why BPR over MSE for rating prediction?
  - Our goal is to rank beers the user will enjoy, not predict the exact rating.
  - MSE treats rating prediction as regression; BPR directly optimises ranking.
  - BPR loss: L = -Σ log σ(score(u,i⁺) - score(u,i⁻)) + λ‖Θ‖²
    For each user, it pushes positive items above randomly sampled negatives.

Cold-Start Strategy:
  - Items/users not seen during training get content-based scores only.
  - Beer item content = style embedding + brewer embedding + normalised ABV.
  - This provides reasonable recommendations even for beers with zero ratings.

Reference:
  He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020).
  LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.
  ACM SIGIR 2020. https://arxiv.org/abs/2002.02126
"""

import os
import ast
import time
import warnings
import gc
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# §1  CONFIGURATION
# ─────────────────────────────────────────────────────────────
# All tuneable hyperparameters live here so you can run sweeps
# without touching model code.

CONFIG = {
    # ── Paths ──────────────────────────────────────────────
    # Override with DATA_DIR environment variable when running on Colab:
    #   DATA_DIR=/content python graph_recommender.py
    "data_dir": os.environ.get("DATA_DIR", os.path.join(
        os.path.expanduser("~"),
        "Desktop/SJSU Grad/sem 2/256 Recommender Systems/Final Project",
    )),

    # ── Data Filtering ─────────────────────────────────────
    # K-core: only keep users AND items that each have at least
    # `min_interactions` reviews in the dataset.
    # Higher k → denser graph → better CF quality, fewer entities.
    # Typical range: 5–20.  Start at 10 for this dataset.
    "min_user_interactions": 10,
    "min_item_interactions": 10,

    # A rating ≥ threshold on [0,1] scale counts as a "positive"
    # interaction for BPR training.  0.6 ≈ 3/5 on BeerAdvocate scale.
    "rating_threshold": 0.6,

    # Fraction of each user's interactions held out for test / val.
    # We use the MOST RECENT reviews as test to avoid leakage.
    "test_ratio": 0.10,
    "val_ratio": 0.10,

    # ── Model ──────────────────────────────────────────────
    # Dimensionality of user and item latent embeddings.
    # Higher → more capacity, more memory.  Try 32/64/128.
    "embedding_dim": 64,

    # Number of LightGCN propagation layers.
    # Each layer aggregates one additional hop of neighbours.
    # 2–4 is the sweet spot; beyond 4 often hurts (over-smoothing).
    "n_layers": 3,

    # Weight of the content (cold-start) embedding relative to the
    # collaborative embedding when scoring items.
    # 0.0 = pure LightGCN CF;  1.0 = pure content.
    # 0.15–0.25 works well empirically when content features are noisy.
    "content_weight": 0.2,

    # ── Training ───────────────────────────────────────────
    "n_epochs": 50,   # increase to 50 for full run

    # Number of (user, pos_item, neg_item) BPR triples per batch.
    # Larger batches → faster GPU utilisation; 2048–8192 is typical.
    # We use 8192 here because the per-epoch propagation optimisation
    # means each batch is cheap; bigger batches reduce Python loop overhead.
    "batch_size": 8192,

    # Adam learning rate.  1e-3 is a reliable starting point.
    "learning_rate": 1e-3,

    # L2 weight-decay applied to embeddings only (prevents norm blow-up).
    # 1e-4 to 1e-5 is typical for LightGCN.
    "weight_decay": 1e-4,

    # Negative samples drawn per positive in each BPR batch.
    # 1 is standard; increase to 4–8 for harder negatives.
    "neg_samples": 1,

    # How often (in epochs) to run the full validation evaluation.
    # Evaluation is expensive (all-item scoring for every user).
    "eval_every": 5,

    # ── Evaluation ─────────────────────────────────────────
    # Compute metrics at each of these cutoffs.
    "top_k": [10, 20],

    # ── Hardware ───────────────────────────────────────────
    "device": (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    ),

    # Limit rows loaded during quick debugging.  Set to None for full run.
    "debug_sample": None,   # set to None for full overnight run
}


# ─────────────────────────────────────────────────────────────
# §2  DATA LOADING
# ─────────────────────────────────────────────────────────────

def _parse_file(path: str, n_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Parse one data file where every line is a Python dict literal
    (ast.literal_eval format, not true JSON).

    Returns a raw DataFrame.
    """
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if n_rows and i >= n_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                records.append(ast.literal_eval(line))
            except Exception:
                pass  # skip malformed lines
    return pd.DataFrame(records)


def _parse_rating(val, max_val: float = 1.0) -> float:
    """
    Normalise a rating value to [0, 1].
      - Plain float string → divide by max_val.
      - Fraction string 'a/b' → a/b (RateBeer uses this for sub-scores).
    """
    try:
        if isinstance(val, str) and "/" in val:
            num, denom = val.split("/")
            return float(num) / float(denom)
        return float(val) / max_val
    except Exception:
        return float("nan")


def load_and_merge(cfg: dict) -> pd.DataFrame:
    """
    Load BeerAdvocate + RateBeer files, normalise ratings to [0,1],
    create a globally unique item_id, deduplicate, and return a clean
    DataFrame with the columns needed by all model components.

    Key output columns
    ------------------
    user_raw   : raw username string
    item_id    : globally unique beer ID ('beeradvocate_47986', 'ratebeer_63836')
    rating     : normalised overall rating in [0, 1]
    beer_style : beer style string (179 unique values)
    beer_abv   : alcohol-by-volume float (may be NaN)
    brewer_id  : brewer identifier string
    timestamp  : Unix timestamp integer
    """
    sample = cfg.get("debug_sample")
    data_dir = cfg["data_dir"]

    print("Loading BeerAdvocate …")
    ba = _parse_file(os.path.join(data_dir, "beeradvocate.json"), sample)
    ba["source"] = "beeradvocate"

    print("Loading RateBeer …")
    rb = _parse_file(os.path.join(data_dir, "ratebeer.json"), sample)
    rb["source"] = "ratebeer"

    # ── Normalise ratings ─────────────────────────────────
    # BeerAdvocate uses a 0-5 scale; RateBeer uses fractions.
    # We parse both to [0,1] so the two platforms are comparable.
    for df, max_val in [(ba, 5.0), (rb, 1.0)]:
        for col in ["review/overall", "review/appearance",
                    "review/aroma", "review/palate", "review/taste"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda v: _parse_rating(v, max_val))

    combined = pd.concat([ba, rb], ignore_index=True)
    del ba, rb
    gc.collect()

    # ── Build globally unique item ID ─────────────────────
    # Raw beer IDs are platform-local integers that can overlap,
    # so we prefix them with the source platform name.
    beer_id_col = "beer/beerId" if "beer/beerId" in combined.columns else "beer/id"
    combined["item_id"] = combined["source"] + "_" + combined[beer_id_col].astype(str)

    # ── Standardise column names for downstream code ──────
    combined.rename(columns={
        "review/profileName": "user_raw",
        "review/overall": "rating",
        "beer/style": "beer_style",
        "beer/ABV": "beer_abv",
        "beer/brewerId": "brewer_id",
        "review/time": "timestamp",
    }, inplace=True)

    # Drop rows missing the three fields essential to every variant.
    combined.dropna(subset=["user_raw", "item_id", "rating"], inplace=True)

    # Keep only the most recent rating when a user rated the same beer twice.
    combined.sort_values("timestamp", inplace=True)
    combined.drop_duplicates(subset=["user_raw", "item_id"], keep="last", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # Convert ABV to numeric (some rows contain non-numeric strings).
    combined["beer_abv"] = pd.to_numeric(combined.get("beer_abv"), errors="coerce")

    print(f"Loaded {len(combined):,} interactions | "
          f"{combined['user_raw'].nunique():,} users | "
          f"{combined['item_id'].nunique():,} items")
    return combined


# ─────────────────────────────────────────────────────────────
# §3  K-CORE FILTERING
# ─────────────────────────────────────────────────────────────

def k_core_filter(df: pd.DataFrame, min_u: int, min_i: int) -> pd.DataFrame:
    """
    Iteratively remove users with < min_u interactions and items with
    < min_i interactions until convergence.

    Why iterative?  Removing a low-activity user may push a previously
    valid item below the threshold, so we repeat until no more removals.

    This produces the largest sub-dataset where every entity meets the
    minimum activity threshold — crucial for collaborative filtering
    quality on this very sparse beer dataset.
    """
    print(f"K-core filtering (min_user={min_u}, min_item={min_i}) …")
    prev_len = -1
    iteration = 0
    while len(df) != prev_len:
        prev_len = len(df)
        iteration += 1
        user_counts = df["user_raw"].value_counts()
        item_counts = df["item_id"].value_counts()
        valid_users = user_counts[user_counts >= min_u].index
        valid_items = item_counts[item_counts >= min_i].index
        df = df[df["user_raw"].isin(valid_users) & df["item_id"].isin(valid_items)]
    df = df.reset_index(drop=True)
    print(f"  After {iteration} passes: {len(df):,} interactions | "
          f"{df['user_raw'].nunique():,} users | {df['item_id'].nunique():,} items")
    return df


# ─────────────────────────────────────────────────────────────
# §4  INTEGER ENCODING
# ─────────────────────────────────────────────────────────────

def encode_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    Map raw string user/item IDs to contiguous integers starting at 0.
    This is required before building the adjacency matrix or embedding tables.

    Returns the encoded DataFrame and the two encoders (needed to decode
    back to raw IDs during evaluation).
    """
    user_enc = LabelEncoder().fit(df["user_raw"])
    item_enc = LabelEncoder().fit(df["item_id"])
    df = df.copy()
    df["user"] = user_enc.transform(df["user_raw"])
    df["item"] = item_enc.transform(df["item_id"])
    return df, user_enc, item_enc


# ─────────────────────────────────────────────────────────────
# §5  TEMPORAL TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame, val_ratio: float, test_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each user, sort their reviews by timestamp, then hold out the
    most recent `test_ratio` fraction as test and the next most recent
    `val_ratio` fraction as validation.  Everything else is training.

    Using temporal ordering prevents future-data leakage: we only
    recommend based on reviews that existed at the time of prediction.

    If a user has too few reviews to create a distinct test/val split
    they are kept in train only (cold-start evaluation uses content scores).
    """
    print("Splitting data temporally …")
    train_idx, val_idx, test_idx = [], [], []

    for _, grp in df.groupby("user"):
        grp_sorted = grp.sort_values("timestamp")
        n = len(grp_sorted)

        # Minimum 3 interactions needed to have train + val + test.
        if n < 3:
            train_idx.extend(grp_sorted.index.tolist())
            continue

        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))

        test_idx.extend(grp_sorted.index[-n_test:].tolist())
        val_idx.extend(grp_sorted.index[-(n_test + n_val):-n_test].tolist())
        train_idx.extend(grp_sorted.index[:-(n_test + n_val)].tolist())

    train = df.loc[train_idx].reset_index(drop=True)
    val   = df.loc[val_idx].reset_index(drop=True)
    test  = df.loc[test_idx].reset_index(drop=True)

    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
    return train, val, test


# ─────────────────────────────────────────────────────────────
# §6  CONTENT FEATURE ENCODING
# ─────────────────────────────────────────────────────────────

def build_item_features(
    df_all: pd.DataFrame,
    item_enc: LabelEncoder,
    device: str,
) -> tuple:
    """
    Build integer-encoded categorical features and normalised continuous
    features for every item in the k-core filtered catalogue.

    This replaces the old random-projection approach with proper input
    tensors that a *trainable* ContentEncoder will learn to embed.

    Categorical features (returned as integer ID tensors):
      • beer_style  — 179 unique style labels (e.g. "American IPA")
      • brewer_id   — brewery identifier

    Continuous features (returned as a [n_items, 5] float tensor):
      • ABV               — alcohol-by-volume, clipped to [0, 20] then /20
      • avg_appearance    — mean per-item sub-rating from reviews
      • avg_aroma         — mean per-item sub-rating
      • avg_palate        — mean per-item sub-rating
      • avg_taste         — mean per-item sub-rating

    Sub-ratings are aggregated over ALL reviews for each beer (not just
    training reviews), which gives stable item-level quality signals
    without leaking individual user preferences — these are objective
    sensory attributes, not personalised scores.

    Returns
    -------
    style_ids   : LongTensor  [n_items]
    brewer_ids  : LongTensor  [n_items]
    cont_feats  : FloatTensor [n_items, 5]
    n_styles    : int
    n_brewers   : int
    """
    n_items = len(item_enc.classes_)

    # ── Per-item metadata (one row per beer) ─────────────
    meta = (
        df_all[["item_id", "beer_style", "brewer_id", "beer_abv"]]
        .drop_duplicates("item_id")
        .copy()
    )
    meta["item"] = item_enc.transform(meta["item_id"])
    meta.sort_values("item", inplace=True)
    meta.reset_index(drop=True, inplace=True)

    # ── Per-item mean sub-ratings ─────────────────────────
    # Aggregate the four sensory sub-scores from all reviews for each
    # beer.  These capture objective quality dimensions (appearance,
    # aroma, palate, taste) independently of the overall rating target.
    sub_cols = ["review/appearance", "review/aroma",
                "review/palate", "review/taste"]
    available = [c for c in sub_cols if c in df_all.columns]

    if available:
        sub_mean = (
            df_all.groupby("item")[available]
            .mean()
            .reindex(range(n_items))   # ensure every item has a row
        )
        # Fill items with no sub-rating data with the global mean.
        sub_mean = sub_mean.fillna(sub_mean.mean())
        sub_arr = sub_mean.values.astype(np.float32)   # [n_items, len(available)]
        # Pad with zeros if some sub-rating columns were absent.
        if sub_arr.shape[1] < 4:
            pad = np.zeros((n_items, 4 - sub_arr.shape[1]), dtype=np.float32)
            sub_arr = np.concatenate([sub_arr, pad], axis=1)
    else:
        sub_arr = np.zeros((n_items, 4), dtype=np.float32)

    # ── ABV normalised to [0, 1] ─────────────────────────
    abv = meta["beer_abv"].values.astype(np.float32)
    abv_mean = float(np.nanmean(abv)) if np.any(~np.isnan(abv)) else 0.05
    abv = np.where(np.isnan(abv), abv_mean, abv)
    abv = np.clip(abv, 0.0, 20.0) / 20.0               # [n_items, 1]

    # ── Continuous feature matrix [n_items, 5] ───────────
    cont_feats = np.concatenate(
        [abv.reshape(-1, 1), sub_arr], axis=1
    ).astype(np.float32)

    # ── Categorical integer IDs ───────────────────────────
    # Style
    styles = meta["beer_style"].fillna("Unknown").astype(str)
    unique_styles = ["<PAD>"] + sorted(styles.unique().tolist())
    style2idx = {s: i for i, s in enumerate(unique_styles)}
    style_ids_arr = styles.map(style2idx).fillna(0).values.astype(np.int64)
    n_styles = len(unique_styles)

    # Brewer
    brewers = meta["brewer_id"].fillna("Unknown").astype(str)
    unique_brewers = ["<PAD>"] + sorted(brewers.unique().tolist())
    brewer2idx = {b: i for i, b in enumerate(unique_brewers)}
    brewer_ids_arr = brewers.map(brewer2idx).fillna(0).values.astype(np.int64)
    n_brewers = len(unique_brewers)

    # ── Convert to tensors on device ─────────────────────
    style_ids   = torch.tensor(style_ids_arr,  dtype=torch.long).to(device)
    brewer_ids  = torch.tensor(brewer_ids_arr, dtype=torch.long).to(device)
    cont_feats  = torch.tensor(cont_feats,     dtype=torch.float32).to(device)

    print(f"  Content features: {n_styles} styles | {n_brewers} brewers | "
          f"5 continuous (ABV + 4 sub-ratings)")

    return style_ids, brewer_ids, cont_feats, n_styles, n_brewers


# ─────────────────────────────────────────────────────────────
# §6b  CONTENT ENCODER  (trainable)
# ─────────────────────────────────────────────────────────────

class ContentEncoder(nn.Module):
    """
    Trainable content feature encoder for beer items.

    Uses LEARNED embeddings for categorical features (style, brewer) and
    a learned linear projection for continuous features (ABV + sub-ratings),
    making full use of all available side-information columns in the dataset.

    This satisfies the project requirement of using data beyond user/item/rating:
      • Beer style  (categorical, 179 unique values)
      • Brewer ID   (categorical, ~5 000+ unique values)
      • ABV         (continuous)
      • Appearance  (continuous, per-item aggregate sub-rating)
      • Aroma       (continuous, per-item aggregate sub-rating)
      • Palate      (continuous, per-item aggregate sub-rating)
      • Taste       (continuous, per-item aggregate sub-rating)

    Architecture:
      style_emb(style_id)   → [emb_dim]  ┐
      brewer_emb(brewer_id) → [emb_dim]  ├─ additive combination → LayerNorm
      Linear(5 → emb_dim)  → [emb_dim]  ┘

    Additive combination mirrors LightGCN's layer-mean convention and
    avoids the over-parameterisation of a concatenation + MLP approach.
    """

    def __init__(self, n_styles: int, n_brewers: int, embedding_dim: int):
        super().__init__()

        # Learned embeddings for categorical features.
        # padding_idx=0 maps the <PAD> token to a zero vector that never
        # receives gradient updates.
        self.style_emb  = nn.Embedding(n_styles,  embedding_dim, padding_idx=0)
        self.brewer_emb = nn.Embedding(n_brewers, embedding_dim, padding_idx=0)

        # Single linear layer to project 5 continuous inputs to embedding_dim.
        # Bias is included so the network can shift each dimension independently.
        self.cont_proj = nn.Linear(5, embedding_dim)

        # LayerNorm stabilises the combined embedding before it is blended
        # with the collaborative (LightGCN) component.
        self.norm = nn.LayerNorm(embedding_dim)

        # Xavier initialisation for stable early gradients.
        nn.init.xavier_uniform_(self.style_emb.weight)
        nn.init.xavier_uniform_(self.brewer_emb.weight)
        nn.init.xavier_uniform_(self.cont_proj.weight)

    def forward(
        self,
        style_ids:   torch.Tensor,   # [n_items] long
        brewer_ids:  torch.Tensor,   # [n_items] long
        cont_feats:  torch.Tensor,   # [n_items, 5] float
    ) -> torch.Tensor:
        """
        Returns a [n_items, embedding_dim] content embedding matrix.
        Called once per training epoch (not per batch) so the overhead is minimal.
        """
        s = self.style_emb(style_ids)       # [n_items, emb_dim]
        b = self.brewer_emb(brewer_ids)     # [n_items, emb_dim]
        c = self.cont_proj(cont_feats)      # [n_items, emb_dim]

        # Additive combination, then normalise.
        return self.norm((s + b + c) / 3.0)


# ─────────────────────────────────────────────────────────────
# §7  GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────

def build_norm_adjacency(
    train_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    rating_threshold: float,
    device: str,
) -> torch.Tensor:
    """
    Build the symmetrically normalised adjacency matrix used by LightGCN.

    The full adjacency matrix is block-structured:
        A = [ 0   R  ]
            [ R^T  0  ]
    where R[u, i] = 1 if user u gave item i a positive rating.

    Symmetric normalisation:  Â = D^{-1/2} · A · D^{-1/2}
    where D is the degree diagonal matrix.

    This normalisation ensures that aggregated messages are scaled by the
    square root of both the sender's and receiver's degree, preventing
    high-degree nodes from dominating low-degree ones.

    Returns a torch sparse COO tensor (stored on `device`).
    """
    # Keep only positive interactions as graph edges.
    pos = train_df[train_df["rating"] >= rating_threshold]

    rows = pos["user"].values          # user node IDs
    cols = pos["item"].values + n_users  # item node IDs (offset by n_users)
    N = n_users + n_items

    # Build upper and lower triangle of block adjacency simultaneously.
    row_all = np.concatenate([rows, cols])
    col_all = np.concatenate([cols, rows])
    data_all = np.ones(len(row_all), dtype=np.float32)

    # scipy sparse for degree computation and normalisation.
    A = sp.csr_matrix((data_all, (row_all, col_all)), shape=(N, N))

    # D^{-1/2} diagonal.
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)

    # Â = D^{-1/2} A D^{-1/2}
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    A_norm = A_norm.tocoo()

    indices = torch.tensor(
        np.vstack([A_norm.row, A_norm.col]), dtype=torch.long
    )
    values = torch.tensor(A_norm.data, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()

    # MPS (Apple Silicon) does not support sparse tensor operations, so the
    # adjacency matrix must stay on CPU when using MPS.  On CUDA and CPU
    # devices, move it to the target device so sparse matmul runs natively
    # on the GPU — this is the critical path for training speed.
    if device != "mps":
        adj = adj.to(device)

    return adj


# ─────────────────────────────────────────────────────────────
# §8  LIGHTGCN MODEL
# ─────────────────────────────────────────────────────────────

class LightGCN(nn.Module):
    """
    LightGCN: Light Graph Convolution Network for Recommendation.

    The model maintains two embedding tables:
      • E_user  : [n_users, embedding_dim]
      • E_item  : [n_items, embedding_dim]

    Forward pass (K layers of graph convolution):
      E^0 = concat(E_user, E_item)       # initial embeddings
      E^k = Â · E^{k-1}                  # neighbourhood aggregation
      E_final = (1/K+1) * Σ_{k=0}^{K} E^k  # layer-combination

    This propagation diffuses collaborative signal across multi-hop
    neighbours.  k=1 gives immediate neighbours; k=3 gives friends-of-
    friends-of-friends, capturing long-range taste patterns.

    Scoring:
      score(u, i) = e_u_final · e_i_final   (dot product)
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
        n_layers: int,
        adj_matrix: torch.Tensor,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.adj = adj_matrix  # pre-built normalised adjacency

        # Learnable embedding tables — the only parameters in LightGCN.
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)

        # Xavier uniform initialisation keeps gradients stable at start.
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run all K layers of graph convolution and return the layer-mean
        user and item embeddings.

        The layer-combination (mean over k=0..K) is a key design choice
        in LightGCN: it acts as an implicit ensemble of different-order
        neighbourhood signals and empirically outperforms using only the
        last-layer embedding.

        Implementation note: on MPS (Apple Silicon), sparse ops are unsupported
        so we move embeddings to CPU for the matmul.  On CUDA and CPU the
        adjacency matrix is already on the right device, so no transfer is
        needed — this is what makes CUDA training fast.
        """
        device = self.user_emb.weight.device
        on_mps  = device.type == "mps"   # torch.device("mps:0").type == "mps"

        # Stack user and item embeddings into one joint matrix [N, d].
        E = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        if on_mps:
            E = E.cpu()   # MPS: must move to CPU for sparse matmul

        layer_embeddings = [E]  # include layer-0 (raw embeddings) in the sum
        for _ in range(self.n_layers):
            # Sparse matrix multiply: [N, N] × [N, d] → [N, d]
            # Runs on GPU when device is CUDA, on CPU when device is MPS.
            E = torch.sparse.mm(self.adj, E)
            layer_embeddings.append(E)

        E_final = torch.stack(layer_embeddings, dim=0).mean(dim=0)
        if on_mps:
            E_final = E_final.to(device)   # move result back to MPS

        user_final = E_final[: self.n_users]
        item_final = E_final[self.n_users :]
        return user_final, item_final

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute embeddings for a training batch and return the three
        vectors needed for BPR loss: user, positive item, negative item.
        """
        user_all, item_all = self.propagate()
        e_u  = user_all[users]
        e_i_pos = item_all[pos_items]
        e_i_neg = item_all[neg_items]
        return e_u, e_i_pos, e_i_neg

    def score_all_items(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute scores for `user_ids` against ALL items (used in evaluation).
        Returns a [len(user_ids), n_items] score matrix.
        """
        user_all, item_all = self.propagate()
        e_u = user_all[user_ids]                 # [B, d]
        scores = e_u @ item_all.T                # [B, n_items]
        return scores


# ─────────────────────────────────────────────────────────────
# §9  HYBRID RECOMMENDER (LightGCN + Content)
# ─────────────────────────────────────────────────────────────

class HybridGraphRecommender(nn.Module):
    """
    Hybrid recommender that blends collaborative (LightGCN) item embeddings
    with fixed content (style + brewer + ABV) item embeddings.

    Blending formula for item representation:
      e_i_hybrid = (1 - α) · e_i_cf  +  α · e_i_content

    where α = content_weight (CONFIG["content_weight"]).

    Why blend?
      • Pure CF (α=0): excellent for warm users/items but zero signal
        for cold-start items not in the training graph.
      • Pure content (α=1): handles cold-start but ignores the rich
        preference patterns captured by neighbourhood propagation.
      • Blending (0 < α < 1): best of both worlds.  α=0.2 preserves
        most of the CF quality while adding cold-start robustness.
    """

    def __init__(
        self,
        lightgcn: LightGCN,
        content_encoder: "ContentEncoder",  # trainable — updated by optimiser
        item_style_ids:  torch.Tensor,      # [n_items] long — fixed input
        item_brewer_ids: torch.Tensor,      # [n_items] long — fixed input
        item_cont_feats: torch.Tensor,      # [n_items, 5] float — fixed input
        content_weight: float,
    ):
        super().__init__()
        self.lightgcn        = lightgcn
        self.content_encoder = content_encoder  # parameters ARE trained

        # Feature index tensors are fixed inputs, not parameters.
        # register_buffer makes them move to the right device automatically.
        self.register_buffer("item_style_ids",  item_style_ids)
        self.register_buffer("item_brewer_ids", item_brewer_ids)
        self.register_buffer("item_cont_feats", item_cont_feats)
        self.alpha = content_weight

    def _hybrid_item_emb(self, item_all: torch.Tensor) -> torch.Tensor:
        """
        Blend collaborative (LightGCN) and content (learned encoder) embeddings.

        The content encoder is called once here; its output depends on the
        learnable style/brewer embedding tables and the continuous feature
        projection, all of which receive gradient updates through this path.
        """
        content_emb = self.content_encoder(
            self.item_style_ids,
            self.item_brewer_ids,
            self.item_cont_feats,
        )                                               # [n_items, emb_dim]
        return (1 - self.alpha) * item_all + self.alpha * content_emb

    def get_all_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute full-catalogue embeddings (user, hybrid-item) in one go.

        This is the heart of our performance optimisation: graph propagation
        is expensive (3× CPU sparse matmuls on a 67K×67K adjacency), so we
        call it ONCE and reuse the result across all mini-batches in the
        epoch — a standard LightGCN convention.  Without this, propagate()
        would be called n_batches times per epoch (1549× for our dataset),
        making training ~20× slower.
        """
        user_all, item_all = self.lightgcn.propagate()
        item_hybrid = self._hybrid_item_emb(item_all)
        return user_all, item_hybrid

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convenience path when per-batch propagation is acceptable (unused
        in the optimised training loop, which pre-computes embeddings once
        per epoch for a massive speedup)."""
        user_all, item_hybrid = self.get_all_embeddings()

        e_u     = user_all[users]
        e_i_pos = item_hybrid[pos_items]
        e_i_neg = item_hybrid[neg_items]
        return e_u, e_i_pos, e_i_neg

    def score_all_items(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Score all items for the given users.  Used in single-call
        evaluation paths; the batched evaluator below pre-computes the full
        embedding matrix once to avoid repeated propagation."""
        user_all, item_hybrid = self.get_all_embeddings()
        e_u = user_all[user_ids]
        return e_u @ item_hybrid.T


# ─────────────────────────────────────────────────────────────
# §10  BPR DATASET & DATALOADER
# ─────────────────────────────────────────────────────────────

class BPRDataset(Dataset):
    """
    PyTorch Dataset that yields (user, positive_item, negative_item) triples.

    For each sample:
      • user and positive_item are drawn from training positives.
      • negative_item is sampled uniformly from items the user has NOT
        positively interacted with.

    Uniform negative sampling is the standard BPR approach.  For harder
    negatives (items the model currently scores high but are not truly
    positive), you can implement popularity-based or in-batch negatives
    — both usually improve NDCG by ~1–2 points on sparse datasets.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        n_items: int,
        rating_threshold: float,
        neg_samples: int = 1,
    ):
        pos = train_df[train_df["rating"] >= rating_threshold]
        self.users     = pos["user"].values.astype(np.int64)
        self.pos_items = pos["item"].values.astype(np.int64)
        self.n_items   = n_items
        self.neg_samples = neg_samples

        # Build per-user set of positive items for fast exclusion check.
        # Using frozenset for O(1) lookup.
        self.user_pos: dict[int, set] = defaultdict(set)
        for u, i in zip(self.users, self.pos_items):
            self.user_pos[u].add(i)

    def __len__(self) -> int:
        return len(self.users) * self.neg_samples

    def __getitem__(self, idx: int):
        real_idx = idx // self.neg_samples
        u = int(self.users[real_idx])
        i_pos = int(self.pos_items[real_idx])

        # Sample a negative item the user hasn't rated positively.
        # At most 100 attempts; if all fail (densely rated user), accept anyway.
        pos_set = self.user_pos[u]
        for _ in range(100):
            i_neg = np.random.randint(0, self.n_items)
            if i_neg not in pos_set:
                break

        return (
            torch.tensor(u,     dtype=torch.long),
            torch.tensor(i_pos, dtype=torch.long),
            torch.tensor(i_neg, dtype=torch.long),
        )


# ─────────────────────────────────────────────────────────────
# §11  BPR LOSS
# ─────────────────────────────────────────────────────────────

def bpr_loss(
    e_u: torch.Tensor,
    e_i_pos: torch.Tensor,
    e_i_neg: torch.Tensor,
    weight_decay: float,
) -> torch.Tensor:
    """
    Bayesian Personalized Ranking loss.

    Mathematical form:
      L = -1/|S| · Σ log σ(ê_u·ê_i⁺ - ê_u·ê_i⁻)  + λ·(‖e_u‖² + ‖e_i⁺‖² + ‖e_i⁻‖²)

    The main term maximises the score gap between positives and negatives.
    The L2 regularisation term (λ = weight_decay) prevents embeddings
    from growing unboundedly, which would saturate the sigmoid.

    Note: only the INITIAL (layer-0) embeddings are regularised, not the
    propagated ones — this is the convention in LightGCN's original code.
    """
    pos_scores = (e_u * e_i_pos).sum(dim=1)  # dot product per sample
    neg_scores = (e_u * e_i_neg).sum(dim=1)

    # Main BPR term
    bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

    # L2 regularisation on the batch embeddings
    reg = weight_decay * (
        e_u.norm(2).pow(2) +
        e_i_pos.norm(2).pow(2) +
        e_i_neg.norm(2).pow(2)
    ) / e_u.shape[0]

    return bpr + reg


# ─────────────────────────────────────────────────────────────
# §12  EVALUATION METRICS
# ─────────────────────────────────────────────────────────────

def ndcg_at_k(ranked_items: list[int], relevant: set[int], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at K.

    DCG@K  = Σ_{j=1}^{K} rel_j / log2(j+1)
    IDCG@K = Σ_{j=1}^{min(|relevant|,K)} 1 / log2(j+1)  (ideal ordering)
    NDCG@K = DCG@K / IDCG@K

    NDCG penalises relevant items ranked lower.  It is the primary metric
    for ranking quality in recommender systems literature.
    """
    dcg = sum(
        1.0 / np.log2(j + 2)
        for j, item in enumerate(ranked_items[:k])
        if item in relevant
    )
    idcg = sum(1.0 / np.log2(j + 2) for j in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_items: list[int], relevant: set[int], k: int) -> float:
    """Recall@K = |relevant ∩ top-K| / |relevant|"""
    hits = sum(1 for item in ranked_items[:k] if item in relevant)
    return hits / len(relevant) if relevant else 0.0


def precision_at_k(ranked_items: list[int], relevant: set[int], k: int) -> float:
    """Precision@K = |relevant ∩ top-K| / K"""
    hits = sum(1 for item in ranked_items[:k] if item in relevant)
    return hits / k


def mrr(ranked_items: list[int], relevant: set[int]) -> float:
    """
    Mean Reciprocal Rank: 1/rank_of_first_hit.
    MRR focuses on how quickly the first relevant item appears in the
    ranked list — useful when users care most about the top result.
    """
    for rank, item in enumerate(ranked_items, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


@torch.no_grad()
def evaluate(
    model: HybridGraphRecommender,
    eval_df: pd.DataFrame,
    train_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    top_k_list: list[int],
    device: str,
    batch_size: int = 256,
) -> dict[str, float]:
    """
    Full-ranking evaluation: for each user in `eval_df`, score ALL items,
    mask out training items, and compute ranking metrics.

    Returns a dict mapping metric names to their mean values over users.
    E.g., {'ndcg@10': 0.142, 'recall@10': 0.183, 'precision@10': 0.018, 'mrr': 0.211}

    We use full-ranking (all N items) rather than sampled negatives because
    sampled evaluation is biased and can produce misleading comparisons
    between methods (Krichene & Rendle, RecSys 2020).
    """
    model.eval()
    max_k = max(top_k_list)

    # ── Build lookup structures ───────────────────────────
    # user → set of items seen in training (to exclude from recommendations)
    train_pos: dict[int, set] = defaultdict(set)
    for u, i in zip(train_df["user"].values, train_df["item"].values):
        train_pos[u].add(i)

    # user → set of items in the eval split (ground truth)
    eval_pos: dict[int, set] = defaultdict(set)
    for u, i, r in zip(eval_df["user"].values,
                        eval_df["item"].values,
                        eval_df["rating"].values):
        if r >= 0.6:   # only count positive interactions as relevant
            eval_pos[u].add(i)

    # Users that appear in eval AND have at least one positive.
    eval_users = [u for u in eval_pos if eval_pos[u]]

    # Initialise metric accumulators.
    metrics: dict[str, list[float]] = {
        f"ndcg@{k}": [] for k in top_k_list
    }
    for k in top_k_list:
        metrics[f"recall@{k}"] = []
        metrics[f"precision@{k}"] = []
    metrics["mrr"] = []

    # ── Pre-compute full embedding matrix ONCE ────────────
    # Critical optimisation: without this, propagate() would be called
    # ceil(n_eval_users / batch_size) times (≈75 for our dataset),
    # multiplying evaluation time by 75×.
    user_all, item_hybrid = model.get_all_embeddings()
    item_hybrid_T = item_hybrid.T.contiguous()   # pre-transpose for fast matmul

    # ── Batch score and rank ──────────────────────────────
    for start in range(0, len(eval_users), batch_size):
        batch_users = eval_users[start: start + batch_size]
        user_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)

        # Slice pre-computed user embeddings, then MPS matmul against all items.
        # scores: [batch, n_items]
        e_u = user_all[user_tensor]
        scores = (e_u @ item_hybrid_T).cpu().numpy()

        for i, u in enumerate(batch_users):
            score_vec = scores[i]

            # Mask training items with -inf so they cannot appear in top-K.
            for excl in train_pos.get(u, set()):
                if excl < n_items:
                    score_vec[excl] = -np.inf

            # Rank all items descending; take top max_k.
            top_indices = np.argpartition(score_vec, -max_k)[-max_k:]
            top_indices = top_indices[np.argsort(score_vec[top_indices])[::-1]]
            ranked = top_indices.tolist()
            relevant = eval_pos[u]

            for k in top_k_list:
                metrics[f"ndcg@{k}"].append(ndcg_at_k(ranked, relevant, k))
                metrics[f"recall@{k}"].append(recall_at_k(ranked, relevant, k))
                metrics[f"precision@{k}"].append(precision_at_k(ranked, relevant, k))
            metrics["mrr"].append(mrr(ranked, relevant))

    return {name: float(np.mean(vals)) for name, vals in metrics.items()}


# ─────────────────────────────────────────────────────────────
# §13  TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def train(
    model: HybridGraphRecommender,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_items: int,
    cfg: dict,
) -> dict[str, list]:
    """
    Full training loop with:
      • Adam optimiser with weight-decay on embedding parameters only.
      • BPR loss computed over batches of (user, pos, neg) triples.
      • Periodic validation evaluation.
      • Early stopping based on NDCG@10.
      • Learning-rate scheduler (cosine annealing) to improve convergence.

    Returns a history dict with train loss and val metrics per epoch.
    """
    device   = cfg["device"]
    n_epochs = cfg["n_epochs"]
    lr       = cfg["learning_rate"]
    wd       = cfg["weight_decay"]
    top_k    = cfg["top_k"]
    k_main   = min(top_k)  # primary metric for early stopping

    # Build BPR dataset
    dataset = BPRDataset(
        train_df,
        n_items,
        cfg["rating_threshold"],
        cfg["neg_samples"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,   # set >0 only if your OS supports fork-based multiprocessing
        pin_memory=(device == "cuda"),
    )

    # Optimise LightGCN embeddings AND the ContentEncoder (style/brewer
    # embedding tables + continuous feature projection).
    # The content buffers (style_ids, brewer_ids, cont_feats) are NOT
    # parameters — only the encoder's weight matrices are trained.
    optimizer = optim.Adam(
        list(model.lightgcn.parameters()) +
        list(model.content_encoder.parameters()),
        lr=lr,
        weight_decay=0,   # L2 regularisation is applied manually in bpr_loss
    )
    # Cosine annealing: smoothly decays LR to lr/10 by the last epoch.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr / 10
    )

    history = {
        "train_loss": [],
        **{f"val_ndcg@{k}": [] for k in top_k},
        **{f"val_recall@{k}": [] for k in top_k},
    }

    best_ndcg = -1.0
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        # ── Pre-compute LightGCN propagation ONCE per epoch ──────────
        # propagate() does 3 CPU sparse matmuls (~0.4 s total).
        # Calling it once per epoch instead of once per mini-batch saves
        # n_batches × 0.4 s ≈ 150 s per epoch on the full dataset.
        #
        # Why is it safe to retain this graph across batches?
        # Because LightGCN's backward through nn.Embedding does NOT need
        # the saved weight values — it only uses the saved indices and
        # the incoming gradient.  So optimizer.step() modifying
        # user_emb.weight / item_emb.weight in-place between batches
        # cannot invalidate the retained graph.
        #
        # The ContentEncoder (nn.Linear + LayerNorm) is DIFFERENT: its
        # backward DOES need saved activations, which are invalidated by
        # in-place weight updates.  So we recompute it fresh per batch —
        # it is cheap (two embedding lookups + one linear on n_items rows).
        user_cf_all, item_cf_all = model.lightgcn.propagate()

        n_batches = len(loader)
        for batch_idx, (users, pos_items, neg_items) in enumerate(loader):
            users     = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            # Content encoder: recomputed each batch so its computation
            # graph is always fresh (no retain_graph needed for this part).
            content_emb = model.content_encoder(
                model.item_style_ids,
                model.item_brewer_ids,
                model.item_cont_feats,
            )
            # Combine CF (from retained epoch graph) + content (fresh graph).
            item_hybrid = (
                (1 - model.alpha) * item_cf_all + model.alpha * content_emb
            )

            e_u     = user_cf_all[users]
            e_i_pos = item_hybrid[pos_items]
            e_i_neg = item_hybrid[neg_items]
            loss    = bpr_loss(e_u, e_i_pos, e_i_neg, wd)

            optimizer.zero_grad()
            # Retain only the LightGCN propagation graph (safe).
            # ContentEncoder graph is released normally after each backward.
            is_last_batch = (batch_idx == n_batches - 1)
            loss.backward(retain_graph=not is_last_batch)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        # Release retained graph memory promptly.
        del user_cf_all, item_cf_all

        scheduler.step()
        avg_loss = total_loss / len(loader)
        history["train_loss"].append(avg_loss)

        # ── Validation ───────────────────────────────────
        if epoch % cfg["eval_every"] == 0 or epoch == n_epochs:
            val_metrics = evaluate(
                model, val_df, train_df,
                model.lightgcn.n_users,
                model.lightgcn.n_items,
                top_k, device,
            )
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"Loss: {avg_loss:.4f} | "
                + " | ".join(f"NDCG@{k}: {val_metrics[f'ndcg@{k}']:.4f}" for k in top_k)
                + f" | {elapsed:.1f}s"
            )
            for k in top_k:
                history[f"val_ndcg@{k}"].append(val_metrics[f"ndcg@{k}"])
                history[f"val_recall@{k}"].append(val_metrics[f"recall@{k}"])

            # Early stopping on primary metric
            if val_metrics[f"ndcg@{k_main}"] > best_ndcg:
                best_ndcg = val_metrics[f"ndcg@{k_main}"]
                # Save both LightGCN and ContentEncoder weights.
                best_state = {
                    "lightgcn": {k: v.clone() for k, v in model.lightgcn.state_dict().items()},
                    "content_encoder": {k: v.clone() for k, v in model.content_encoder.state_dict().items()},
                }
        else:
            elapsed = time.time() - t0
            print(f"Epoch {epoch:3d}/{n_epochs} | Loss: {avg_loss:.4f} | {elapsed:.1f}s")

    # Restore best weights before returning.
    if best_state is not None:
        model.lightgcn.load_state_dict(best_state["lightgcn"])
        model.content_encoder.load_state_dict(best_state["content_encoder"])
        print(f"\nBest model restored (Val NDCG@{k_main} = {best_ndcg:.4f})")

    return history


# ─────────────────────────────────────────────────────────────
# §14  INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def recommend_for_user(
    model: HybridGraphRecommender,
    user_raw_id: str,
    user_enc: LabelEncoder,
    item_enc: LabelEncoder,
    item_meta: pd.DataFrame,
    train_df: pd.DataFrame,
    top_n: int,
    device: str,
) -> pd.DataFrame:
    """
    Generate top-N beer recommendations for a single user.

    If the user is not in the training set (cold-start user), falls back
    to globally popular beers — a simple but strong cold-start baseline.

    Returns a DataFrame with columns: [beer_name, beer_style, beer_abv,
    item_id, score].
    """
    model.eval()

    # ── Cold-start user ───────────────────────────────────
    if user_raw_id not in user_enc.classes_:
        print(f"User '{user_raw_id}' not in training set — returning popular beers.")
        popular = (
            train_df.groupby("item")["rating"]
            .agg(["mean", "count"])
            .query("count >= 10")
            .sort_values("mean", ascending=False)
            .head(top_n)
            .index.tolist()
        )
        recs = []
        for iid in popular:
            raw_id = item_enc.inverse_transform([iid])[0]
            row = item_meta[item_meta["item_id"] == raw_id]
            recs.append({
                "item_id": raw_id,
                "beer_name": row["beer/name"].values[0] if len(row) else "Unknown",
                "beer_style": row["beer_style"].values[0] if len(row) else "Unknown",
                "beer_abv": row["beer_abv"].values[0] if len(row) else float("nan"),
                "score": float("nan"),
            })
        return pd.DataFrame(recs)

    # ── Known user ────────────────────────────────────────
    u_int = int(user_enc.transform([user_raw_id])[0])
    user_tensor = torch.tensor([u_int], dtype=torch.long, device=device)
    scores = model.score_all_items(user_tensor)[0].cpu().numpy()

    # Exclude items already rated in training.
    seen = set(train_df[train_df["user"] == u_int]["item"].values)
    for i in seen:
        scores[i] = -np.inf

    top_indices = np.argsort(scores)[::-1][:top_n]
    raw_item_ids = item_enc.inverse_transform(top_indices)

    recs = []
    for idx, raw_id in zip(top_indices, raw_item_ids):
        row = item_meta[item_meta["item_id"] == raw_id]
        recs.append({
            "item_id": raw_id,
            "beer_name": row["beer/name"].values[0] if len(row) else "Unknown",
            "beer_style": row["beer_style"].values[0] if len(row) else "Unknown",
            "beer_abv": row["beer_abv"].values[0] if len(row) else float("nan"),
            "score": float(scores[idx]),
        })
    return pd.DataFrame(recs)


# ─────────────────────────────────────────────────────────────
# §15  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def main():
    cfg = CONFIG
    print(f"\n{'='*60}")
    print("  LightGCN Beer Recommender — CMPE 256 / SJSU")
    print(f"  Device: {cfg['device']}")
    print(f"{'='*60}\n")

    # ── 1. Load & merge ───────────────────────────────────
    df = load_and_merge(cfg)

    # Keep item metadata BEFORE k-core filter (needed for cold-start inference).
    item_meta = (
        df[["item_id", "beer/name", "beer_style", "beer_abv", "brewer_id"]]
        .drop_duplicates("item_id")
        .copy()
        if "beer/name" in df.columns
        else df[["item_id", "beer_style", "beer_abv", "brewer_id"]]
        .drop_duplicates("item_id")
        .copy()
    )

    # ── 2. K-core filter ──────────────────────────────────
    df = k_core_filter(df, cfg["min_user_interactions"], cfg["min_item_interactions"])

    # ── 3. Integer encode ─────────────────────────────────
    df, user_enc, item_enc = encode_ids(df)
    n_users = df["user"].nunique()
    n_items = df["item"].nunique()
    print(f"Graph: {n_users} users × {n_items} items\n")

    # ── 4. Temporal split ─────────────────────────────────
    train_df, val_df, test_df = temporal_split(df, cfg["val_ratio"], cfg["test_ratio"])

    # ── 5. Content features (categorical IDs + continuous) ─
    print("Building item content features …")
    style_ids, brewer_ids, cont_feats, n_styles, n_brewers = build_item_features(
        df, item_enc, cfg["device"]
    )

    # ── 6. Normalised adjacency matrix ────────────────────
    print("Building normalised graph adjacency matrix …")
    adj = build_norm_adjacency(
        train_df, n_users, n_items, cfg["rating_threshold"], cfg["device"]
    )

    # ── 7. Build model ────────────────────────────────────
    lightgcn = LightGCN(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=cfg["embedding_dim"],
        n_layers=cfg["n_layers"],
        adj_matrix=adj,
    ).to(cfg["device"])

    # Trainable content encoder: learns style/brewer embeddings and a
    # projection for ABV + 4 sub-ratings.  Parameters are updated jointly
    # with LightGCN embeddings via the shared Adam optimiser.
    content_encoder = ContentEncoder(
        n_styles=n_styles,
        n_brewers=n_brewers,
        embedding_dim=cfg["embedding_dim"],
    ).to(cfg["device"])

    model = HybridGraphRecommender(
        lightgcn=lightgcn,
        content_encoder=content_encoder,
        item_style_ids=style_ids,
        item_brewer_ids=brewer_ids,
        item_cont_feats=cont_feats,
        content_weight=cfg["content_weight"],
    ).to(cfg["device"])

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # ── 8. Train ──────────────────────────────────────────
    print("\n── Training ──────────────────────────────────────────")
    history = train(model, train_df, val_df, n_items, cfg)

    # ── 9. Final test evaluation ──────────────────────────
    print("\n── Test Evaluation ───────────────────────────────────")
    test_metrics = evaluate(
        model, test_df, train_df,
        n_users, n_items,
        cfg["top_k"], cfg["device"],
    )
    print("\n  Final Test Metrics:")
    for name, val in sorted(test_metrics.items()):
        print(f"    {name:<20} {val:.4f}")

    # ── 10. Save model ────────────────────────────────────
    save_path = os.path.join(cfg["data_dir"], "lightgcn_model.pt")
    # Save train/test edge lists so evaluate_graph.py never needs the raw
    # data files (which may not be present on the evaluation machine).
    # Only keep columns needed for evaluation to minimise checkpoint size.
    train_edges = train_df[["user", "item", "rating"]].reset_index(drop=True)
    test_edges  = test_df[["user", "item", "rating"]].reset_index(drop=True)
    # Item metadata: integer item id + beer style (for diversity metrics).
    item_meta   = df[["item", "beer_style"]].drop_duplicates(
        subset="item").reset_index(drop=True)

    torch.save({
        "lightgcn_state_dict":        model.lightgcn.state_dict(),
        "content_encoder_state_dict": model.content_encoder.state_dict(),
        "item_style_ids":             style_ids.cpu(),
        "item_brewer_ids":            brewer_ids.cpu(),
        "item_cont_feats":            cont_feats.cpu(),
        "n_styles":                   n_styles,
        "n_brewers":                  n_brewers,
        "user_enc_classes":           user_enc.classes_,
        "item_enc_classes":           item_enc.classes_,
        "train_edges":                train_edges,
        "test_edges":                 test_edges,
        "item_meta":                  item_meta,
        "config":                     cfg,
        "test_metrics":               test_metrics,
        "history":                    history,
    }, save_path)
    print(f"\nModel saved to: {save_path}")

    # ── 11. Sample recommendations ────────────────────────
    print("\n── Sample Recommendations ────────────────────────────")
    sample_user = df["user_raw"].iloc[0]
    recs = recommend_for_user(
        model, sample_user, user_enc, item_enc,
        item_meta, train_df, top_n=10, device=cfg["device"],
    )
    print(f"\nTop-10 recommendations for user '{sample_user}':")
    print(recs[["beer_name", "beer_style", "beer_abv", "score"]].to_string(index=False))

    return model, history, test_metrics


if __name__ == "__main__":
    main()
