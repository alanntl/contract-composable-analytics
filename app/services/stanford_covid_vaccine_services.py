"""
Stanford COVID Vaccine (OpenVaccine) - Contract-Composable Analytics Services
======================================================

Competition: https://www.kaggle.com/competitions/stanford-covid-vaccine
Problem Type: Sequence-to-sequence regression (RNA degradation prediction)
Targets: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
Metric: MCRMSE (Mean Columnwise RMSE)

Services:
  - prepare_rna_dataset: Load JSONL + BPPS → token-encoded feature arrays
  - train_rna_gru_lstm: Train bidirectional GRU+LSTM ensemble with k-fold
  - predict_rna_ensemble: Predict on test (107+130 length sequences)
  - format_rna_submission: Format predictions into Kaggle submission CSV
"""

import os
import sys
import json
import pickle
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract

# Lazy torch imports
_torch = None
_nn = None
_F = None


def _import_torch():
    global _torch, _nn, _F
    if _torch is None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        _torch = torch
        _nn = nn
        _F = F
    return _torch, _nn, _F


def _get_device():
    """Get the best available compute device."""
    torch, _, _ = _import_torch()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# CONSTANTS
# =============================================================================

TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
ERROR_COLS = ["reactivity_error", "deg_error_Mg_pH10", "deg_error_pH10",
              "deg_error_Mg_50C", "deg_error_50C"]
TOKEN_MAP = {x: i for i, x in enumerate("().ACGUBEHIMSX")}
N_TOKENS = len(TOKEN_MAP)  # 14

# One-hot encoding maps for AE pipeline
_SEQ_MAP = {c: i for i, c in enumerate("ACGU")}
_STRUCT_MAP = {c: i for i, c in enumerate("().")}
_LOOP_MAP = {c: i for i, c in enumerate("BEHIMSX")}


# =============================================================================
# DATA UTILITIES (AE PIPELINE)
# =============================================================================

def _one_hot_encode(sequence, structure, loop_type):
    """Encode RNA features as 14-dim one-hot vector per position."""
    L = len(sequence)
    feat = np.zeros((L, 14), dtype=np.float32)
    for i in range(L):
        feat[i, _SEQ_MAP.get(sequence[i], 0)] = 1.0
        feat[i, 4 + _STRUCT_MAP.get(structure[i], 0)] = 1.0
        feat[i, 7 + _LOOP_MAP.get(loop_type[i], 0)] = 1.0
    return feat


def _compute_structure_adj(structure):
    """Compute structure adjacency matrix from dot-bracket notation."""
    L = len(structure)
    adj = np.zeros((L, L), dtype=np.float32)
    stack = []
    for i, c in enumerate(structure):
        if c == '(':
            stack.append(i)
        elif c == ')' and stack:
            j = stack.pop()
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    return adj


_DIST_CACHE = {}

def _compute_distance_matrix(seq_len):
    """Compute 3-channel distance matrix (powers 1, 2, 4)."""
    if seq_len in _DIST_CACHE:
        return _DIST_CACHE[seq_len]
    idx = np.arange(seq_len, dtype=np.float32)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.clip(dist, 1, None)
    d1 = 1.0 / dist
    d2 = 1.0 / dist ** 2
    d4 = 1.0 / dist ** 4
    np.fill_diagonal(d1, 0)
    np.fill_diagonal(d2, 0)
    np.fill_diagonal(d4, 0)
    result = np.stack([d1, d2, d4], axis=0)  # (3, L, L)
    _DIST_CACHE[seq_len] = result
    return result


def _load_sample_bpp_features(sample_id, bpps_dir, structure, seq_len):
    """Load BPP matrix and compute 5-channel 2D features (BPP + adj + dist*3)."""
    bpp_path = os.path.join(bpps_dir, f"{sample_id}.npy")
    if os.path.exists(bpp_path):
        bpp = np.load(bpp_path).astype(np.float32)
    else:
        bpp = np.zeros((seq_len, seq_len), dtype=np.float32)
    adj = _compute_structure_adj(structure)
    dist = _compute_distance_matrix(seq_len)
    return np.concatenate([[bpp], [adj], dist], axis=0)  # (5, L, L)


# =============================================================================
# SERVICE 1: PREPARE RNA DATASET
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "json", "required": True},
        "test_data": {"format": "json", "required": True},
    },
    outputs={
        "train_features": {"format": "npz"},
        "test_features": {"format": "npz"},
    },
    description="Load RNA JSONL data with BPPS features, token-encode, and build feature arrays",
    tags=["preprocessing", "rna", "feature-engineering"],
)
def prepare_rna_dataset(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    bpps_dir: str = "",
    token_map: str = "().ACGUBEHIMSX",
    **params,
) -> str:
    """Load train/test JSONL, extract BPPS, token-encode, save as npz."""

    tok2int = {x: i for i, x in enumerate(token_map)}

    def load_jsonl(path):
        records = []
        with open(path) as f:
            for line in f:
                records.append(json.loads(line.strip()))
        return pd.DataFrame(records)

    def extract_bpps_features(df, bpps_base):
        bpps_sum_list, bpps_max_list = [], []
        for mol_id in df["id"]:
            npy_path = os.path.join(bpps_base, f"{mol_id}.npy")
            if os.path.exists(npy_path):
                bpp = np.load(npy_path)
                bpps_sum_list.append(bpp.sum(axis=1))
                bpps_max_list.append(bpp.max(axis=1))
            else:
                seq_len = len(df[df["id"] == mol_id]["sequence"].values[0])
                bpps_sum_list.append(np.zeros(seq_len))
                bpps_max_list.append(np.zeros(seq_len))
        return bpps_sum_list, bpps_max_list

    def encode_features(df, bpps_sum, bpps_max, tok2int):
        """Build (n_samples, seq_len, 5) feature array."""
        features_list = []
        for i in range(len(df)):
            row = df.iloc[i]
            sl = len(row["sequence"])
            feat = np.zeros((sl, 5), dtype=np.float32)
            # 3 categorical channels (as integer tokens)
            feat[:, 0] = [tok2int.get(c, 0) for c in row["sequence"]]
            feat[:, 1] = [tok2int.get(c, 0) for c in row["structure"]]
            feat[:, 2] = [tok2int.get(c, 0) for c in row["predicted_loop_type"]]
            # 2 numerical channels
            feat[:, 3] = bpps_sum[i]
            feat[:, 4] = bpps_max[i]
            features_list.append(feat)

        return features_list

    # Resolve bpps directory
    if bpps_dir and not os.path.isabs(bpps_dir):
        # Try relative to base_path (same dir as inputs)
        base = os.path.dirname(os.path.dirname(inputs["train_data"]))
        bpps_base = os.path.join(base, bpps_dir)
        if not os.path.isdir(bpps_base):
            bpps_base = bpps_dir
    else:
        bpps_base = bpps_dir or os.path.join(os.path.dirname(inputs["train_data"]), "bpps")

    # Load data
    train_df = load_jsonl(inputs["train_data"])
    test_df = load_jsonl(inputs["test_data"])
    print(f"  Loaded train: {len(train_df)} samples, test: {len(test_df)} samples")

    # Extract BPPS features
    print(f"  Extracting BPPS features from {bpps_base}...")
    train_bpps_sum, train_bpps_max = extract_bpps_features(train_df, bpps_base)
    test_bpps_sum, test_bpps_max = extract_bpps_features(test_df, bpps_base)

    # Encode features
    train_features = encode_features(train_df, train_bpps_sum, train_bpps_max, tok2int)
    test_features = encode_features(test_df, test_bpps_sum, test_bpps_max, tok2int)

    # Build target arrays for train
    train_targets = []
    for i in range(len(train_df)):
        row = train_df.iloc[i]
        scored = int(row["seq_scored"])
        t = np.zeros((scored, 5), dtype=np.float32)
        for j, col in enumerate(TARGET_COLS):
            vals = row[col]
            t[:, j] = vals[:scored]
        train_targets.append(t)

    # Extract metadata
    train_ids = train_df["id"].values
    test_ids = test_df["id"].values
    train_sn = train_df["signal_to_noise"].values.astype(np.float32)
    train_snf = train_df["SN_filter"].values.astype(np.float32)
    train_seq_lengths = train_df["seq_length"].values.astype(np.int32)
    train_seq_scored = train_df["seq_scored"].values.astype(np.int32)
    test_seq_lengths = test_df["seq_length"].values.astype(np.int32)
    test_seq_scored = test_df["seq_scored"].values.astype(np.int32)

    # For train: all same length, stack directly
    train_X = np.array(train_features, dtype=np.float32)
    train_y = np.array(train_targets, dtype=np.float32)

    # For test: group by length
    test_107_mask = test_seq_lengths == 107
    test_130_mask = test_seq_lengths == 130

    test_107_X = np.array([f for f, m in zip(test_features, test_107_mask) if m], dtype=np.float32)
    test_130_X = np.array([f for f, m in zip(test_features, test_130_mask) if m], dtype=np.float32)
    test_107_ids = test_ids[test_107_mask]
    test_130_ids = test_ids[test_130_mask]

    # Save
    os.makedirs(os.path.dirname(outputs["train_features"]) or ".", exist_ok=True)
    np.savez_compressed(
        outputs["train_features"],
        X=train_X, y=train_y,
        ids=train_ids, signal_to_noise=train_sn, SN_filter=train_snf,
        seq_lengths=train_seq_lengths, seq_scored=train_seq_scored,
    )

    os.makedirs(os.path.dirname(outputs["test_features"]) or ".", exist_ok=True)
    np.savez_compressed(
        outputs["test_features"],
        X_107=test_107_X, ids_107=test_107_ids,
        X_130=test_130_X, ids_130=test_130_ids,
        seq_lengths=test_seq_lengths, seq_scored=test_seq_scored,
    )

    print(f"  Train features: {train_X.shape}, targets: {train_y.shape}")
    print(f"  Test 107: {test_107_X.shape}, Test 130: {test_130_X.shape}")
    return f"prepare_rna_dataset: train={len(train_df)}, test={len(test_df)} (107:{test_107_X.shape[0]}, 130:{test_130_X.shape[0]})"


# =============================================================================
# SERVICE 2: TRAIN RNA GRU+LSTM
# =============================================================================

def _build_rna_model(rnn_type, n_tokens, embed_dim, hidden_dim, n_layers,
                     dropout, spatial_dropout, n_targets=5):
    """Build PyTorch RNA degradation model."""
    torch, nn, F = _import_torch()

    class SpatialDropout1D(nn.Module):
        def __init__(self, p):
            super().__init__()
            self.p = p

        def forward(self, x):
            if not self.training or self.p == 0:
                return x
            # x: (batch, seq_len, features)
            x = x.permute(0, 2, 1)  # (batch, features, seq_len)
            x = F.dropout2d(x, p=self.p, training=self.training)
            return x.permute(0, 2, 1)

    class RNADegradationModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 3 embedding layers for 3 categorical features
            self.embed_seq = nn.Embedding(n_tokens, embed_dim)
            self.embed_struct = nn.Embedding(n_tokens, embed_dim)
            self.embed_loop = nn.Embedding(n_tokens, embed_dim)

            self.spatial_dropout = SpatialDropout1D(spatial_dropout)

            input_dim = embed_dim * 3 + 2  # 3 embeddings + 2 numerical

            RNNClass = nn.GRU if rnn_type == "gru" else nn.LSTM
            self.rnn = RNNClass(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0,
                bidirectional=True,
                batch_first=True,
            )

            self.output_layer = nn.Linear(hidden_dim * 2, n_targets)

        def forward(self, x, pred_len=None):
            # x: (batch, seq_len, 5) — 3 cat + 2 num
            cat_feats = x[:, :, :3].long()
            num_feats = x[:, :, 3:]

            e_seq = self.embed_seq(cat_feats[:, :, 0])
            e_struct = self.embed_struct(cat_feats[:, :, 1])
            e_loop = self.embed_loop(cat_feats[:, :, 2])

            embedded = torch.cat([e_seq, e_struct, e_loop, num_feats], dim=2)
            embedded = self.spatial_dropout(embedded)

            rnn_out, _ = self.rnn(embedded)

            if pred_len is not None:
                rnn_out = rnn_out[:, :pred_len, :]

            out = self.output_layer(rnn_out)
            return out

    return RNADegradationModel()


def _mcrmse_loss(y_pred, y_true, n_targets=5):
    """MCRMSE loss for PyTorch."""
    torch, _, _ = _import_torch()
    score = torch.tensor(0.0, device=y_pred.device)
    for i in range(n_targets):
        mse = torch.mean((y_pred[:, :, i] - y_true[:, :, i]) ** 2)
        score = score + torch.sqrt(mse + 1e-9) / n_targets
    return score


@contract(
    inputs={
        "train_features": {"format": "npz", "required": True},
    },
    outputs={
        "model": {"format": "pickle"},
        "metrics": {"format": "json"},
    },
    description="Train bidirectional GRU+LSTM ensemble for RNA degradation prediction",
    tags=["training", "deep-learning", "rna", "gru", "lstm"],
)
def train_rna_gru_lstm(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    embed_dim: int = 200,
    hidden_dim: int = 256,
    n_layers: int = 3,
    dropout: float = 0.4,
    spatial_dropout: float = 0.2,
    n_folds: int = 4,
    n_repeats: int = 1,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    lr_patience: int = 8,
    seed: int = 42,
    **params,
) -> str:
    """Train GRU+LSTM ensemble with k-fold cross-validation."""
    torch, nn, F = _import_torch()
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.cluster import KMeans

    # Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

    # Load data
    data = np.load(inputs["train_features"], allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    signal_to_noise = data["signal_to_noise"].astype(np.float32)
    sn_filter = data["SN_filter"].astype(np.float32)
    seq_scored = int(data["seq_scored"][0])
    seq_length = X.shape[1]

    n_samples = X.shape[0]
    print(f"  Train: {n_samples} samples, seq_len={seq_length}, scored={seq_scored}")

    # Cluster sequences for GroupKFold
    kmeans = KMeans(n_clusters=min(200, n_samples // 5), random_state=seed, n_init=10)
    kmeans.fit(X[:, :, 0])  # cluster by sequence token features
    cluster_ids = kmeans.labels_

    # Sample weights
    epsilon = 0.1
    sample_weights = np.log1p(signal_to_noise + epsilon) / 2

    # Training loop for both GRU and LSTM
    all_model_weights = {}
    all_cv_scores = {}

    for rnn_type in ["gru", "lstm"]:
        print(f"\n  === Training {rnn_type.upper()} ===")
        fold_weights = []
        fold_scores = []

        for repeat in range(n_repeats):
            gkf = GroupKFold(n_splits=n_folds)

            for fold_idx, (train_idx, val_idx) in enumerate(
                gkf.split(X, y, cluster_ids)
            ):
                fold_label = f"{rnn_type}_r{repeat}_f{fold_idx}"
                print(f"    {fold_label}: train={len(train_idx)}, val={len(val_idx)}")

                # Build model
                model = _build_rna_model(
                    rnn_type=rnn_type,
                    n_tokens=N_TOKENS,
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                    dropout=dropout,
                    spatial_dropout=spatial_dropout,
                )
                model = model.to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.1, patience=lr_patience,
                )

                # Data
                X_trn = torch.tensor(X[train_idx], dtype=torch.float32)
                y_trn = torch.tensor(y[train_idx], dtype=torch.float32)
                w_trn = torch.tensor(sample_weights[train_idx], dtype=torch.float32)

                val_sn_mask = sn_filter[val_idx] == 1.0
                X_val = torch.tensor(X[val_idx][val_sn_mask], dtype=torch.float32).to(device)
                y_val = torch.tensor(y[val_idx][val_sn_mask], dtype=torch.float32).to(device)

                best_val_loss = float("inf")
                best_state = None

                for epoch in range(epochs):
                    model.train()
                    # Shuffle
                    perm = torch.randperm(len(X_trn))
                    epoch_loss = 0.0
                    n_batches = 0

                    for start in range(0, len(X_trn), batch_size):
                        end = min(start + batch_size, len(X_trn))
                        idx = perm[start:end]

                        xb = X_trn[idx].to(device)
                        yb = y_trn[idx].to(device)
                        wb = w_trn[idx].to(device)

                        pred = model(xb, pred_len=seq_scored)

                        # Weighted MCRMSE loss per sample
                        per_sample_loss = torch.zeros(len(xb), device=device)
                        for t in range(5):
                            per_sample_loss += torch.sqrt(
                                torch.mean((pred[:, :, t] - yb[:, :, t]) ** 2, dim=1) + 1e-9
                            ) / 5
                        weighted_loss = (per_sample_loss * wb).mean()

                        optimizer.zero_grad()
                        weighted_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                        epoch_loss += weighted_loss.item()
                        n_batches += 1

                    # Validation
                    model.eval()
                    with torch.no_grad():
                        val_pred = model(X_val, pred_len=seq_scored)
                        val_loss = _mcrmse_loss(val_pred, y_val).item()

                    scheduler.step(val_loss)

                    avg_train_loss = epoch_loss / max(n_batches, 1)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                    if (epoch + 1) % 10 == 0 or epoch == 0:
                        print(f"      Epoch {epoch+1}/{epochs}: train={avg_train_loss:.4f}, val={val_loss:.4f} (best={best_val_loss:.4f})")

                    # Early stop if LR is very small
                    current_lr = optimizer.param_groups[0]["lr"]
                    if current_lr < 1e-6:
                        print(f"      Early stop at epoch {epoch+1} (lr={current_lr:.2e})")
                        break

                fold_weights.append(best_state)
                fold_scores.append(best_val_loss)
                print(f"    {fold_label}: best val MCRMSE = {best_val_loss:.5f}")

                del model
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        all_model_weights[rnn_type] = fold_weights
        mean_score = np.mean(fold_scores)
        all_cv_scores[rnn_type] = {
            "fold_scores": [float(s) for s in fold_scores],
            "mean": float(mean_score),
            "std": float(np.std(fold_scores)),
        }
        print(f"  {rnn_type.upper()} CV: {mean_score:.5f} +/- {np.std(fold_scores):.5f}")

    # Save model weights
    model_data = {
        "weights": all_model_weights,
        "config": {
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "dropout": dropout,
            "spatial_dropout": spatial_dropout,
            "n_tokens": N_TOKENS,
            "n_folds": n_folds,
            "n_repeats": n_repeats,
        },
    }
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(model_data, f)

    # Save metrics
    metrics = {
        "cv_scores": all_cv_scores,
        "overall_mean": float(np.mean(
            all_cv_scores["gru"]["fold_scores"] + all_cv_scores["lstm"]["fold_scores"]
        )),
    }
    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return f"train_rna_gru_lstm: GRU={all_cv_scores['gru']['mean']:.5f}, LSTM={all_cv_scores['lstm']['mean']:.5f}"


# =============================================================================
# SERVICE 3: PREDICT RNA ENSEMBLE
# =============================================================================

@contract(
    inputs={
        "model": {"format": "pickle", "required": True},
        "test_features": {"format": "npz", "required": True},
    },
    outputs={
        "predictions": {"format": "pickle"},
    },
    description="Predict RNA degradation using trained GRU+LSTM ensemble",
    tags=["prediction", "deep-learning", "rna", "ensemble"],
)
def predict_rna_ensemble(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    gru_weight: float = 0.5,
    lstm_weight: float = 0.5,
    embed_dim: int = 200,
    hidden_dim: int = 256,
    n_layers: int = 3,
    **params,
) -> str:
    """Predict on test sequences using trained GRU+LSTM ensemble."""
    torch, nn, F = _import_torch()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load model
    with open(inputs["model"], "rb") as f:
        model_data = pickle.load(f)

    config = model_data["config"]
    weights = model_data["weights"]

    # Load test features
    test_data = np.load(inputs["test_features"], allow_pickle=True)
    X_107 = test_data["X_107"].astype(np.float32)
    X_130 = test_data["X_130"].astype(np.float32)
    ids_107 = test_data["ids_107"]
    ids_130 = test_data["ids_130"]

    print(f"  Test 107: {X_107.shape}, Test 130: {X_130.shape}")

    def predict_with_models(X, rnn_type, fold_weights, pred_len, device):
        """Predict using all fold models and average."""
        all_preds = np.zeros((len(X), pred_len, 5), dtype=np.float32)
        n_models = len(fold_weights)

        for state_dict in fold_weights:
            model = _build_rna_model(
                rnn_type=rnn_type,
                n_tokens=config["n_tokens"],
                embed_dim=config["embed_dim"],
                hidden_dim=config["hidden_dim"],
                n_layers=config["n_layers"],
                dropout=0,  # No dropout at inference
                spatial_dropout=0,
            )
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()

            with torch.no_grad():
                # Predict in batches
                batch_preds = []
                bs = 128
                for start in range(0, len(X), bs):
                    end = min(start + bs, len(X))
                    xb = torch.tensor(X[start:end], dtype=torch.float32).to(device)
                    pred = model(xb, pred_len=pred_len)
                    batch_preds.append(pred.cpu().numpy())

                preds = np.concatenate(batch_preds, axis=0)
                all_preds += preds / n_models

            del model

        return all_preds

    # Predict for each RNN type and sequence length
    rnn_preds = {}
    for rnn_type in ["gru", "lstm"]:
        if rnn_type not in weights:
            continue
        fold_weights = weights[rnn_type]
        print(f"  Predicting {rnn_type.upper()} ({len(fold_weights)} models)...")

        preds_107 = predict_with_models(X_107, rnn_type, fold_weights, 107, device)
        preds_130 = predict_with_models(X_130, rnn_type, fold_weights, 130, device)
        rnn_preds[rnn_type] = {"107": preds_107, "130": preds_130}

    # Blend
    first_type = list(rnn_preds.keys())[0]
    final_107 = np.zeros_like(rnn_preds[first_type]["107"])
    final_130 = np.zeros_like(rnn_preds[first_type]["130"])

    blend_weights = {"gru": gru_weight, "lstm": lstm_weight}
    total_weight = sum(blend_weights[k] for k in rnn_preds)

    for rnn_type, preds in rnn_preds.items():
        w = blend_weights[rnn_type] / total_weight
        final_107 += preds["107"] * w
        final_130 += preds["130"] * w

    # Save predictions
    predictions = {
        "ids_107": ids_107,
        "preds_107": final_107,
        "ids_130": ids_130,
        "preds_130": final_130,
        "target_cols": TARGET_COLS,
    }
    os.makedirs(os.path.dirname(outputs["predictions"]) or ".", exist_ok=True)
    with open(outputs["predictions"], "wb") as f:
        pickle.dump(predictions, f)

    return f"predict_rna_ensemble: 107-len={final_107.shape[0]}, 130-len={final_130.shape[0]}"


# =============================================================================
# SERVICE 4: FORMAT RNA SUBMISSION
# =============================================================================

@contract(
    inputs={
        "predictions": {"format": "pickle", "required": True},
        "sample_submission": {"format": "csv", "required": True},
    },
    outputs={
        "submission": {"format": "csv"},
    },
    description="Format RNA degradation predictions into Kaggle submission CSV",
    tags=["submission", "formatting"],
)
def format_rna_submission(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_cols: Optional[List[str]] = None,
    **params,
) -> str:
    """Format predictions into submission CSV with id_seqpos format."""
    if target_cols is None:
        target_cols = TARGET_COLS

    # Load predictions
    with open(inputs["predictions"], "rb") as f:
        predictions = pickle.load(f)

    ids_107 = predictions["ids_107"]
    preds_107 = predictions["preds_107"]
    ids_130 = predictions["ids_130"]
    preds_130 = predictions["preds_130"]

    # Build submission rows
    rows = []
    for group_ids, group_preds in [(ids_107, preds_107), (ids_130, preds_130)]:
        pred_len = group_preds.shape[1]
        for i, uid in enumerate(group_ids):
            for pos in range(pred_len):
                row = {"id_seqpos": f"{uid}_{pos}"}
                for j, col in enumerate(target_cols):
                    row[col] = float(group_preds[i, pos, j])
                rows.append(row)

    preds_df = pd.DataFrame(rows)

    # Merge with sample submission to ensure correct order
    sample_sub = pd.read_csv(inputs["sample_submission"])
    submission = sample_sub[["id_seqpos"]].merge(preds_df, on="id_seqpos", how="left")

    # Fill any missing values with 0
    submission = submission.fillna(0.0)

    os.makedirs(os.path.dirname(outputs["submission"]) or ".", exist_ok=True)
    submission.to_csv(outputs["submission"], index=False)

    return f"format_rna_submission: {len(submission)} rows, {len(target_cols)} targets"


# =============================================================================
# SERVICE 5: PREDICT RNA WITH PRETRAINED RIBONANZANET
# =============================================================================

@contract(
    inputs={
        "test_data": {"format": "json", "required": True},
    },
    outputs={
        "predictions": {"format": "pickle"},
    },
    description="Predict RNA degradation using pretrained RibonanzaNet transformer",
    tags=["prediction", "deep-learning", "rna", "ribonanzanet", "pretrained"],
)
def predict_rna_ribonanzanet(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    model_dir: str = "stanford-covid-vaccine/datasets/ribonanzanet",
    config_file: str = "configs/pairwise.yaml",
    weights_file: str = "RibonanzaNet-Deg.pt",
    **params,
) -> str:
    """Predict RNA degradation using pretrained RibonanzaNet model.

    Uses the RibonanzaNet architecture (pretrained on Ribonanza competition data)
    with a fine-tuned linear decoder for 5 degradation targets.
    No training required — inference only with pretrained weights.
    """
    torch, nn, F = _import_torch()
    import yaml

    # Resolve model_dir: test_data is in .../datasets/test.json
    # so ribonanzanet code is in .../datasets/ribonanzanet/
    datasets_dir = os.path.dirname(inputs["test_data"])
    ribonanzanet_dir = os.path.join(datasets_dir, "ribonanzanet")

    # Add RibonanzaNet source to path for Network.py and dropout.py imports
    sys.path.insert(0, ribonanzanet_dir)
    from Network import RibonanzaNet

    # Load config
    config_path = os.path.join(ribonanzanet_dir, config_file)
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    class Config:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    config = Config(**config_dict)

    # Define fine-tuned model (linear decoder for 5 targets)
    class FinetunedRibonanzaNet(RibonanzaNet):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.decoder = nn.Linear(cfg.ninp, 5)

        def forward(self, src):
            seq_feats, _ = self.get_embeddings(
                src, torch.ones_like(src).long().to(src.device)
            )
            return self.decoder(seq_feats)

    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

    # Load model and weights
    model = FinetunedRibonanzaNet(config)
    weights_path = os.path.join(ribonanzanet_dir, weights_file)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Load test data
    test_data = pd.read_json(inputs["test_data"], lines=True)
    print(f"  Test data: {len(test_data)} sequences")

    # Tokenize sequences (ACGU → 0,1,2,3)
    token_map = {nt: i for i, nt in enumerate("ACGU")}

    # Separate by sequence length (107 vs 130)
    test_107 = test_data[test_data["seq_length"] == 107].reset_index(drop=True)
    test_130 = test_data[test_data["seq_length"] == 130].reset_index(drop=True)
    print(f"  Test 107: {len(test_107)}, Test 130: {len(test_130)}")

    # Run inference
    def predict_group(df):
        preds_list = []
        with torch.no_grad():
            for idx in range(len(df)):
                seq = df.loc[idx, "sequence"]
                tokens = [token_map[nt] for nt in seq]
                tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
                pred = model(tokens_tensor)  # (1, seq_len, 5)
                preds_list.append(pred.cpu().numpy()[0])
        return np.array(preds_list)

    preds_107 = predict_group(test_107) if len(test_107) > 0 else np.zeros((0, 107, 5))
    preds_130 = predict_group(test_130) if len(test_130) > 0 else np.zeros((0, 130, 5))

    ids_107 = test_107["id"].tolist() if len(test_107) > 0 else []
    ids_130 = test_130["id"].tolist() if len(test_130) > 0 else []

    print(f"  Predictions 107: {preds_107.shape}, 130: {preds_130.shape}")

    # Save in same format as predict_rna_ensemble
    predictions = {
        "ids_107": ids_107,
        "preds_107": preds_107,
        "ids_130": ids_130,
        "preds_130": preds_130,
    }

    os.makedirs(os.path.dirname(outputs["predictions"]) or ".", exist_ok=True)
    with open(outputs["predictions"], "wb") as f:
        pickle.dump(predictions, f)

    total = len(ids_107) + len(ids_130)
    return f"predict_rna_ribonanzanet: {total} sequences (107:{len(ids_107)}, 130:{len(ids_130)})"


# =============================================================================
# AE PIPELINE: MODEL ARCHITECTURE
# =============================================================================

def _build_ae_model_classes():
    """Build and return model classes for the AE pretraining pipeline.

    Architecture: Multi-scale CNN + BPP Attention + Transformer + BiLSTM + BiGRU
    Based on top OpenVaccine competition solutions.
    """
    torch, nn, F = _import_torch()

    class Conv1dBlock(nn.Module):
        def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, dropout=0.1):
            super().__init__()
            self.conv1 = nn.Conv1d(in_c, out_c, kernel_size, padding=padding, dilation=dilation)
            self.bn1 = nn.BatchNorm1d(out_c)
            self.conv2 = nn.Conv1d(out_c, out_c, kernel_size, padding=padding, dilation=dilation)
            self.bn2 = nn.BatchNorm1d(out_c)
            self.skip = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            res = self.skip(x)
            out = F.leaky_relu(self.dropout(self.bn1(self.conv1(x))))
            out = F.leaky_relu(self.dropout(self.bn2(self.conv2(out))))
            return out + res

    class Conv2dBlock(nn.Module):
        def __init__(self, in_c, out_c, kernel_size=3, padding=1, dropout=0.1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, padding=padding)
            self.bn1 = nn.BatchNorm2d(out_c)
            self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, padding=padding)
            self.bn2 = nn.BatchNorm2d(out_c)
            self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            res = self.skip(x)
            out = F.leaky_relu(self.dropout(self.bn1(self.conv1(x))))
            out = F.leaky_relu(self.dropout(self.bn2(self.conv2(out))))
            return out + res

    class SeqEncoder(nn.Module):
        """Multi-scale 1D CNN: kernels 3,6,15,30 → concat 128+64+32+32=256."""
        def __init__(self, in_c):
            super().__init__()
            self.conv1 = Conv1dBlock(in_c, 128, kernel_size=3, padding=1)
            self.conv2 = Conv1dBlock(128, 64, kernel_size=6, padding=5, dilation=2)
            self.conv3 = Conv1dBlock(64, 32, kernel_size=15, padding=7)
            self.conv4 = Conv1dBlock(32, 32, kernel_size=30, padding=29, dilation=2)

        def forward(self, x):
            o1 = self.conv1(x)
            o2 = self.conv2(o1)
            o3 = self.conv3(o2)
            o4 = self.conv4(o3)
            return torch.cat([o1, o2, o3, o4], dim=1)

    class BppAttn(nn.Module):
        """BPP-based attention: 2D conv on pairwise features, matmul with 1D."""
        def __init__(self, d=256):
            super().__init__()
            self.conv_1d = Conv1dBlock(d, 128)
            self.conv_2d = Conv2dBlock(5, 128)

        def forward(self, x, bpp):
            x = self.conv_1d(x)
            bpp = self.conv_2d(bpp)
            return torch.matmul(bpp, x.unsqueeze(-1)).squeeze(-1)

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=200):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer('pe', pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class RnnLayers(nn.Module):
        """Transformer (2L, 8H) → BiLSTM → BiGRU, all 512-dim."""
        def __init__(self, d_model):
            super().__init__()
            self.pos_enc = PositionalEncoding(d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=8, dim_feedforward=d_model * 4,
                batch_first=True, dropout=0.1
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=2)
            self.lstm = nn.LSTM(d_model, d_model // 2, batch_first=True, bidirectional=True)
            self.gru = nn.GRU(d_model, d_model // 2, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(0.3)

        def forward(self, x):
            x = self.pos_enc(x)
            x = self.transformer(x)
            x = self.drop(x)
            x, _ = self.lstm(x)
            x = self.drop(x)
            x, _ = self.gru(x)
            return x

    class BaseAttnModel(nn.Module):
        """Backbone: SeqEncoder(18→256) + BppAttn(256→128→256) → 512 → RNN → 512."""
        def __init__(self, d=256):
            super().__init__()
            self.linear_feat = nn.Linear(17, 1)
            self.seq_encoder = SeqEncoder(18)
            self.bpp_attn = BppAttn(d)
            self.bpp_seq_encoder = SeqEncoder(128)
            self.rnn = RnnLayers(d * 2)

        def forward(self, node_feat, bpp_feat):
            bpp_mat = bpp_feat[:, 0]
            bpp_max = bpp_mat.max(dim=-1).values.unsqueeze(-1)
            bpp_sum = bpp_mat.sum(dim=-1).unsqueeze(-1)
            bpp_nb = ((bpp_mat > 0).float().sum(dim=-1) / bpp_mat.shape[-1]).unsqueeze(-1)
            bpp_nb = (bpp_nb - 0.077522) / 0.08914
            x17 = torch.cat([node_feat, bpp_max, bpp_sum, bpp_nb], dim=-1)
            learned = self.linear_feat(x17)
            x18 = torch.cat([x17, learned], dim=-1)
            x = x18.permute(0, 2, 1)
            seq_feat = self.seq_encoder(x)
            bpp_out = self.bpp_attn(seq_feat, bpp_feat)
            bpp_out = self.bpp_seq_encoder(bpp_out)
            combined = torch.cat([seq_feat, bpp_out], dim=1).permute(0, 2, 1)
            return self.rnn(combined)

    class AEModel(nn.Module):
        """Denoising autoencoder: corrupt input → backbone → reconstruct 14-dim one-hot."""
        def __init__(self):
            super().__init__()
            self.seq = BaseAttnModel()
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(512, 14)

        def forward(self, node_feat, bpp_feat):
            corrupted = F.dropout2d(
                node_feat.permute(0, 2, 1), p=0.3, training=self.training
            ).permute(0, 2, 1)
            out = self.seq(corrupted, bpp_feat)
            return self.fc(self.drop(out))  # raw logits (use BCE with logits)

    class FromAeModel(nn.Module):
        """Supervised model: backbone → Linear(512, 5) for degradation targets."""
        def __init__(self):
            super().__init__()
            self.seq = BaseAttnModel()
            self.fc = nn.Linear(512, 5)

        def forward(self, node_feat, bpp_feat, pred_len=None):
            out = self.seq(node_feat, bpp_feat)
            out = self.fc(out)
            if pred_len is not None:
                out = out[:, :pred_len]
            return out

    return {'AEModel': AEModel, 'FromAeModel': FromAeModel}


# =============================================================================
# SERVICE 6: PREPARE RNA AE FEATURES
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "json", "required": True},
        "test_data": {"format": "json", "required": True},
    },
    outputs={
        "ae_features": {"format": "pickle"},
    },
    description="Prepare one-hot features and metadata for AE pretraining pipeline",
    tags=["preprocessing", "rna", "ae", "feature-engineering"],
)
def prepare_rna_ae_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    bpps_dir: str = "bpps",
    **params,
) -> str:
    """Load RNA JSON data, one-hot encode, save features for AE pipeline."""
    train_df = pd.read_json(inputs["train_data"], lines=True)
    test_df = pd.read_json(inputs["test_data"], lines=True)

    datasets_dir = os.path.dirname(inputs["train_data"])
    abs_bpps_dir = os.path.join(datasets_dir, bpps_dir) if not os.path.isabs(bpps_dir) else bpps_dir

    def encode_samples(df):
        return [
            _one_hot_encode(row["sequence"], row["structure"], row["predicted_loop_type"])
            for _, row in df.iterrows()
        ]

    print(f"  Encoding train ({len(train_df)} samples)...")
    train_node_features = np.array(encode_samples(train_df), dtype=np.float32)

    seq_scored = int(train_df["seq_scored"].iloc[0])
    train_labels = np.zeros((len(train_df), seq_scored, 5), dtype=np.float32)
    for i, (_, row) in enumerate(train_df.iterrows()):
        for j, col in enumerate(TARGET_COLS):
            train_labels[i, :, j] = row[col][:seq_scored]

    test_107 = test_df[test_df["seq_length"] == 107].reset_index(drop=True)
    test_130 = test_df[test_df["seq_length"] == 130].reset_index(drop=True)

    print(f"  Encoding test_107 ({len(test_107)}), test_130 ({len(test_130)})...")
    test_107_features = np.array(encode_samples(test_107), dtype=np.float32)
    test_130_features = np.array(encode_samples(test_130), dtype=np.float32)

    data = {
        'train_node_features': train_node_features,
        'train_labels': train_labels,
        'train_ids': train_df["id"].tolist(),
        'train_structures': train_df["structure"].tolist(),
        'train_sn': train_df["signal_to_noise"].values.astype(np.float32),
        'train_snf': train_df["SN_filter"].values.astype(np.float32),
        'test_107_node_features': test_107_features,
        'test_107_ids': test_107["id"].tolist(),
        'test_107_structures': test_107["structure"].tolist(),
        'test_130_node_features': test_130_features,
        'test_130_ids': test_130["id"].tolist(),
        'test_130_structures': test_130["structure"].tolist(),
        'bpps_dir': abs_bpps_dir,
    }

    os.makedirs(os.path.dirname(outputs["ae_features"]) or ".", exist_ok=True)
    with open(outputs["ae_features"], "wb") as f:
        pickle.dump(data, f)

    print(f"  Train: {train_node_features.shape}, Labels: {train_labels.shape}")
    print(f"  Test 107: {test_107_features.shape}, Test 130: {test_130_features.shape}")
    return f"prepare_rna_ae_features: train={len(train_df)}, test_107={len(test_107)}, test_130={len(test_130)}"


# =============================================================================
# SERVICE 7: PRETRAIN RNA AUTOENCODER
# =============================================================================

@contract(
    inputs={
        "ae_features": {"format": "pickle", "required": True},
    },
    outputs={
        "ae_backbone": {"format": "pickle"},
    },
    description="Self-supervised denoising autoencoder pretraining on all RNA data",
    tags=["pretraining", "autoencoder", "rna", "self-supervised"],
)
def pretrain_rna_autoencoder(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    epochs_per_round: int = 5,
    n_rounds: int = 4,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    seed: int = 42,
    **params,
) -> str:
    """Train denoising autoencoder on all RNA sequences (train + test).

    The AE corrupts input one-hot features with dropout and learns to
    reconstruct them. This provides a pretrained backbone for supervised
    fine-tuning. Uses all data including test (no labels needed).
    """
    torch, nn, F = _import_torch()
    from torch.utils.data import Dataset, DataLoader

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = _get_device()
    print(f"  Device: {device}")

    with open(inputs["ae_features"], "rb") as f:
        data = pickle.load(f)

    bpps_dir = data["bpps_dir"]
    models_cls = _build_ae_model_classes()

    class RNADataset(Dataset):
        def __init__(self, node_features, ids, structures):
            self.node_features = node_features
            self.ids = ids
            self.structures = structures
            self._bpp_cache = {}

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            nf = self.node_features[idx]
            sid = self.ids[idx]
            struct = self.structures[idx]
            seq_len = nf.shape[0]
            if sid not in self._bpp_cache:
                self._bpp_cache[sid] = _load_sample_bpp_features(sid, bpps_dir, struct, seq_len)
            bpp = self._bpp_cache[sid]
            return torch.tensor(nf, dtype=torch.float32), torch.tensor(bpp, dtype=torch.float32)

    train_ds = RNADataset(data['train_node_features'], data['train_ids'], data['train_structures'])
    test_107_ds = RNADataset(data['test_107_node_features'], data['test_107_ids'], data['test_107_structures'])
    test_130_ds = RNADataset(data['test_130_node_features'], data['test_130_ids'], data['test_130_structures'])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_107_loader = DataLoader(test_107_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_130_loader = DataLoader(test_130_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = models_cls['AEModel']().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"  AE Pretraining: {n_rounds} rounds x {epochs_per_round} epochs x 3 datasets")

    best_backbone_state = None

    def _sanitize_gradients(model):
        for p in model.parameters():
            if p.grad is not None:
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=1.0, neginf=-1.0)

    def train_epoch(loader):
        model.train()
        total_loss, n = 0.0, 0
        for nf, bpp in loader:
            nf, bpp = nf.to(device), bpp.to(device)
            pred = model(nf, bpp)
            loss = F.binary_cross_entropy_with_logits(pred, nf)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            _sanitize_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item() * nf.size(0)
            n += nf.size(0)
        return total_loss / max(n, 1)

    last_losses = {}
    nan_detected = False
    total_epochs = n_rounds * epochs_per_round
    for epoch_idx in range(total_epochs):
        # Interleave: train on all three datasets each epoch
        last_losses['train'] = train_epoch(train_loader)
        last_losses['test_107'] = train_epoch(test_107_loader)
        last_losses['test_130'] = train_epoch(test_130_loader)

        if np.isnan(last_losses['train']):
            print(f"    Epoch {epoch_idx+1}: NaN detected, stopping AE pretrain")
            nan_detected = True
            break

        # Save backbone checkpoint
        best_backbone_state = {k: v.cpu().clone() for k, v in model.seq.state_dict().items()}

        if (epoch_idx + 1) % 5 == 0 or epoch_idx == 0:
            print(f"    Epoch {epoch_idx+1}/{total_epochs}: train={last_losses['train']:.4f}, "
                  f"t107={last_losses['test_107']:.4f}, t130={last_losses['test_130']:.4f}")

    # Use best saved backbone or current one
    if nan_detected and best_backbone_state is not None:
        backbone_state = best_backbone_state
        print(f"  Using best saved backbone (before NaN)")
    else:
        backbone_state = {k: v.cpu().clone() for k, v in model.seq.state_dict().items()}
    os.makedirs(os.path.dirname(outputs["ae_backbone"]) or ".", exist_ok=True)
    with open(outputs["ae_backbone"], "wb") as f:
        pickle.dump(backbone_state, f)

    total_epochs = n_rounds * epochs_per_round * 3
    print(f"  Saved AE backbone ({len(backbone_state)} tensors, {total_epochs} total epochs)")
    return f"pretrain_rna_autoencoder: {total_epochs} epochs, train_bce={last_losses.get('train', float('nan')):.4f}"


# =============================================================================
# SERVICE 8: TRAIN RNA AE MODEL (SUPERVISED FINE-TUNING)
# =============================================================================

@contract(
    inputs={
        "ae_features": {"format": "pickle", "required": True},
        "ae_backbone": {"format": "pickle", "required": True},
    },
    outputs={
        "fold_models": {"format": "pickle"},
        "metrics": {"format": "json"},
    },
    description="Supervised fine-tuning of AE backbone with k-fold CV for RNA degradation",
    tags=["training", "deep-learning", "rna", "transformer", "fine-tuning"],
)
def train_rna_ae_model(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    n_folds: int = 5,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    seed: int = 42,
    **params,
) -> str:
    """Fine-tune pretrained AE backbone for RNA degradation prediction.

    Loads pretrained backbone, adds linear head for 5 targets,
    trains with SNR-weighted MCRMSE loss using k-fold CV.
    """
    torch, nn, F = _import_torch()
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import ShuffleSplit

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = _get_device()
    print(f"  Device: {device}")

    with open(inputs["ae_features"], "rb") as f:
        data = pickle.load(f)
    with open(inputs["ae_backbone"], "rb") as f:
        backbone_state = pickle.load(f)

    bpps_dir = data["bpps_dir"]
    node_features = data["train_node_features"]
    labels = data["train_labels"]
    sn = data["train_sn"]
    snf = data["train_snf"]
    ids = data["train_ids"]
    structures = data["train_structures"]
    seq_scored = labels.shape[1]

    models_cls = _build_ae_model_classes()

    class TrainDataset(Dataset):
        def __init__(self, indices):
            self.indices = indices
            self._bpp_cache = {}

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            idx = self.indices[i]
            nf = node_features[idx]
            lab = labels[idx]
            sn_w = np.float32(0.5 * np.log(sn[idx] + 1.01))
            sid = ids[idx]
            struct = structures[idx]
            seq_len = nf.shape[0]
            if sid not in self._bpp_cache:
                self._bpp_cache[sid] = _load_sample_bpp_features(sid, bpps_dir, struct, seq_len)
            bpp = self._bpp_cache[sid]
            return (
                torch.tensor(nf, dtype=torch.float32),
                torch.tensor(bpp, dtype=torch.float32),
                torch.tensor(lab, dtype=torch.float32),
                torch.tensor(sn_w, dtype=torch.float32),
            )

    splitter = ShuffleSplit(n_splits=n_folds, test_size=0.1, random_state=seed)
    all_indices = np.arange(len(ids))

    fold_weights = []
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(all_indices)):
        print(f"\n  === Fold {fold_idx+1}/{n_folds} (train={len(train_idx)}, val={len(val_idx)}) ===")

        model = models_cls['FromAeModel']()
        model.seq.load_state_dict(backbone_state)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_ds = TrainDataset(train_idx)
        val_ds = TrainDataset(val_idx)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            # Training
            model.train()
            epoch_loss, n_samples = 0.0, 0
            for nf, bpp, lab, sn_w in train_loader:
                nf, bpp, lab, sn_w = nf.to(device), bpp.to(device), lab.to(device), sn_w.to(device)
                pred = model(nf, bpp, pred_len=seq_scored)
                per_col = torch.zeros(pred.size(0), device=device)
                for t in range(5):
                    per_col += torch.sqrt(((pred[:, :, t] - lab[:, :, t]) ** 2).mean(dim=1) + 1e-9) / 5
                loss = (per_col * sn_w).mean()
                optimizer.zero_grad()
                loss.backward()
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=1.0, neginf=-1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                epoch_loss += loss.item() * nf.size(0)
                n_samples += nf.size(0)

            # Validation
            model.eval()
            val_preds, val_labels_list = [], []
            with torch.no_grad():
                for nf, bpp, lab, _ in val_loader:
                    nf, bpp = nf.to(device), bpp.to(device)
                    pred = model(nf, bpp, pred_len=seq_scored)
                    val_preds.append(pred.cpu().numpy())
                    val_labels_list.append(lab.numpy())

            vp = np.concatenate(val_preds, axis=0)
            vl = np.concatenate(val_labels_list, axis=0)
            val_snf_mask = snf[val_idx] == 1.0
            if val_snf_mask.any():
                vp_clean, vl_clean = vp[val_snf_mask], vl[val_snf_mask]
            else:
                vp_clean, vl_clean = vp, vl
            mcrmse = np.mean([np.sqrt(np.mean((vp_clean[:, :, t] - vl_clean[:, :, t]) ** 2)) for t in range(5)])

            if mcrmse < best_val_loss:
                best_val_loss = mcrmse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_train = epoch_loss / max(n_samples, 1)
                print(f"    Epoch {epoch+1}/{epochs}: train={avg_train:.4f}, val_mcrmse={mcrmse:.5f} (best={best_val_loss:.5f})")

        fold_weights.append(best_state)
        fold_scores.append(best_val_loss)
        print(f"  Fold {fold_idx+1}: best val MCRMSE = {best_val_loss:.5f}")

        del model
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    os.makedirs(os.path.dirname(outputs["fold_models"]) or ".", exist_ok=True)
    with open(outputs["fold_models"], "wb") as f:
        pickle.dump(fold_weights, f)

    metrics = {
        "fold_scores": [float(s) for s in fold_scores],
        "mean_mcrmse": float(np.mean(fold_scores)),
        "std_mcrmse": float(np.std(fold_scores)),
    }
    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Mean CV MCRMSE: {metrics['mean_mcrmse']:.5f} +/- {metrics['std_mcrmse']:.5f}")
    return f"train_rna_ae_model: mean_mcrmse={metrics['mean_mcrmse']:.5f}"


# =============================================================================
# SERVICE 9: PREDICT RNA AE ENSEMBLE
# =============================================================================

@contract(
    inputs={
        "fold_models": {"format": "pickle", "required": True},
        "ae_features": {"format": "pickle", "required": True},
    },
    outputs={
        "predictions": {"format": "pickle"},
    },
    description="Predict RNA degradation using trained AE model ensemble",
    tags=["prediction", "deep-learning", "rna", "ensemble", "transformer"],
)
def predict_rna_ae_ensemble(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    **params,
) -> str:
    """Predict on test sequences using ensemble of trained AE models."""
    torch, nn, F = _import_torch()

    device = _get_device()
    print(f"  Device: {device}")

    with open(inputs["fold_models"], "rb") as f:
        fold_weights = pickle.load(f)
    with open(inputs["ae_features"], "rb") as f:
        data = pickle.load(f)

    bpps_dir = data["bpps_dir"]
    models_cls = _build_ae_model_classes()

    def predict_group(node_features, ids, structures, pred_len):
        if len(ids) == 0:
            return np.zeros((0, pred_len, 5), dtype=np.float32)
        all_preds = np.zeros((len(ids), pred_len, 5), dtype=np.float32)
        n_folds = len(fold_weights)

        for fold_i, state_dict in enumerate(fold_weights):
            model = models_cls['FromAeModel']()
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            bs = 16
            with torch.no_grad():
                for start in range(0, len(ids), bs):
                    end = min(start + bs, len(ids))
                    nf_batch = torch.tensor(node_features[start:end], dtype=torch.float32).to(device)
                    bpp_list = [
                        _load_sample_bpp_features(ids[i], bpps_dir, structures[i], node_features[i].shape[0])
                        for i in range(start, end)
                    ]
                    bpp_batch = torch.tensor(np.array(bpp_list), dtype=torch.float32).to(device)
                    pred = model(nf_batch, bpp_batch, pred_len=pred_len)
                    all_preds[start:end] += pred.cpu().numpy() / n_folds
            del model
            print(f"    Fold {fold_i+1}/{n_folds} done")

        return all_preds

    print(f"  Predicting test_107 ({len(data['test_107_ids'])} samples)...")
    preds_107 = predict_group(
        data['test_107_node_features'], data['test_107_ids'],
        data['test_107_structures'], 107
    )
    print(f"  Predicting test_130 ({len(data['test_130_ids'])} samples)...")
    preds_130 = predict_group(
        data['test_130_node_features'], data['test_130_ids'],
        data['test_130_structures'], 130
    )

    predictions = {
        "ids_107": data['test_107_ids'],
        "preds_107": preds_107,
        "ids_130": data['test_130_ids'],
        "preds_130": preds_130,
    }
    os.makedirs(os.path.dirname(outputs["predictions"]) or ".", exist_ok=True)
    with open(outputs["predictions"], "wb") as f:
        pickle.dump(predictions, f)

    return f"predict_rna_ae_ensemble: 107={len(preds_107)}, 130={len(preds_130)}"


# =============================================================================
# PIPELINE SPECIFICATION
# =============================================================================

PIPELINE_SPEC = [
    {
        "service": "prepare_rna_ae_features",
        "inputs": {
            "train_data": "stanford-covid-vaccine/datasets/train.json",
            "test_data": "stanford-covid-vaccine/datasets/test.json",
        },
        "outputs": {
            "ae_features": "stanford-covid-vaccine/artifacts/ae_features.pkl",
        },
        "params": {
            "bpps_dir": "bpps",
        },
        "module": "stanford_covid_vaccine_services",
    },
    {
        "service": "pretrain_rna_autoencoder",
        "inputs": {
            "ae_features": "stanford-covid-vaccine/artifacts/ae_features.pkl",
        },
        "outputs": {
            "ae_backbone": "stanford-covid-vaccine/artifacts/ae_backbone.pkl",
        },
        "params": {
            "epochs_per_round": 5,
            "n_rounds": 4,
            "batch_size": 32,
        },
        "module": "stanford_covid_vaccine_services",
    },
    {
        "service": "train_rna_ae_model",
        "inputs": {
            "ae_features": "stanford-covid-vaccine/artifacts/ae_features.pkl",
            "ae_backbone": "stanford-covid-vaccine/artifacts/ae_backbone.pkl",
        },
        "outputs": {
            "fold_models": "stanford-covid-vaccine/artifacts/fold_models.pkl",
            "metrics": "stanford-covid-vaccine/artifacts/ae_metrics.json",
        },
        "params": {
            "n_folds": 5,
            "epochs": 100,
            "batch_size": 32,
        },
        "module": "stanford_covid_vaccine_services",
    },
    {
        "service": "predict_rna_ae_ensemble",
        "inputs": {
            "fold_models": "stanford-covid-vaccine/artifacts/fold_models.pkl",
            "ae_features": "stanford-covid-vaccine/artifacts/ae_features.pkl",
        },
        "outputs": {
            "predictions": "stanford-covid-vaccine/artifacts/predictions.pkl",
        },
        "params": {},
        "module": "stanford_covid_vaccine_services",
    },
    {
        "service": "format_rna_submission",
        "inputs": {
            "predictions": "stanford-covid-vaccine/artifacts/predictions.pkl",
            "sample_submission": "stanford-covid-vaccine/datasets/sample_submission.csv",
        },
        "outputs": {
            "submission": "stanford-covid-vaccine/submission.csv",
        },
        "params": {
            "target_cols": ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"],
        },
        "module": "stanford_covid_vaccine_services",
    },
]


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "format_rna_submission": format_rna_submission,
    "prepare_rna_ae_features": prepare_rna_ae_features,
    "pretrain_rna_autoencoder": pretrain_rna_autoencoder,
    "train_rna_ae_model": train_rna_ae_model,
    "predict_rna_ae_ensemble": predict_rna_ae_ensemble,
}


# =============================================================================
# PIPELINE RUNNER
# =============================================================================

def run_pipeline(base_path: str, verbose: bool = True) -> Dict[str, Any]:
    """Run the full Stanford COVID Vaccine pipeline."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline_runner import PipelineRunner

    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kb.sqlite"
    )
    runner = PipelineRunner(
        db_path=db_path,
        verbose=verbose,
        storage=base_path,
        modules=["stanford_covid_vaccine_services"],
    )
    return runner.run(PIPELINE_SPEC, base_path=base_path, pipeline_name="stanford-covid-vaccine")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stanford COVID Vaccine Pipeline")
    parser.add_argument("--base-path", default="storage")
    args = parser.parse_args()

    print(f"\nRunning Stanford COVID Vaccine Pipeline from {args.base_path}\n")
    result = run_pipeline(args.base_path)
    print(json.dumps(result, indent=2))
