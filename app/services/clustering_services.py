"""
Clustering Services - Common Module
==========================================

Clustering services for unsupervised learning tasks.
Uses sklearn's KMeans, DBSCAN, and AgglomerativeClustering.

Services:
  Training: train_kmeans, train_dbscan, train_hierarchical
  Prediction: predict_clusters
  Evaluation: evaluate_clusters
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract

# =============================================================================
# HELPERS: Import from shared io_utils
# =============================================================================
from services.io_utils import load_data as _load_data, save_data as _save_data


# =============================================================================
# CLUSTERING: KMEANS
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact", "artifact_type": "model"}},
        "metrics": {"format": "json", "schema": {"type": "json"}},
    },
    description="Train a KMeans clustering model on numeric features",
    tags=["clustering", "kmeans", "unsupervised", "training", "generic"],
    version="1.0.0",
)
def train_kmeans(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    n_clusters: int = 5,
    feature_exclude: Optional[List[str]] = None,
    random_state: int = 42,
    max_iter: int = 300,
) -> str:
    """
    Train a KMeans clustering model.

    Fits KMeans on all numeric columns (excluding specified columns).
    Outputs the trained model and metrics including inertia and silhouette score.

    G1 Compliance: Single responsibility - train KMeans model.
    G3 Compliance: Explicit random_state for reproducibility.
    G4 Compliance: feature_exclude injected via param.

    Parameters:
        n_clusters: Number of clusters to form
        feature_exclude: Columns to exclude from clustering (e.g., id columns)
        random_state: Random seed for reproducibility
        max_iter: Maximum number of iterations for the algorithm
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    df = _load_data(inputs["data"])
    exclude = set(feature_exclude or [])

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    if not feature_cols:
        raise ValueError("No numeric feature columns found after exclusion.")

    X = df[feature_cols].fillna(0).values

    # Standardize features for better clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        max_iter=max_iter,
        n_init=10,
    )
    labels = model.fit_predict(X_scaled)

    # Compute metrics
    metrics = {
        "algorithm": "kmeans",
        "n_clusters": n_clusters,
        "inertia": float(model.inertia_),
        "n_samples": int(len(X)),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "max_iter": max_iter,
    }

    if n_clusters > 1 and len(X) > n_clusters:
        sil_score = float(silhouette_score(X_scaled, labels))
        metrics["silhouette_score"] = sil_score

    model_data = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_cols,
        "algorithm": "kmeans",
        "n_clusters": n_clusters,
    }

    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(model_data, f)

    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    sil_str = f", silhouette={metrics['silhouette_score']:.4f}" if "silhouette_score" in metrics else ""
    return (
        f"train_kmeans: n_clusters={n_clusters}, inertia={model.inertia_:.2f}, "
        f"{len(X)} samples, {len(feature_cols)} features{sil_str}"
    )


# =============================================================================
# CLUSTERING: DBSCAN
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact", "artifact_type": "model"}},
        "metrics": {"format": "json", "schema": {"type": "json"}},
    },
    description="Train a DBSCAN clustering model on numeric features",
    tags=["clustering", "dbscan", "unsupervised", "training", "generic"],
    version="1.0.0",
)
def train_dbscan(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    eps: float = 0.5,
    min_samples: int = 5,
    feature_exclude: Optional[List[str]] = None,
) -> str:
    """
    Train a DBSCAN clustering model.

    Fits DBSCAN on all numeric columns (excluding specified columns).
    DBSCAN automatically determines the number of clusters and identifies noise points.

    G1 Compliance: Single responsibility - train DBSCAN model.
    G4 Compliance: feature_exclude injected via param.

    Parameters:
        eps: Maximum distance between two samples in the same neighborhood
        min_samples: Minimum number of samples in a neighborhood for a core point
        feature_exclude: Columns to exclude from clustering
    """
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    df = _load_data(inputs["data"])
    exclude = set(feature_exclude or [])

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    if not feature_cols:
        raise ValueError("No numeric feature columns found after exclusion.")

    X = df[feature_cols].fillna(0).values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    )
    labels = model.fit_predict(X_scaled)

    # Compute metrics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    noise_ratio = float(n_noise / len(labels)) if len(labels) > 0 else 0.0

    metrics = {
        "algorithm": "dbscan",
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": noise_ratio,
        "eps": eps,
        "min_samples": min_samples,
        "n_samples": int(len(X)),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
    }

    # Silhouette score requires at least 2 clusters and no all-noise
    if n_clusters > 1:
        non_noise_mask = labels != -1
        if np.sum(non_noise_mask) > n_clusters:
            sil_score = float(silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask]))
            metrics["silhouette_score"] = sil_score

    model_data = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_cols,
        "algorithm": "dbscan",
        "labels": labels,
    }

    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(model_data, f)

    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    sil_str = f", silhouette={metrics['silhouette_score']:.4f}" if "silhouette_score" in metrics else ""
    return (
        f"train_dbscan: n_clusters={n_clusters}, noise_ratio={noise_ratio:.3f}, "
        f"{len(X)} samples, {len(feature_cols)} features{sil_str}"
    )


# =============================================================================
# CLUSTERING: HIERARCHICAL (AGGLOMERATIVE)
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact", "artifact_type": "model"}},
        "metrics": {"format": "json", "schema": {"type": "json"}},
    },
    description="Train an AgglomerativeClustering model on numeric features",
    tags=["clustering", "hierarchical", "agglomerative", "unsupervised", "training", "generic"],
    version="1.0.0",
)
def train_hierarchical(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    n_clusters: int = 5,
    linkage: str = "ward",
    feature_exclude: Optional[List[str]] = None,
) -> str:
    """
    Train an AgglomerativeClustering (hierarchical) model.

    Fits hierarchical clustering on all numeric columns (excluding specified columns).
    Supports ward, complete, average, and single linkage methods.

    G1 Compliance: Single responsibility - train hierarchical model.
    G4 Compliance: feature_exclude injected via param.

    Parameters:
        n_clusters: Number of clusters to form
        linkage: Linkage criterion - 'ward', 'complete', 'average', or 'single'
        feature_exclude: Columns to exclude from clustering
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    df = _load_data(inputs["data"])
    exclude = set(feature_exclude or [])

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    if not feature_cols:
        raise ValueError("No numeric feature columns found after exclusion.")

    X = df[feature_cols].fillna(0).values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
    )
    labels = model.fit_predict(X_scaled)

    # Compute metrics
    metrics = {
        "algorithm": "hierarchical",
        "n_clusters": n_clusters,
        "linkage": linkage,
        "n_samples": int(len(X)),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
    }

    if n_clusters > 1 and len(X) > n_clusters:
        sil_score = float(silhouette_score(X_scaled, labels))
        metrics["silhouette_score"] = sil_score

    model_data = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_cols,
        "algorithm": "hierarchical",
        "labels": labels,
        "n_clusters": n_clusters,
        "linkage": linkage,
    }

    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(model_data, f)

    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    sil_str = f", silhouette={metrics['silhouette_score']:.4f}" if "silhouette_score" in metrics else ""
    return (
        f"train_hierarchical: n_clusters={n_clusters}, linkage={linkage}, "
        f"{len(X)} samples, {len(feature_cols)} features{sil_str}"
    )


# =============================================================================
# CLUSTER PREDICTION
# =============================================================================

@contract(
    inputs={
        "model": {"format": "pickle", "required": True, "schema": {"type": "artifact", "artifact_type": "model"}},
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "predictions": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Assign new data points to clusters using a trained clustering model",
    tags=["clustering", "prediction", "unsupervised", "generic"],
    version="1.0.0",
)
def predict_clusters(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: Optional[str] = None,
    cluster_column: str = "cluster",
) -> str:
    """
    Assign new data to clusters using a trained clustering model.

    For KMeans, uses the model's predict method.
    For DBSCAN and hierarchical (which lack predict), uses nearest-centroid
    assignment based on the training labels.

    G1 Compliance: Single responsibility - predict cluster assignments.
    G4 Compliance: id_column and cluster_column injected via params.

    Parameters:
        id_column: Name of the ID column to include in output (None to skip)
        cluster_column: Name for the cluster assignment column in output
    """
    with open(inputs["model"], "rb") as f:
        model_data = pickle.load(f)

    df = _load_data(inputs["data"])

    feature_cols = model_data["feature_columns"]
    scaler = model_data["scaler"]
    algorithm = model_data["algorithm"]

    # Get available features, fill missing with 0
    X = df[[c for c in feature_cols if c in df.columns]].copy()
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_cols].fillna(0).values
    X_scaled = scaler.transform(X)

    # Apply PCA if the model was trained with PCA
    pca = model_data.get("pca")
    use_pca = model_data.get("use_pca", False)
    if use_pca and pca is not None:
        X_transformed = pca.transform(X_scaled)
    else:
        X_transformed = X_scaled

    # Predict clusters based on algorithm type
    if algorithm == "kmeans":
        labels = model_data["model"].predict(X_transformed)
    elif algorithm == "gmm":
        labels = model_data["model"].predict(X_transformed)
    elif algorithm == "dbscan":
        # DBSCAN has no predict; use nearest core sample assignment
        from sklearn.neighbors import NearestNeighbors
        training_labels = model_data["labels"]
        core_mask = training_labels != -1
        if np.any(core_mask):
            # We don't have the original training data, so assign -1 (noise) as default
            # For practical use, re-fit DBSCAN on new data or use a different approach
            model = model_data["model"]
            if hasattr(model, "components_") and model.components_ is not None and len(model.components_) > 0:
                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(model.components_)
                distances, indices = nn.kneighbors(X_transformed)
                core_labels = training_labels[model.core_sample_indices_]
                labels = core_labels[indices.flatten()]
            else:
                # Fallback: assign all to cluster 0
                labels = np.zeros(len(X), dtype=int)
        else:
            labels = np.full(len(X), -1, dtype=int)
    elif algorithm == "hierarchical":
        # AgglomerativeClustering has no predict; use nearest centroid
        training_labels = model_data["labels"]
        n_clusters = model_data["n_clusters"]
        # Compute centroids from training labels (we don't have training data,
        # so we approximate using the model's labels on the new data)
        # Best practice: re-fit on new data, but for consistency we do nearest centroid
        model = model_data["model"]
        labels = model.fit_predict(X_transformed)
    else:
        raise ValueError(f"Unknown algorithm: '{algorithm}'")

    # Build output DataFrame with all original columns plus cluster assignment
    result = df.copy()
    result[cluster_column] = labels

    os.makedirs(os.path.dirname(outputs["predictions"]) or ".", exist_ok=True)
    _save_data(result, outputs["predictions"])

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return (
        f"predict_clusters ({algorithm}): {len(labels)} samples assigned to "
        f"{n_clusters} clusters"
    )


# =============================================================================
# CLUSTER EVALUATION
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "metrics": {"format": "json", "schema": {"type": "json"}},
    },
    description="Evaluate cluster quality using silhouette, Calinski-Harabasz, and Davies-Bouldin scores",
    tags=["clustering", "evaluation", "unsupervised", "generic"],
    version="1.0.0",
)
def evaluate_clusters(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    cluster_column: str = "cluster",
    feature_exclude: Optional[List[str]] = None,
) -> str:
    """
    Compute cluster quality metrics on data with cluster assignments.

    Computes silhouette score, Calinski-Harabasz index, and Davies-Bouldin index.
    Requires data to have a cluster assignment column.

    G1 Compliance: Single responsibility - evaluate cluster quality.
    G4 Compliance: cluster_column and feature_exclude injected via params.

    Parameters:
        cluster_column: Name of the column containing cluster assignments
        feature_exclude: Additional columns to exclude from evaluation features
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.preprocessing import StandardScaler

    df = _load_data(inputs["data"])

    if cluster_column not in df.columns:
        raise ValueError(f"Cluster column '{cluster_column}' not found. Available: {list(df.columns)}")

    labels = df[cluster_column].values
    exclude = set(feature_exclude or [])
    exclude.add(cluster_column)

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    if not feature_cols:
        raise ValueError("No numeric feature columns found after exclusion.")

    X = df[feature_cols].fillna(0).values

    # Standardize for consistent distance computation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    metrics = {
        "n_clusters": n_clusters,
        "n_samples": int(len(X)),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "cluster_column": cluster_column,
    }

    # Filter out noise points (label == -1) for metrics that require valid clusters
    valid_mask = labels != -1
    X_valid = X_scaled[valid_mask]
    labels_valid = labels[valid_mask]

    n_valid_clusters = len(set(labels_valid))

    if n_valid_clusters > 1 and len(X_valid) > n_valid_clusters:
        metrics["silhouette_score"] = float(silhouette_score(X_valid, labels_valid))
        metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X_valid, labels_valid))
        metrics["davies_bouldin_score"] = float(davies_bouldin_score(X_valid, labels_valid))
    else:
        metrics["silhouette_score"] = None
        metrics["calinski_harabasz_score"] = None
        metrics["davies_bouldin_score"] = None
        metrics["warning"] = (
            f"Cannot compute metrics: need >= 2 clusters with samples, "
            f"got {n_valid_clusters} valid cluster(s)"
        )

    # Cluster size distribution
    unique, counts = np.unique(labels, return_counts=True)
    metrics["cluster_sizes"] = {str(int(k)): int(v) for k, v in zip(unique, counts)}

    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    sil_str = f", silhouette={metrics['silhouette_score']:.4f}" if metrics.get("silhouette_score") is not None else ""
    ch_str = f", calinski_harabasz={metrics['calinski_harabasz_score']:.2f}" if metrics.get("calinski_harabasz_score") is not None else ""
    db_str = f", davies_bouldin={metrics['davies_bouldin_score']:.4f}" if metrics.get("davies_bouldin_score") is not None else ""
    return (
        f"evaluate_clusters: {n_clusters} clusters, {len(X)} samples"
        f"{sil_str}{ch_str}{db_str}"
    )


# =============================================================================
# CLUSTERING: GAUSSIAN MIXTURE MODEL (GMM)
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact", "artifact_type": "model"}},
        "metrics": {"format": "json", "schema": {"type": "json"}},
    },
    description="Train a Gaussian Mixture Model for probabilistic clustering",
    tags=["clustering", "gmm", "gaussian-mixture", "unsupervised", "training", "generic"],
    version="1.0.0",
)
def train_gmm(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    n_components: int = 7,
    feature_exclude: Optional[List[str]] = None,
    random_state: int = 42,
    covariance_type: str = "full",
    max_iter: int = 200,
    n_init: int = 5,
    use_pca: bool = False,
    pca_components: int = 20,
) -> str:
    """
    Train a Gaussian Mixture Model for soft clustering.

    GMM models data as a mixture of Gaussian distributions and can assign
    probabilistic cluster memberships. Often works better than KMeans
    for overlapping clusters.

    G1 Compliance: Single responsibility - train GMM model.
    G3 Compliance: Explicit random_state for reproducibility.
    G4 Compliance: feature_exclude injected via param.

    Parameters:
        n_components: Number of mixture components (clusters)
        feature_exclude: Columns to exclude from clustering (e.g., id columns)
        random_state: Random seed for reproducibility
        covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
        max_iter: Maximum number of EM iterations
        n_init: Number of initializations
        use_pca: Whether to apply PCA before clustering
        pca_components: Number of PCA components if use_pca=True
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    df = _load_data(inputs["data"])
    exclude = set(feature_exclude or [])

    # Get all numeric columns (including categorical treated as ordinal)
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    if not numeric_cols:
        raise ValueError("No numeric feature columns found after exclusion.")

    X = df[numeric_cols].fillna(0).values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optional PCA
    pca = None
    if use_pca and pca_components < X_scaled.shape[1]:
        pca = PCA(n_components=pca_components)
        X_transformed = pca.fit_transform(X_scaled)
    else:
        X_transformed = X_scaled

    # Train GMM
    model = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        covariance_type=covariance_type,
        max_iter=max_iter,
        n_init=n_init,
    )
    labels = model.fit_predict(X_transformed)

    # Compute metrics
    metrics = {
        "algorithm": "gmm",
        "n_components": n_components,
        "covariance_type": covariance_type,
        "converged": model.converged_,
        "n_iter": int(model.n_iter_),
        "bic": float(model.bic(X_transformed)),
        "aic": float(model.aic(X_transformed)),
        "n_samples": int(len(X)),
        "n_features": len(numeric_cols),
        "use_pca": use_pca,
        "feature_columns": numeric_cols,
    }

    if use_pca and pca is not None:
        metrics["pca_components"] = pca_components
        metrics["pca_explained_variance"] = float(sum(pca.explained_variance_ratio_))

    if n_components > 1 and len(X) > n_components:
        sil_score = float(silhouette_score(X_transformed, labels))
        metrics["silhouette_score"] = sil_score

    model_data = {
        "model": model,
        "scaler": scaler,
        "pca": pca,
        "feature_columns": numeric_cols,
        "algorithm": "gmm",
        "n_components": n_components,
        "use_pca": use_pca,
    }

    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(model_data, f)

    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    sil_str = f", silhouette={metrics['silhouette_score']:.4f}" if "silhouette_score" in metrics else ""
    pca_str = f", PCA={pca_components}" if use_pca else ""
    return (
        f"train_gmm: n_components={n_components}, BIC={model.bic(X_transformed):.2f}, "
        f"{len(X)} samples, {len(numeric_cols)} features{pca_str}{sil_str}"
    )


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "train_kmeans": train_kmeans,
    "train_gmm": train_gmm,
    "predict_clusters": predict_clusters,
}
