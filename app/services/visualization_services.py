"""
Visualization Services - Common Module
=============================================

Generic visualization services reusable across any tabular competition.

All services follow G1-G6 design principles:
- G1: Each service does exactly ONE visualization task
- G2: Explicit I/O contracts with @contract
- G3: Pure functions, deterministic output
- G4: No hardcoded column names (injected via params)
- G5: DAG compatible - PASS-THROUGH pattern (data in = data out)
- G6: Semantic metadata via docstrings/tags

DESIGN DECISION - Pass-Through Pattern:
    Visualization services output BOTH the plot AND pass through the data.
    This allows insertion anywhere in a pipeline without breaking the flow.

    [preprocess] → [plot_distributions] → [train]
                         ↓
                    (plot.png)   +   (data passes through)

Services:
  plot_feature_distributions: Histogram/KDE for numeric, bar for categorical
  plot_correlation_matrix: Heatmap of feature correlations
  plot_missing_values: Visualize missing data patterns
  plot_target_distribution: Distribution of target variable
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Literal, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract

from services.io_utils import load_data as _load_data, save_data as _save_data


# =============================================================================
# PLOT FEATURE DISTRIBUTIONS - Generic for any tabular data
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},  # Pass-through
        "plot": {"format": "png", "schema": {"type": "image"}},
    },
    description="Plot feature distributions (histograms for numeric, bar charts for categorical). Pass-through pattern for pipeline composability.",
    tags=["visualization", "eda", "distributions", "generic", "pass-through"],
    version="1.0.0",
)
def plot_feature_distributions(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    # G4 Compliance: Column selection via parameters, not hardcoded
    columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None,
    # Plot configuration
    max_columns: int = 20,
    figsize_per_plot: tuple = (4, 3),
    n_cols: int = 4,
    bins: int = 30,
    # For categorical
    max_categories: int = 20,
    # Output options
    title: Optional[str] = None,
    dpi: int = 100,
) -> str:
    """
    Plot distributions of features in a dataset.

    G1 Compliance: Single task - visualize distributions only.
    G2 Compliance: Explicit inputs (CSV) and outputs (CSV + PNG).
    G3 Compliance: Deterministic - same data produces same plot.
    G4 Compliance: No hardcoded column names.
    G5 Compliance: Pass-through pattern - data flows through unchanged.
    G6 Compliance: Tagged for discovery.

    Parameters:
        columns: Specific columns to plot. If None, auto-detect from data.
        exclude_columns: Columns to exclude (e.g., id, target).
        max_columns: Maximum number of columns to plot (prevents huge plots).
        figsize_per_plot: Size of each subplot (width, height).
        n_cols: Number of columns in subplot grid.
        bins: Number of bins for histograms.
        max_categories: Max unique values before treating as numeric.
        title: Optional plot title.
        dpi: Resolution of output image.

    Returns:
        Status message with plot details.
    """
    # Lazy import to avoid loading matplotlib unless needed
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load data
    df = _load_data(inputs["data"])
    original_shape = df.shape

    # Determine columns to plot
    exclude = set(exclude_columns or [])

    if columns:
        # Use specified columns (G4: injected via param)
        plot_cols = [c for c in columns if c in df.columns and c not in exclude]
    else:
        # Auto-detect: all columns except excluded
        plot_cols = [c for c in df.columns if c not in exclude]

    # Limit to max_columns
    if len(plot_cols) > max_columns:
        plot_cols = plot_cols[:max_columns]

    if not plot_cols:
        # Nothing to plot - still pass through data
        _save_data(df, outputs["data"])
        return f"plot_feature_distributions: No columns to plot (excluded all)"

    # Classify columns
    numeric_cols = []
    categorical_cols = []

    for col in plot_cols:
        n_unique = df[col].nunique()
        if df[col].dtype in [np.number, 'int64', 'float64', 'int32', 'float32']:
            if n_unique <= max_categories:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        elif df[col].dtype == 'object' or df[col].dtype.name == 'category':
            if n_unique <= max_categories:
                categorical_cols.append(col)
            else:
                # Too many categories - skip or sample
                categorical_cols.append(col)
        else:
            # Default: try as numeric
            try:
                pd.to_numeric(df[col], errors='raise')
                numeric_cols.append(col)
            except:
                categorical_cols.append(col)

    all_cols = numeric_cols + categorical_cols
    n_plots = len(all_cols)

    if n_plots == 0:
        _save_data(df, outputs["data"])
        return f"plot_feature_distributions: No valid columns to plot"

    # Calculate grid dimensions
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)
    )

    # Flatten axes for easy iteration
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    plot_idx = 0

    # Plot numeric columns (histogram + KDE)
    for col in numeric_cols:
        ax = axes[plot_idx]
        try:
            data = df[col].dropna()
            if len(data) > 0:
                sns.histplot(data, bins=bins, kde=True, ax=ax, color='steelblue')
                ax.set_title(f'{col}\n(n={len(data)}, μ={data.mean():.2f})', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{col}\n(empty)', fontsize=9)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:20]}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{col}', fontsize=9)

        ax.set_xlabel('')
        ax.tick_params(labelsize=8)
        plot_idx += 1

    # Plot categorical columns (bar chart)
    for col in categorical_cols:
        ax = axes[plot_idx]
        try:
            value_counts = df[col].value_counts().head(max_categories)
            if len(value_counts) > 0:
                value_counts.plot(kind='bar', ax=ax, color='coral')
                ax.set_title(f'{col}\n({len(value_counts)} categories)', fontsize=9)
                ax.tick_params(axis='x', rotation=45, labelsize=7)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{col}\n(empty)', fontsize=9)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:20]}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{col}', fontsize=9)

        ax.set_xlabel('')
        plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)

    # Title
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')
    else:
        fig.suptitle(f'Feature Distributions ({original_shape[0]} rows, {n_plots} features)', fontsize=12)

    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(outputs["plot"]) or ".", exist_ok=True)
    plt.savefig(outputs["plot"], dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # PASS-THROUGH: Save data unchanged
    _save_data(df, outputs["data"])

    return f"plot_feature_distributions: plotted {len(numeric_cols)} numeric, {len(categorical_cols)} categorical columns → {outputs['plot']}"


# =============================================================================
# PLOT CORRELATION MATRIX
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},  # Pass-through
        "plot": {"format": "png", "schema": {"type": "image"}},
    },
    description="Plot correlation matrix heatmap for numeric features. Pass-through pattern.",
    tags=["visualization", "eda", "correlation", "generic", "pass-through"],
    version="1.0.0",
)
def plot_correlation_matrix(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    figsize: tuple = (12, 10),
    annot: bool = True,
    cmap: str = "coolwarm",
    title: Optional[str] = None,
    dpi: int = 100,
) -> str:
    """
    Plot correlation matrix heatmap for numeric features.

    G1 Compliance: Single task - visualize correlations only.
    G2 Compliance: Explicit inputs (CSV) and outputs (CSV + PNG).
    G3 Compliance: Deterministic - same data produces same plot.
    G4 Compliance: Column selection via parameters, not hardcoded.
    G5 Compliance: Pass-through pattern - data flows through unchanged.
    G6 Compliance: Tagged for discovery.

    Parameters:
        inputs: Dict with 'data' key pointing to input CSV path.
        outputs: Dict with 'data' (pass-through CSV) and 'plot' (PNG) paths.
        columns: Specific columns to include. If None, auto-detect numeric.
        exclude_columns: Columns to exclude (e.g., id columns).
        method: Correlation method - 'pearson', 'spearman', or 'kendall'.
        figsize: Figure size as (width, height) tuple.
        annot: Whether to annotate cells with correlation values.
        cmap: Colormap for heatmap (e.g., 'coolwarm', 'viridis').
        title: Optional plot title.
        dpi: Resolution of output image.

    Returns:
        Status message with number of features and output path.

    Example:
        >>> plot_correlation_matrix(
        ...     inputs={"data": "train.csv"},
        ...     outputs={"data": "train_out.csv", "plot": "corr.png"},
        ...     exclude_columns=["id"],
        ...     method="spearman"
        ... )
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = _load_data(inputs["data"])
    exclude = set(exclude_columns or [])

    # Select numeric columns
    if columns:
        numeric_cols = [c for c in columns if c in df.columns and c not in exclude]
    else:
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    if len(numeric_cols) < 2:
        _save_data(df, outputs["data"])
        return "plot_correlation_matrix: Need at least 2 numeric columns"

    # Limit to prevent huge matrices
    if len(numeric_cols) > 50:
        numeric_cols = numeric_cols[:50]

    # Calculate correlation
    corr_matrix = df[numeric_cols].corr(method=method)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Adjust annotation size based on matrix size
    annot_size = max(6, min(10, 200 // len(numeric_cols)))

    sns.heatmap(
        corr_matrix,
        annot=annot and len(numeric_cols) <= 20,
        fmt='.2f',
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={'size': annot_size}
    )

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    else:
        ax.set_title(f'Correlation Matrix ({method}, {len(numeric_cols)} features)', fontsize=12)

    plt.tight_layout()

    os.makedirs(os.path.dirname(outputs["plot"]) or ".", exist_ok=True)
    plt.savefig(outputs["plot"], dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Pass-through
    _save_data(df, outputs["data"])

    return f"plot_correlation_matrix: {len(numeric_cols)} features, method={method} → {outputs['plot']}"


# =============================================================================
# PLOT TARGET DISTRIBUTION
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},  # Pass-through
        "plot": {"format": "png", "schema": {"type": "image"}},
    },
    description="Plot distribution of target variable (histogram for regression, bar for classification).",
    tags=["visualization", "eda", "target", "generic", "pass-through"],
    version="1.0.0",
)
def plot_target_distribution(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    # G4: Target column injected via parameter
    target_column: str = "target",
    task_type: str = "auto",  # auto, regression, classification
    figsize: tuple = (10, 6),
    bins: int = 50,
    title: Optional[str] = None,
    dpi: int = 100,
) -> str:
    """
    Plot distribution of target variable.

    Automatically detects task type based on unique values, or use explicit setting.
    - Regression: histogram with KDE, mean/median lines, and statistics
    - Classification: bar chart with class counts and majority class info

    G1 Compliance: Single task - visualize target distribution only.
    G2 Compliance: Explicit inputs (CSV) and outputs (CSV + PNG).
    G3 Compliance: Deterministic - same data produces same plot.
    G4 Compliance: Target column injected via parameter, not hardcoded.
    G5 Compliance: Pass-through pattern - data flows through unchanged.
    G6 Compliance: Tagged for discovery.

    Parameters:
        inputs: Dict with 'data' key pointing to input CSV path.
        outputs: Dict with 'data' (pass-through CSV) and 'plot' (PNG) paths.
        target_column: Name of the target column to plot.
        task_type: 'auto' (detect), 'regression', or 'classification'.
        figsize: Figure size as (width, height) tuple.
        bins: Number of bins for regression histograms.
        title: Optional plot title.
        dpi: Resolution of output image.

    Returns:
        Status message with target column, task type, and sample count.

    Example:
        >>> plot_target_distribution(
        ...     inputs={"data": "train.csv"},
        ...     outputs={"data": "train_out.csv", "plot": "target.png"},
        ...     target_column="SalePrice",
        ...     task_type="regression"
        ... )
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = _load_data(inputs["data"])

    if target_column not in df.columns:
        _save_data(df, outputs["data"])
        return f"plot_target_distribution: target column '{target_column}' not found"

    target = df[target_column].dropna()

    # Auto-detect task type
    if task_type == "auto":
        n_unique = target.nunique()
        if n_unique <= 30 and (target.dtype == 'object' or n_unique <= 10):
            task_type = "classification"
        else:
            task_type = "regression"

    fig, ax = plt.subplots(figsize=figsize)

    if task_type == "regression":
        sns.histplot(target, bins=bins, kde=True, ax=ax, color='steelblue')
        ax.axvline(float(target.mean()), color='red', linestyle='--', label=f'Mean: {target.mean():.2f}')
        ax.axvline(float(target.median()), color='green', linestyle='--', label=f'Median: {target.median():.2f}')
        ax.legend()
        stats_text = f'μ={target.mean():.2f}, σ={target.std():.2f}, min={target.min():.2f}, max={target.max():.2f}'
    else:
        value_counts = target.value_counts()
        value_counts.plot(kind='bar', ax=ax, color='coral')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        stats_text = f'{len(value_counts)} classes, majority: {value_counts.index[0]} ({value_counts.iloc[0]} samples)'

    if title:
        ax.set_title(f'{title}\n{stats_text}', fontsize=12, fontweight='bold')
    else:
        ax.set_title(f'Target Distribution: {target_column} ({task_type})\n{stats_text}', fontsize=12)

    ax.set_xlabel(target_column)
    ax.set_ylabel('Count')

    plt.tight_layout()

    os.makedirs(os.path.dirname(outputs["plot"]) or ".", exist_ok=True)
    plt.savefig(outputs["plot"], dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Pass-through
    _save_data(df, outputs["data"])

    return f"plot_target_distribution: {target_column} ({task_type}, {len(target)} samples) → {outputs['plot']}"
