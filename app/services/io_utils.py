"""
I/O Utilities - Shared helper functions for Contract-Composable Analytics services
===========================================================

This module provides common file I/O functions used across all service modules.
All services should import from here instead of defining their own versions.

Usage:
    from services.io_utils import load_data, save_data
"""

import os
import pandas as pd
from typing import Union


def load_data(path: str) -> pd.DataFrame:
    """
    Load data from CSV, TSV, or Parquet file.

    Automatically detects format based on file extension.

    Args:
        path: Path to data file (.csv, .tsv, or .parquet)

    Returns:
        pandas DataFrame
    """
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    elif path.endswith('.tsv'):
        return pd.read_csv(path, sep='\t')
    return pd.read_csv(path)


def save_data(df: pd.DataFrame, path: str) -> None:
    """
    Save DataFrame to CSV or Parquet file.

    Automatically detects format based on file extension.
    Creates parent directories if they don't exist.

    Args:
        df: DataFrame to save
        path: Output path (.csv or .parquet)
    """
    # Create parent directory if needed
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    if path.endswith('.parquet'):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


# Aliases for backward compatibility (some files use _load_data/_save_data)
_load_data = load_data
_save_data = save_data
