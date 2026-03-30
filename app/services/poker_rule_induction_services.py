"""
Poker Rule Induction - Contract-Composable Analytics Services
=======================================
Competition: https://www.kaggle.com/competitions/poker-rule-induction
Problem Type: Multiclass Classification (10 classes: 0-9)
Target: hand
Metric: Accuracy

Features: S1-S5 (suit 1-4), C1-C5 (card rank 1-13)
Hand classes: 0=Nothing, 1=One pair, 2=Two pairs, 3=Three of a kind,
             4=Straight, 5=Flush, 6=Full house, 7=Four of a kind,
             8=Straight flush, 9=Royal flush

Solution Insights (from top 3 notebooks):
- Raw features alone give poor accuracy (~0.5) because poker hands are
  combinatorial rules, not simple feature thresholds
- Solution 3 (100% accuracy) engineers boolean features for each hand type:
  flush (all same suit), straight (consecutive ranks), pairs, etc.
- After engineering, a RandomForest on boolean features achieves near-perfect accuracy

Competition-specific services:
- preprocess_poker_hands: Engineer poker hand features from suit/rank columns
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract
from services.io_utils import load_data as _load_data, save_data as _save_data


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Engineer poker hand features from suit and rank columns",
    tags=["preprocessing", "feature-engineering", "poker", "classification", "generic"],
    version="1.0.0",
)
def preprocess_poker_hands(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    suit_columns: Optional[List[str]] = None,
    rank_columns: Optional[List[str]] = None,
    drop_original: bool = True,
    id_column: str = "id",
    target_column: str = "hand",
) -> str:
    """
    Engineer poker hand features from card suit and rank columns.

    Creates boolean features for each poker hand type:
    - is_flush: All 5 cards share the same suit
    - is_straight: 5 consecutive ranks (including ace-low A-2-3-4-5)
    - is_royal_flush: Straight flush with 10-J-Q-K-A
    - is_straight_flush: Both flush and straight (not royal)
    - is_four_of_a_kind: 4 cards of the same rank
    - is_full_house: 3 of one rank + 2 of another
    - is_three_of_a_kind: 3 cards of same rank (+ 2 different)
    - is_two_pairs: 2 different pairs
    - is_one_pair: Exactly one pair (4 unique ranks)
    - is_nothing: 5 unique ranks, no flush, no straight
    - n_unique_ranks: Count of distinct rank values

    Parameters:
        suit_columns: Suit column names (default: S1-S5)
        rank_columns: Rank column names (default: C1-C5)
        drop_original: Whether to drop original suit/rank columns
        id_column: ID column name
        target_column: Target column name
    """
    df = _load_data(inputs["data"])

    if suit_columns is None:
        suit_columns = ["S1", "S2", "S3", "S4", "S5"]
    if rank_columns is None:
        rank_columns = ["C1", "C2", "C3", "C4", "C5"]

    suits = df[suit_columns].values
    ranks = df[rank_columns].values

    n = len(df)

    # Flush: all 5 suits identical
    is_flush = np.all(suits == suits[:, [0]], axis=1)

    # Straight detection: 5 consecutive ranks (also handle ace-low: A-2-3-4-5)
    is_straight = np.zeros(n, dtype=bool)
    for i in range(n):
        sorted_r = np.sort(ranks[i])
        unique_r = np.unique(sorted_r)
        if len(unique_r) == 5:
            # Normal straight: max - min == 4
            if sorted_r[4] - sorted_r[0] == 4:
                is_straight[i] = True
            # Ace-high straight (10-J-Q-K-A = {1,10,11,12,13})
            elif set(sorted_r) == {1, 10, 11, 12, 13}:
                is_straight[i] = True

    # Royal flush: straight flush with 10-J-Q-K-A
    is_royal_flush = is_flush & is_straight & np.array(
        [set(ranks[i]) == {1, 10, 11, 12, 13} for i in range(n)]
    )

    # Straight flush (not royal)
    is_straight_flush = is_flush & is_straight & ~is_royal_flush

    # Flush only (not straight flush or royal)
    is_flush_only = is_flush & ~is_straight

    # Straight only (not flush)
    is_straight_only = is_straight & ~is_flush

    # Count unique ranks and max count per rank (for pair/triple/quad detection)
    n_unique = np.array([len(np.unique(ranks[i])) for i in range(n)])

    max_count = np.zeros(n, dtype=int)
    second_max_count = np.zeros(n, dtype=int)
    for i in range(n):
        _, counts = np.unique(ranks[i], return_counts=True)
        sorted_counts = np.sort(counts)[::-1]
        max_count[i] = sorted_counts[0]
        if len(sorted_counts) > 1:
            second_max_count[i] = sorted_counts[1]

    is_four_of_a_kind = max_count == 4
    is_full_house = (max_count == 3) & (second_max_count == 2)
    is_three_of_a_kind = (max_count == 3) & (second_max_count == 1)
    is_two_pairs = (max_count == 2) & (n_unique == 3)
    is_one_pair = (n_unique == 4)
    is_nothing = (
        (n_unique == 5)
        & ~is_flush
        & ~is_straight
    )

    # Assign features
    df["is_flush"] = is_flush_only.astype(int)
    df["is_straight"] = is_straight_only.astype(int)
    df["is_royal_flush"] = is_royal_flush.astype(int)
    df["is_straight_flush"] = is_straight_flush.astype(int)
    df["is_four_of_a_kind"] = is_four_of_a_kind.astype(int)
    df["is_full_house"] = is_full_house.astype(int)
    df["is_three_of_a_kind"] = is_three_of_a_kind.astype(int)
    df["is_two_pairs"] = is_two_pairs.astype(int)
    df["is_one_pair"] = is_one_pair.astype(int)
    df["is_nothing"] = is_nothing.astype(int)
    df["n_unique_ranks"] = n_unique

    if drop_original:
        df = df.drop(columns=suit_columns + rank_columns, errors="ignore")

    _save_data(df, outputs["data"])

    n_features = sum(1 for c in df.columns if c not in [id_column, target_column])
    return f"preprocess_poker_hands: {n_features} features engineered for {n} rows"


SERVICE_REGISTRY = {
    "preprocess_poker_hands": preprocess_poker_hands,
}
