"""
SLEGO Services for integer-sequence-learning competition
=========================================================
Regression - Target: Last (predict next integer in OEIS sequence)
Knowledge-based integer sequence prediction challenge

G1-G6 compliant services for integer sequence analysis.
Inspired by top solutions: recurrence relation detection, difference analysis,
prefix matching patterns from OEIS sequences.
"""

import os
import json
import math
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional, List

# Import contract decorator and shared utilities
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slego_contract import contract

# =============================================================================
# HELPERS: Import from shared io_utils
# =============================================================================
from services.io_utils import load_data as _load_data, save_data as _save_data

# Import reusable services
from services.preprocessing_services import split_data, create_submission
from services.regression_services import train_lightgbm_regressor, predict_regressor


def _signed_log1p(x):
    """Signed log1p that handles Python big ints and scalar/array inputs."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


# =============================================================================
# SEQUENCE FEATURE EXTRACTION SERVICES
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Parse sequence and add Last element as target",
    tags=["preprocessing", "sequence"],
    version="2.0.0",
)
def parse_sequence_target(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    sequence_column: str = "Sequence",
    target_column: str = "Last",
    remove_last: bool = True,
) -> str:
    """
    Parse sequences and extract the last element as the target.

    For training data where we need to predict the next element,
    this takes the last element as the target and removes it from
    the sequence for feature extraction.

    Applies signed log1p transform to ALL targets to normalize
    the wildly varying magnitudes in OEIS sequences.

    G4 Compliance: Column names injected via params.
    """
    df = _load_data(inputs["data"])

    targets = []
    sequences = []

    for seq_str in df[sequence_column]:
        try:
            seq = [int(x.strip()) for x in str(seq_str).split(",") if x.strip()]
        except Exception:
            seq = [0]

        if len(seq) >= 2 and remove_last:
            target_val = seq[-1]
            sequences.append(",".join(str(x) for x in seq[:-1]))
        else:
            target_val = seq[-1] if seq else 0
            sequences.append(seq_str)

        # Apply signed log1p transform to ALL targets
        # This normalizes targets that range from -inf to +inf with extreme magnitudes
        target_val = float(_signed_log1p(float(target_val)))
        targets.append(target_val)

    df[target_column] = targets
    if remove_last:
        df[sequence_column] = sequences

    _save_data(df, outputs["data"])

    return f"parse_sequence_target: extracted {len(df)} targets (signed log1p transformed)"


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Extract features from integer sequences",
    tags=["feature-engineering", "sequence"],
    version="2.0.0",
)
def extract_sequence_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    sequence_column: str = "Sequence",
    target_column: str = "Last",
    id_column: str = "Id",
    max_sequence_length: int = 20,
    prefix: str = "seq_",
) -> str:
    """
    Extract numerical features from integer sequences.

    Creates features including:
    - Last N elements (log-scaled) of sequence
    - First, second, third differences statistics
    - Ratio features between consecutive elements
    - Statistical features (mean, std, min, max) in log space
    - Recurrence relation detection (order 2, 3, 4)
    - Constant-difference detection (polynomial sequences)
    - GCD of sequence elements
    - Sign change count
    - Growth pattern indicators

    Inspired by top Kaggle solutions (recurrence relations, difference analysis).

    G4 Compliance: Column names and parameters injected via params.
    """
    df = _load_data(inputs["data"])

    features = []

    for idx, row in df.iterrows():
        seq_str = row[sequence_column]

        # Parse sequence - keep as Python ints for exact math
        try:
            seq_int = [int(x.strip()) for x in str(seq_str).split(",") if x.strip()]
        except Exception:
            seq_int = [0]
        # Also create float version (clamped to float64 range)
        seq = [max(min(float(x), 1e15), -1e15) for x in seq_int]

        feat = {}

        # Keep ID
        if id_column in df.columns:
            feat[id_column] = row[id_column]

        # Keep target if present (training data)
        if target_column in df.columns:
            feat[target_column] = row[target_column]

        # --- Basic properties ---
        feat[f"{prefix}length"] = len(seq)

        # --- Last N elements as features (signed log1p scaled) ---
        for i in range(max_sequence_length):
            if i < len(seq):
                val = seq[-(i + 1)]
                feat[f"{prefix}elem_{i}"] = float(_signed_log1p(val))
            else:
                feat[f"{prefix}elem_{i}"] = 0.0

        # --- Statistics in log space ---
        arr = np.array(seq, dtype=np.float64)

        if len(seq) >= 2:
            # Log-space stats
            log_arr = _signed_log1p(arr)
            feat[f"{prefix}log_mean"] = np.mean(log_arr)
            feat[f"{prefix}log_std"] = np.std(log_arr)
            feat[f"{prefix}log_min"] = np.min(log_arr)
            feat[f"{prefix}log_max"] = np.max(log_arr)
            feat[f"{prefix}log_range"] = feat[f"{prefix}log_max"] - feat[f"{prefix}log_min"]
            feat[f"{prefix}log_last"] = log_arr[-1]

            # --- First differences ---
            diffs = np.diff(arr)
            log_diffs = _signed_log1p(diffs)
            feat[f"{prefix}diff_mean"] = np.mean(log_diffs)
            feat[f"{prefix}diff_std"] = np.std(log_diffs) if len(log_diffs) > 1 else 0.0
            feat[f"{prefix}diff_last"] = log_diffs[-1] if len(log_diffs) > 0 else 0.0

            # --- Second differences ---
            if len(diffs) > 1:
                diffs2 = np.diff(diffs)
                log_diffs2 = _signed_log1p(diffs2)
                feat[f"{prefix}diff2_mean"] = np.mean(log_diffs2)
                feat[f"{prefix}diff2_std"] = np.std(log_diffs2) if len(log_diffs2) > 1 else 0.0
                feat[f"{prefix}diff2_last"] = log_diffs2[-1] if len(log_diffs2) > 0 else 0.0
            else:
                feat[f"{prefix}diff2_mean"] = 0.0
                feat[f"{prefix}diff2_std"] = 0.0
                feat[f"{prefix}diff2_last"] = 0.0

            # --- Third differences ---
            if len(diffs) > 2:
                diffs3 = np.diff(np.diff(diffs))
                log_diffs3 = _signed_log1p(diffs3)
                feat[f"{prefix}diff3_mean"] = np.mean(log_diffs3)
            else:
                feat[f"{prefix}diff3_mean"] = 0.0

            # --- Constant difference detection (polynomial sequences) ---
            # If first diffs are constant -> linear sequence
            feat[f"{prefix}is_const_diff1"] = int(
                len(diffs) >= 3 and np.std(diffs) < 1e-6
            )
            # If second diffs are constant -> quadratic sequence
            if len(diffs) > 1:
                diffs2_raw = np.diff(diffs)
                feat[f"{prefix}is_const_diff2"] = int(
                    len(diffs2_raw) >= 3 and np.std(diffs2_raw) < 1e-6
                )
            else:
                feat[f"{prefix}is_const_diff2"] = 0

            # --- Ratios ---
            if seq[-2] != 0:
                ratio = float(seq[-1]) / float(seq[-2])
                sign = 1.0 if ratio >= 0 else -1.0
                feat[f"{prefix}ratio_last"] = float(_signed_log1p(ratio))
            else:
                feat[f"{prefix}ratio_last"] = 0.0

            # Average ratio of consecutive non-zero pairs
            ratios = []
            for j in range(1, min(len(seq), 10)):
                if seq[-(j + 1)] != 0:
                    ratios.append(float(seq[-j]) / float(seq[-(j + 1)]))
            if ratios:
                feat[f"{prefix}ratio_mean"] = np.mean(ratios)
                feat[f"{prefix}ratio_std"] = np.std(ratios) if len(ratios) > 1 else 0.0
                feat[f"{prefix}is_geometric"] = int(
                    len(ratios) >= 3 and np.std(ratios) < 0.01 * (abs(np.mean(ratios)) + 1e-10)
                )
            else:
                feat[f"{prefix}ratio_mean"] = 0.0
                feat[f"{prefix}ratio_std"] = 0.0
                feat[f"{prefix}is_geometric"] = 0

            # --- Growth indicators ---
            feat[f"{prefix}is_increasing"] = int(
                all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))
            )
            feat[f"{prefix}is_decreasing"] = int(
                all(arr[i] >= arr[i + 1] for i in range(len(arr) - 1))
            )

            # --- Sign changes ---
            signs = np.sign(arr)
            nonzero_signs = signs[signs != 0]
            if len(nonzero_signs) > 1:
                feat[f"{prefix}sign_changes"] = int(np.sum(np.diff(nonzero_signs) != 0))
            else:
                feat[f"{prefix}sign_changes"] = 0
            feat[f"{prefix}has_negatives"] = int(np.any(arr < 0))
            feat[f"{prefix}has_zeros"] = int(np.any(arr == 0))

            # --- GCD feature ---
            try:
                abs_seq = [abs(x) for x in seq_int if x != 0]
                if abs_seq:
                    gcd_val = abs_seq[0]
                    for v in abs_seq[1:]:
                        gcd_val = math.gcd(gcd_val, v)
                    feat[f"{prefix}gcd"] = float(np.log1p(float(gcd_val)))
                else:
                    feat[f"{prefix}gcd"] = 0.0
            except (ValueError, OverflowError):
                feat[f"{prefix}gcd"] = 0.0

            # --- Recurrence relation detection ---
            # Inspired by top solution (ncchen/recurrence-relation, 103 votes)
            feat[f"{prefix}recurrence_order"] = 0
            for order in [2, 3, 4]:
                if _check_recurrence(seq_int, order):
                    feat[f"{prefix}recurrence_order"] = order
                    break
        else:
            # Single element sequence - fill with defaults
            val = float(seq[0]) if seq else 0.0
            sign = 1.0 if val >= 0 else -1.0
            log_val = float(_signed_log1p(val))
            feat[f"{prefix}log_mean"] = log_val
            feat[f"{prefix}log_std"] = 0.0
            feat[f"{prefix}log_min"] = log_val
            feat[f"{prefix}log_max"] = log_val
            feat[f"{prefix}log_range"] = 0.0
            feat[f"{prefix}log_last"] = log_val
            feat[f"{prefix}diff_mean"] = 0.0
            feat[f"{prefix}diff_std"] = 0.0
            feat[f"{prefix}diff_last"] = 0.0
            feat[f"{prefix}diff2_mean"] = 0.0
            feat[f"{prefix}diff2_std"] = 0.0
            feat[f"{prefix}diff2_last"] = 0.0
            feat[f"{prefix}diff3_mean"] = 0.0
            feat[f"{prefix}is_const_diff1"] = 0
            feat[f"{prefix}is_const_diff2"] = 0
            feat[f"{prefix}ratio_last"] = 0.0
            feat[f"{prefix}ratio_mean"] = 0.0
            feat[f"{prefix}ratio_std"] = 0.0
            feat[f"{prefix}is_geometric"] = 0
            feat[f"{prefix}is_increasing"] = 0
            feat[f"{prefix}is_decreasing"] = 0
            feat[f"{prefix}sign_changes"] = 0
            feat[f"{prefix}has_negatives"] = int(val < 0)
            feat[f"{prefix}has_zeros"] = int(val == 0)
            feat[f"{prefix}gcd"] = 0.0
            feat[f"{prefix}recurrence_order"] = 0

        features.append(feat)

    result_df = pd.DataFrame(features)

    _save_data(result_df, outputs["data"])

    n_features = len([c for c in result_df.columns if c.startswith(prefix)])
    return f"extract_sequence_features: extracted {n_features} features from {len(df)} sequences"


def _check_recurrence(seq, order, min_length=7):
    """
    Check if sequence satisfies a linear recurrence relation of given order.

    Based on the top-voted Kaggle solution by ncchen (103 votes).
    Solves for coefficients using matrix inversion and validates against
    remaining sequence elements.

    Returns True if the sequence matches a recurrence of the given order.
    """
    if len(seq) < max(2 * order + 1, min_length):
        return False

    try:
        A = []
        b = []
        for i in range(order):
            A.append(seq[i:i + order])
            b.append(seq[i + order])
        A = np.array(A, dtype=np.float64)
        b = np.array(b, dtype=np.float64)

        if abs(np.linalg.det(A)) < 1e-10:
            return False

        coeffs = np.linalg.solve(A, b)

        # Validate against remaining elements
        for i in range(2 * order, len(seq)):
            segment = np.array(seq[i - order:i], dtype=np.float64)
            predicted = np.sum(coeffs * segment)
            if abs(predicted - seq[i]) > 0.5:
                return False

        return True
    except (np.linalg.LinAlgError, ValueError, OverflowError):
        return False


@contract(
    inputs={
        "predictions": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "predictions": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Inverse signed log1p transform and round predictions to integers",
    tags=["postprocessing", "sequence", "inverse-transform"],
    version="1.0.0",
)
def inverse_log_round_predictions(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    prediction_column: str = "Last",
    id_column: str = "Id",
) -> str:
    """
    Inverse the signed log1p transform applied during parse_sequence_target,
    then round to nearest integer for submission.

    Transform: sign(x) * log1p(abs(x)) -> sign(x) * expm1(abs(x))

    G4 Compliance: Column names injected via params.
    """
    pred_df = _load_data(inputs["predictions"])

    raw_preds = pred_df[prediction_column].values.astype(np.float64)

    # Inverse signed log1p: sign(x) * expm1(abs(x))
    signs = np.sign(raw_preds)
    abs_preds = np.abs(raw_preds)

    # Cap abs values before expm1 to avoid overflow beyond float64/int64 range
    # log1p(9.2e18) ~ 43.7, so cap at 43 to stay within int64
    abs_preds = np.minimum(abs_preds, 43.0)
    restored = signs * np.expm1(abs_preds)

    # Round to nearest integer, handle NaN/Inf
    restored = np.nan_to_num(restored, nan=0.0, posinf=0.0, neginf=0.0)
    restored = np.round(restored).astype(np.int64)

    pred_df[prediction_column] = restored

    _save_data(pred_df, outputs["predictions"])

    return (
        f"inverse_log_round_predictions: {len(pred_df)} predictions, "
        f"range=[{restored.min()}, {restored.max()}]"
    )


def _detect_recurrence_and_predict(seq_int, max_order=6):
    """
    Try to detect recurrence relation and predict next element.
    Tests orders 2, 3, 4, 5, 6. Returns predicted int or None.

    Extended to higher orders based on ncchen solution analysis.
    """
    for order in range(2, max_order + 1):
        if len(seq_int) < max(2 * order + 1, 7):
            continue
        try:
            A = []
            b = []
            for i in range(order):
                A.append(seq_int[i:i + order])
                b.append(seq_int[i + order])
            A_arr = np.array(A, dtype=np.float64)
            b_arr = np.array(b, dtype=np.float64)

            if abs(np.linalg.det(A_arr)) < 1e-10:
                continue

            coeffs = np.linalg.solve(A_arr, b_arr)

            # Validate against remaining elements
            valid = True
            for i in range(2 * order, len(seq_int)):
                segment = np.array(seq_int[i - order:i], dtype=np.float64)
                predicted = np.sum(coeffs * segment)
                if abs(predicted - seq_int[i]) > 0.5:
                    valid = False
                    break

            if valid:
                # Predict next element
                last_segment = np.array(seq_int[-order:], dtype=np.float64)
                next_val = np.sum(coeffs * last_segment)
                return int(round(next_val))
        except (np.linalg.LinAlgError, ValueError, OverflowError):
            continue
    return None


def _detect_constant_diff_and_predict(seq_int, max_order=5):
    """
    If k-th order differences are constant, predict next element by
    extending the constant difference pattern. Works for polynomial sequences.
    Returns predicted int or None.
    """
    diffs = list(seq_int)
    for k in range(1, max_order + 1):
        if len(diffs) < 3:
            return None
        new_diffs = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
        # Check if constant
        if len(set(new_diffs)) == 1:
            # Reconstruct: add the constant, then work back up
            predicted = seq_int[-1]
            # Build the last elements of each diff level
            last_vals = [seq_int[-1]]
            d = list(seq_int)
            for _ in range(k):
                d = [d[i + 1] - d[i] for i in range(len(d) - 1)]
                last_vals.append(d[-1])
            # Propagate: diff[k] stays constant, compute back to level 0
            for level in range(k - 1, -1, -1):
                last_vals[level] = last_vals[level] + last_vals[level + 1]
            return last_vals[0]
        diffs = new_diffs
    return None


# =============================================================================
# PREFIX LOOKUP APPROACH (from balzac solution - 2nd highest voted)
# =============================================================================

def _find_gcd(seq):
    """Find GCD of all elements in sequence."""
    if not seq:
        return 1
    gcd = abs(seq[0])
    for i in range(1, len(seq)):
        if seq[i] != 0:
            gcd = math.gcd(gcd, abs(seq[i]))
    return gcd if gcd != 0 else 1


def _find_signature(seq):
    """
    Compute signature of sequence for prefix matching.
    signature(seq) = sign(first nonzero) * seq / GCD(seq)

    This normalizes sequences so [2,4,6,8] -> [1,2,3,4]
    """
    nonzero_seq = [d for d in seq if d != 0]
    if len(nonzero_seq) == 0:
        return tuple(seq)
    sign = 1 if nonzero_seq[0] > 0 else -1
    gcd = _find_gcd(seq)
    if gcd == 0:
        gcd = 1
    return tuple(sign * x // gcd for x in seq)


def _find_derivative(seq):
    """Compute difference array."""
    if len(seq) <= 1:
        return [0]
    return [seq[i] - seq[i-1] for i in range(1, len(seq))]


class PrefixTree:
    """
    Trie for storing FULL sequence signatures (including next element).
    The trie stores the full normalized sequence and allows prefix lookup.
    """
    def __init__(self):
        self.data = {}
        self.count = 0

    def put(self, signature, weight=100):
        """
        Store a full sequence signature in the trie.
        The signature includes the next element as the last item.
        """
        node = self.data
        node_created = False
        for i, item in enumerate(signature):
            if item not in node:
                node[item] = {}
                node_created = True
            node = node[item]

        # Only store if this is a new longer sequence
        if node_created:
            if '_weight' not in node or weight > node['_weight']:
                node['_weight'] = weight
                self.count += 1

    def find_longer_sequences(self, prefix):
        """
        Find all sequences in the trie that have 'prefix' as a prefix
        and are longer than the prefix.

        Returns list of (full_signature, weight) tuples.
        """
        node = self.data
        for item in prefix:
            if item not in node:
                return []
            node = node[item]

        # Now traverse the subtree to find all stored sequences
        results = []
        self._collect_sequences(node, list(prefix), results)
        return results

    def _collect_sequences(self, node, current_seq, results, max_depth=10):
        """Recursively collect sequences from subtree."""
        if max_depth <= 0:
            return

        if '_weight' in node:
            results.append((tuple(current_seq), node['_weight']))

        for key, child in node.items():
            if key != '_weight':
                current_seq.append(key)
                self._collect_sequences(child, current_seq, results, max_depth - 1)
                current_seq.pop()


def _build_prefix_tree_v2(all_sequences):
    """
    Build prefix tree from all sequences (train + test).
    Stores FULL normalized signatures to enable suffix lookup.

    Args:
        all_sequences: list of sequence_ints (lists of integers)

    Returns:
        PrefixTree instance
    """
    trie = PrefixTree()

    # Add a constant sequence for polynomial detection
    const_sig = tuple([1] * 50)
    trie.put(const_sig, weight=50)

    for seq_int in all_sequences:
        if len(seq_int) < 3:
            continue

        der = list(seq_int)

        for diff_depth in range(4):  # up to 4th order differences
            if len(der) < 3:
                break

            seq = list(der)

            for shift in range(min(4, len(seq) - 2)):
                # Skip leading zeros
                while len(seq) > 0 and seq[0] == 0:
                    seq = seq[1:]

                if len(seq) < 3:
                    break

                signature = _find_signature(seq)

                # Weight based on sequence length (longer = more reliable)
                weight = len(seq) * 100 // len(seq_int)

                # Store the full signature (including next element implicitly)
                trie.put(signature, weight)

                if len(seq) <= 3:
                    break
                seq = seq[1:]  # Shift

            der = _find_derivative(der)

    return trie


def _predict_from_trie(seq_int, trie, all_sequences_map):
    """
    Predict next element using prefix tree lookup.

    Strategy:
    1. Compute signature of test sequence
    2. Find training sequences whose signature has test signature as prefix
    3. The next element in the training signature (denormalized) is our prediction

    Args:
        seq_int: sequence as list of ints
        trie: PrefixTree instance
        all_sequences_map: dict mapping signature tuples to original (seq, next_element)

    Returns:
        predicted next element or None
    """
    if len(seq_int) < 3:
        return None

    der = list(seq_int)
    last_elements = [seq_int[-1]]

    for diff_depth in range(4):
        if len(der) < 3:
            return None

        seq = list(der)

        # Skip leading zeros
        while len(seq) > 0 and seq[0] == 0:
            seq = seq[1:]

        if len(seq) < 3:
            der = _find_derivative(der)
            if diff_depth > 0:
                last_elements.append(der[-1] if der else 0)
            continue

        # Compute signature
        signature = _find_signature(seq)

        # Find sequences that have this as prefix and are longer
        candidates = trie.find_longer_sequences(signature)

        if candidates:
            # Sort by weight (higher = better) then by length (shorter extension = better)
            candidates.sort(key=lambda x: (-x[1], len(x[0])))

            for full_sig, weight in candidates:
                if len(full_sig) > len(signature):
                    # The next element in normalized form
                    next_normalized = full_sig[len(signature)]

                    # Denormalize: find the scale factor
                    nonzero_idx = -1
                    for i, v in enumerate(seq):
                        if v != 0:
                            nonzero_idx = i
                            break

                    if nonzero_idx >= 0 and signature[nonzero_idx] != 0:
                        scale = seq[nonzero_idx] // signature[nonzero_idx]
                        next_val = next_normalized * scale

                        # If we're in difference space, reconstruct
                        if diff_depth > 0:
                            result = next_val
                            for d in range(diff_depth - 1, -1, -1):
                                result = last_elements[d] + result
                            return int(result)
                        else:
                            return int(next_val)

        # Move to differences
        der = _find_derivative(der)
        if der:
            last_elements.append(der[-1])

    return None


# Keep old functions for backwards compatibility
def _build_prefix_tree(sequences_with_targets, max_diff_order=4, max_shifts=4):
    """Build prefix tree (legacy wrapper)."""
    all_seqs = [seq for seq, _ in sequences_with_targets if len(seq) >= 3]
    return _build_prefix_tree_v2(all_seqs)


def _find_next_from_trie(seq_int, trie, gcd_factor=1):
    """Legacy function - not used in v2."""
    return None


def _find_next_with_derivatives(seq_int, trie, max_diff_order=4):
    """Legacy wrapper - uses new implementation."""
    return _predict_from_trie(seq_int, trie, {})


def _build_direct_lookup_map(raw_train_df, sequence_column):
    """
    Build a hash map from sequence strings to their next element.

    This enables O(1) direct lookup when a test sequence exactly matches
    a training sequence (minus the last element).

    Returns:
        dict: Maps sequence string (without last element) to the last element
    """
    direct_map = {}

    for _, row in raw_train_df.iterrows():
        seq_str = row[sequence_column]
        try:
            seq_int = [int(x.strip()) for x in str(seq_str).split(",") if x.strip()]
            if len(seq_int) >= 2:
                # Key: sequence without last element
                # Value: last element (the target)
                key = ",".join(str(x) for x in seq_int[:-1])
                direct_map[key] = seq_int[-1]
        except Exception:
            continue

    return direct_map


def _detect_geometric_progression(seq_int):
    """
    Detect if sequence follows a geometric progression: a[n+1] = a[n] * r

    Returns predicted next element or None.
    """
    if len(seq_int) < 3:
        return None

    # Check ratios between consecutive elements
    ratios = []
    for i in range(1, len(seq_int)):
        if seq_int[i-1] == 0:
            return None
        ratio = seq_int[i] / seq_int[i-1]
        ratios.append(ratio)

    if not ratios:
        return None

    # Check if all ratios are approximately equal
    avg_ratio = sum(ratios) / len(ratios)
    if avg_ratio == 0:
        return None

    for r in ratios:
        if abs(r - avg_ratio) > 1e-6 * abs(avg_ratio):
            return None

    # All ratios match - geometric progression
    next_val = seq_int[-1] * avg_ratio
    return int(round(next_val))


def _detect_arithmetic_progression(seq_int):
    """
    Detect if sequence follows an arithmetic progression: a[n+1] = a[n] + d

    Returns predicted next element or None.
    """
    if len(seq_int) < 3:
        return None

    diffs = [seq_int[i] - seq_int[i-1] for i in range(1, len(seq_int))]

    # Check if all differences are equal
    if len(set(diffs)) == 1:
        return seq_int[-1] + diffs[0]

    return None


@contract(
    inputs={
        "test_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "model": {"format": "pickle", "required": True, "schema": {"type": "artifact"}},
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "raw_train_data": {"format": "csv", "required": False, "schema": {"type": "tabular"}},
    },
    outputs={
        "predictions": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Hybrid prediction: exact math methods + direct lookup + prefix lookup + ML fallback",
    tags=["prediction", "sequence", "hybrid"],
    version="3.0.0",
)
def predict_sequence_hybrid(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    sequence_column: str = "Sequence",
    id_column: str = "Id",
    prediction_column: str = "Last",
    use_prefix_lookup: bool = True,
) -> str:
    """
    Hybrid prediction for integer sequences.

    Priority (ordered by reliability):
    0. Direct lookup - exact match of test sequence in training data (highest accuracy)
    1. Recurrence relation detection (order 2-6) - exact mathematical prediction
    2. Constant difference detection (polynomial sequences) - exact prediction
    3. Geometric progression detection - exact prediction
    4. Arithmetic progression detection - exact prediction
    5. Prefix lookup in training data (balzac solution) - signature-based matching
    6. LightGBM model prediction (inverse log transform + round) - ML fallback

    G4 Compliance: Column names injected via params.
    """
    test_df = _load_data(inputs["test_data"])
    train_df = _load_data(inputs["train_data"])

    # Load raw training data for direct lookup and prefix lookup
    raw_train_path = inputs.get("raw_train_data")
    if raw_train_path and os.path.exists(raw_train_path):
        raw_train_df = _load_data(raw_train_path)
    else:
        # Try to load from the standard location
        base_dir = os.path.dirname(os.path.dirname(inputs["test_data"]))
        raw_path = os.path.join(base_dir, "datasets", "train.csv")
        if os.path.exists(raw_path):
            raw_train_df = _load_data(raw_path)
        else:
            raw_train_df = None

    with open(inputs["model"], "rb") as f:
        model = pickle.load(f)

    # Get feature columns from train data (exclude Id and target)
    feature_cols = [c for c in train_df.columns if c != id_column and c != prediction_column]

    # Build direct lookup map (sequence string -> next element)
    direct_map = {}
    if raw_train_df is not None:
        print("Building direct sequence lookup map...")
        direct_map = _build_direct_lookup_map(raw_train_df, sequence_column)
        print(f"Direct lookup map built with {len(direct_map)} entries")

    # Build prefix tree from training data + test data for lookup
    prefix_trie = None
    if use_prefix_lookup and raw_train_df is not None:
        print("Building prefix tree from training data...")
        train_seqs_with_targets = []

        for _, row in raw_train_df.iterrows():
            seq_str = row[sequence_column]
            try:
                seq_int = [int(x.strip()) for x in str(seq_str).split(",") if x.strip()]
                if len(seq_int) >= 2:
                    # The last element is the target we want to predict
                    # Store sequence WITHOUT last element, and the last element as target
                    train_seqs_with_targets.append((seq_int[:-1], seq_int[-1]))
                    # Also store full sequence (helps with longer matches)
                    train_seqs_with_targets.append((seq_int, seq_int[-1]))
            except Exception:
                continue

        # Also add test sequences to the trie (for cross-matching)
        for _, row in test_df.iterrows():
            seq_str = row[sequence_column]
            try:
                seq_int = [int(x.strip()) for x in str(seq_str).split(",") if x.strip()]
                if len(seq_int) >= 3:
                    # We don't know the target, but we can store for prefix matching
                    train_seqs_with_targets.append((seq_int, 0))  # Placeholder
            except Exception:
                continue

        prefix_trie = _build_prefix_tree(train_seqs_with_targets)
        print(f"Prefix tree built with {prefix_trie.count} entries")

    ids = []
    predictions = []
    method_counts = {
        "direct_lookup": 0,
        "recurrence": 0,
        "constant_diff": 0,
        "geometric": 0,
        "arithmetic": 0,
        "prefix_lookup": 0,
        "ml": 0
    }

    for _, row in test_df.iterrows():
        seq_id = row[id_column] if id_column in test_df.columns else _
        seq_str = row[sequence_column]

        # Parse sequence as integers
        try:
            seq_int = [int(x.strip()) for x in str(seq_str).split(",") if x.strip()]
        except Exception:
            seq_int = [0]

        ids.append(seq_id)

        # Method 0: Direct lookup - exact match in training data (HIGHEST PRIORITY)
        # This is the most reliable method - if the test sequence is an exact prefix
        # of a training sequence, we know the answer with 100% certainty
        if seq_str in direct_map:
            predictions.append(direct_map[seq_str])
            method_counts["direct_lookup"] += 1
            continue

        # Method 1: Recurrence relation detection (extended to order 6)
        pred = _detect_recurrence_and_predict(seq_int, max_order=6)
        if pred is not None:
            predictions.append(pred)
            method_counts["recurrence"] += 1
            continue

        # Method 2: Constant difference detection (polynomial sequences)
        pred = _detect_constant_diff_and_predict(seq_int, max_order=5)
        if pred is not None:
            predictions.append(pred)
            method_counts["constant_diff"] += 1
            continue

        # Method 3: Geometric progression detection
        pred = _detect_geometric_progression(seq_int)
        if pred is not None:
            predictions.append(pred)
            method_counts["geometric"] += 1
            continue

        # Method 4: Arithmetic progression detection
        pred = _detect_arithmetic_progression(seq_int)
        if pred is not None:
            predictions.append(pred)
            method_counts["arithmetic"] += 1
            continue

        # Method 5: Prefix lookup in training data (balzac solution)
        if prefix_trie is not None:
            pred = _find_next_with_derivatives(seq_int, prefix_trie)
            if pred is not None:
                predictions.append(int(pred))
                method_counts["prefix_lookup"] += 1
                continue

        # Method 6: ML model fallback
        # Build features for this sequence (same as extract_sequence_features)
        seq = [max(min(float(x), 1e15), -1e15) for x in seq_int]
        feat = _extract_single_sequence_features(seq, seq_int, prefix="seq_",
                                                  max_length=20)

        # Create a DataFrame row matching the training features
        feat_row = {}
        for col in feature_cols:
            feat_row[col] = feat.get(col, 0.0)
        feat_df = pd.DataFrame([feat_row])

        # Predict in log-space
        log_pred = model.predict(feat_df)[0]

        # Inverse signed log1p transform
        sign = 1.0 if log_pred >= 0 else -1.0
        abs_pred = min(abs(log_pred), 43.0)  # Cap to avoid int64 overflow
        original_pred = sign * np.expm1(abs_pred)
        original_pred = int(round(original_pred))

        predictions.append(original_pred)
        method_counts["ml"] += 1

    pred_df = pd.DataFrame({id_column: ids, prediction_column: predictions})
    _save_data(pred_df, outputs["predictions"])

    return (
        f"predict_sequence_hybrid: {len(predictions)} predictions - "
        f"direct_lookup={method_counts['direct_lookup']}, "
        f"recurrence={method_counts['recurrence']}, "
        f"constant_diff={method_counts['constant_diff']}, "
        f"geometric={method_counts['geometric']}, "
        f"arithmetic={method_counts['arithmetic']}, "
        f"prefix_lookup={method_counts['prefix_lookup']}, "
        f"ml={method_counts['ml']}"
    )


def _extract_single_sequence_features(seq, seq_int, prefix="seq_", max_length=20):
    """Extract features for a single sequence (used by hybrid predictor)."""
    feat = {}
    feat[f"{prefix}length"] = len(seq)

    for i in range(max_length):
        if i < len(seq):
            feat[f"{prefix}elem_{i}"] = float(_signed_log1p(seq[-(i + 1)]))
        else:
            feat[f"{prefix}elem_{i}"] = 0.0

    arr = np.array(seq, dtype=np.float64)

    if len(seq) >= 2:
        log_arr = _signed_log1p(arr)
        feat[f"{prefix}log_mean"] = float(np.mean(log_arr))
        feat[f"{prefix}log_std"] = float(np.std(log_arr))
        feat[f"{prefix}log_min"] = float(np.min(log_arr))
        feat[f"{prefix}log_max"] = float(np.max(log_arr))
        feat[f"{prefix}log_range"] = feat[f"{prefix}log_max"] - feat[f"{prefix}log_min"]
        feat[f"{prefix}log_last"] = float(log_arr[-1])

        diffs = np.diff(arr)
        log_diffs = _signed_log1p(diffs)
        feat[f"{prefix}diff_mean"] = float(np.mean(log_diffs))
        feat[f"{prefix}diff_std"] = float(np.std(log_diffs)) if len(log_diffs) > 1 else 0.0
        feat[f"{prefix}diff_last"] = float(log_diffs[-1]) if len(log_diffs) > 0 else 0.0

        if len(diffs) > 1:
            diffs2 = np.diff(diffs)
            log_diffs2 = _signed_log1p(diffs2)
            feat[f"{prefix}diff2_mean"] = float(np.mean(log_diffs2))
            feat[f"{prefix}diff2_std"] = float(np.std(log_diffs2)) if len(log_diffs2) > 1 else 0.0
            feat[f"{prefix}diff2_last"] = float(log_diffs2[-1]) if len(log_diffs2) > 0 else 0.0
        else:
            feat[f"{prefix}diff2_mean"] = 0.0
            feat[f"{prefix}diff2_std"] = 0.0
            feat[f"{prefix}diff2_last"] = 0.0

        if len(diffs) > 2:
            diffs3 = np.diff(np.diff(diffs))
            log_diffs3 = _signed_log1p(diffs3)
            feat[f"{prefix}diff3_mean"] = float(np.mean(log_diffs3))
        else:
            feat[f"{prefix}diff3_mean"] = 0.0

        feat[f"{prefix}is_const_diff1"] = int(len(diffs) >= 3 and np.std(diffs) < 1e-6)
        if len(diffs) > 1:
            diffs2_raw = np.diff(diffs)
            feat[f"{prefix}is_const_diff2"] = int(len(diffs2_raw) >= 3 and np.std(diffs2_raw) < 1e-6)
        else:
            feat[f"{prefix}is_const_diff2"] = 0

        if seq[-2] != 0:
            ratio = seq[-1] / seq[-2]
            feat[f"{prefix}ratio_last"] = float(_signed_log1p(ratio))
        else:
            feat[f"{prefix}ratio_last"] = 0.0

        ratios = []
        for j in range(1, min(len(seq), 10)):
            if seq[-(j + 1)] != 0:
                ratios.append(seq[-j] / seq[-(j + 1)])
        if ratios:
            feat[f"{prefix}ratio_mean"] = float(np.mean(ratios))
            feat[f"{prefix}ratio_std"] = float(np.std(ratios)) if len(ratios) > 1 else 0.0
            feat[f"{prefix}is_geometric"] = int(
                len(ratios) >= 3 and np.std(ratios) < 0.01 * (abs(np.mean(ratios)) + 1e-10))
        else:
            feat[f"{prefix}ratio_mean"] = 0.0
            feat[f"{prefix}ratio_std"] = 0.0
            feat[f"{prefix}is_geometric"] = 0

        feat[f"{prefix}is_increasing"] = int(all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1)))
        feat[f"{prefix}is_decreasing"] = int(all(arr[i] >= arr[i + 1] for i in range(len(arr) - 1)))

        signs = np.sign(arr)
        nonzero_signs = signs[signs != 0]
        feat[f"{prefix}sign_changes"] = int(np.sum(np.diff(nonzero_signs) != 0)) if len(nonzero_signs) > 1 else 0
        feat[f"{prefix}has_negatives"] = int(np.any(arr < 0))
        feat[f"{prefix}has_zeros"] = int(np.any(arr == 0))

        try:
            abs_seq = [abs(x) for x in seq_int if x != 0]
            if abs_seq:
                gcd_val = abs_seq[0]
                for v in abs_seq[1:]:
                    gcd_val = math.gcd(gcd_val, v)
                feat[f"{prefix}gcd"] = float(np.log1p(float(gcd_val)))
            else:
                feat[f"{prefix}gcd"] = 0.0
        except (ValueError, OverflowError):
            feat[f"{prefix}gcd"] = 0.0

        feat[f"{prefix}recurrence_order"] = 0
        for order in [2, 3, 4]:
            if _check_recurrence(seq_int, order):
                feat[f"{prefix}recurrence_order"] = order
                break
    else:
        val = seq[0] if seq else 0.0
        log_val = float(_signed_log1p(val))
        for key in ["log_mean", "log_std", "log_min", "log_max", "log_range", "log_last",
                     "diff_mean", "diff_std", "diff_last", "diff2_mean", "diff2_std",
                     "diff2_last", "diff3_mean", "ratio_last", "ratio_mean", "ratio_std"]:
            feat[f"{prefix}{key}"] = log_val if "mean" in key or "min" in key or "max" in key or "last" in key else 0.0
        for key in ["is_const_diff1", "is_const_diff2", "is_geometric", "is_increasing",
                     "is_decreasing", "sign_changes", "has_zeros", "recurrence_order"]:
            feat[f"{prefix}{key}"] = 0
        feat[f"{prefix}has_negatives"] = int(val < 0)
        feat[f"{prefix}gcd"] = 0.0

    return feat


# =============================================================================
# SERVICE REGISTRY (Required for pipeline_runner discovery)
# =============================================================================

SERVICE_REGISTRY = {
    "parse_sequence_target": parse_sequence_target,
    "extract_sequence_features": extract_sequence_features,
    "predict_sequence_hybrid": predict_sequence_hybrid,
    "split_data": split_data,
    "train_lightgbm_regressor": train_lightgbm_regressor,
    "predict_regressor": predict_regressor,
    "create_submission": create_submission,
}

# =============================================================================
# PIPELINE SPECIFICATION
# =============================================================================

PIPELINE_SPEC = {
    "name": "integer-sequence-learning",
    "description": "Hybrid pipeline for integer sequence prediction (OEIS). "
                   "Uses recurrence relation detection and constant-difference analysis "
                   "for exact predictions, with LightGBM ML fallback. "
                   "Signed log1p target transform, 47 enhanced features.",
    "version": "3.0.0",
    "problem_type": "regression",
    "target_column": "Last",
    "id_column": "Id",
    "steps": [
        # --- Train data preprocessing ---
        {
            "service": "parse_sequence_target",
            "inputs": {"data": "integer-sequence-learning/datasets/train.csv"},
            "outputs": {"data": "integer-sequence-learning/artifacts/train_01_parsed.csv"},
            "params": {
                "sequence_column": "Sequence",
                "target_column": "Last",
                "remove_last": True,
            },
            "module": "sequence_services",
        },
        {
            "service": "extract_sequence_features",
            "inputs": {"data": "integer-sequence-learning/artifacts/train_01_parsed.csv"},
            "outputs": {"data": "integer-sequence-learning/artifacts/train_02_features.csv"},
            "params": {
                "sequence_column": "Sequence",
                "target_column": "Last",
                "id_column": "Id",
                "max_sequence_length": 20,
                "prefix": "seq_",
            },
            "module": "sequence_services",
        },
        # --- Train/validation split ---
        {
            "service": "split_data",
            "inputs": {"data": "integer-sequence-learning/artifacts/train_02_features.csv"},
            "outputs": {
                "train_data": "integer-sequence-learning/artifacts/train_split.csv",
                "valid_data": "integer-sequence-learning/artifacts/valid_split.csv",
            },
            "params": {"test_size": 0.2, "random_state": 42},
            "module": "preprocessing_services",
        },
        # --- Model training ---
        {
            "service": "train_lightgbm_regressor",
            "inputs": {
                "train_data": "integer-sequence-learning/artifacts/train_split.csv",
                "valid_data": "integer-sequence-learning/artifacts/valid_split.csv",
            },
            "outputs": {
                "model": "integer-sequence-learning/artifacts/model.pkl",
                "metrics": "integer-sequence-learning/artifacts/metrics.json",
            },
            "params": {
                "label_column": "Last",
                "id_column": "Id",
                "n_estimators": 500,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "max_depth": -1,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "objective": "mae",
            },
            "module": "regression_services",
        },
        # --- Hybrid prediction: exact math + ML fallback ---
        {
            "service": "predict_sequence_hybrid",
            "inputs": {
                "test_data": "integer-sequence-learning/datasets/test.csv",
                "model": "integer-sequence-learning/artifacts/model.pkl",
                "train_data": "integer-sequence-learning/artifacts/train_02_features.csv",
            },
            "outputs": {
                "predictions": "integer-sequence-learning/artifacts/predictions.csv",
            },
            "params": {
                "sequence_column": "Sequence",
                "id_column": "Id",
                "prediction_column": "Last",
            },
            "module": "sequence_services",
        },
        # --- Create submission ---
        {
            "service": "create_submission",
            "inputs": {
                "predictions": "integer-sequence-learning/artifacts/predictions.csv",
            },
            "outputs": {
                "submission": "integer-sequence-learning/submission.csv",
            },
            "params": {
                "id_column": "Id",
                "prediction_column": "Last",
            },
            "module": "preprocessing_services",
        },
    ],
}
