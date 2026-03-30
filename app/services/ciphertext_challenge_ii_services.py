"""
Ciphertext Challenge II - SLEGO Services
==========================================
Competition: https://www.kaggle.com/competitions/ciphertext-challenge-ii
Problem Type: Text Matching (Cipher Decryption)
Target: index (plaintext index matching the ciphertext)
ID Column: ciphertext_id

This competition involves decrypting ciphertexts at various difficulty levels
and matching them to their corresponding plaintexts from the training data.

Top solution approach (frequency analysis + substitution cipher):
1. Build character frequency mapping between plaintext and ciphertext
2. Refine mapping by matching decrypted ciphertexts to known plaintexts
3. Decrypt all ciphertexts and match to plaintexts
4. Create submission with matched indices

These services are reusable for any substitution cipher competition
(e.g., ciphertext-challenge-ii, ciphertext-challenge-iii).
"""

import os
import sys
import json
import math
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slego_contract import contract


# =============================================================================
# CIPHER DECRYPTION SERVICES (Reusable for substitution cipher challenges)
# =============================================================================

@contract(
    inputs={
        "plaintext_data": {"format": "csv", "required": True},
        "ciphertext_data": {"format": "csv", "required": True},
    },
    outputs={
        "mapping": {"format": "pickle", "schema": {"type": "artifact", "artifact_type": "cipher_mapping"}},
    },
    description="Build character substitution mapping using frequency analysis between plaintext and ciphertext corpora",
    tags=["cryptography", "frequency-analysis", "substitution-cipher", "generic"],
    version="1.0.0",
)
def build_substitution_mapping(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    text_column: str = "text",
    cipher_column: str = "ciphertext",
    difficulty_column: str = "difficulty",
    difficulty_level: int = 1,
    n_chars: int = 85,
) -> str:
    """
    Build initial character substitution mapping using frequency analysis.

    Counts character frequencies in both the plaintext corpus and the
    ciphertext corpus (filtered by difficulty), then maps characters
    by frequency rank (most common cipher char -> most common plain char).

    Works with any substitution cipher competition.

    Args:
        text_column: Column containing plaintext
        cipher_column: Column containing ciphertext
        difficulty_column: Column indicating cipher difficulty level
        difficulty_level: Which difficulty to process
        n_chars: Number of most-common characters to map
    """
    plaintext_df = pd.read_csv(inputs["plaintext_data"])
    test_df = pd.read_csv(inputs["ciphertext_data"])

    # Filter ciphertexts by difficulty
    cipher_df = test_df[test_df[difficulty_column] == difficulty_level]

    # Count character frequencies in plaintext corpus
    plain_corpus = ''.join(plaintext_df[text_column].astype(str).values)
    plain_chars = ''.join([c for c, _ in Counter(plain_corpus).most_common(n_chars)])

    # Count character frequencies in ciphertext corpus
    cipher_corpus = ''.join(cipher_df[cipher_column].astype(str).values)
    cipher_chars = ''.join([c for c, _ in Counter(cipher_corpus).most_common(n_chars)])

    # Map by frequency rank: most common cipher char -> most common plain char
    map_len = min(len(cipher_chars), len(plain_chars))
    decrypt_map = {}
    for c, p in zip(cipher_chars[:map_len], plain_chars[:map_len]):
        decrypt_map[c] = p

    mapping = {
        "decrypt_map": decrypt_map,
        "cipher_chars": cipher_chars,
        "plain_chars": plain_chars,
        "difficulty_level": difficulty_level,
    }

    os.makedirs(os.path.dirname(outputs["mapping"]) or ".", exist_ok=True)
    with open(outputs["mapping"], "wb") as f:
        pickle.dump(mapping, f)

    return f"build_substitution_mapping: mapped {len(decrypt_map)} chars for difficulty {difficulty_level}"


@contract(
    inputs={
        "plaintext_data": {"format": "csv", "required": True},
        "ciphertext_data": {"format": "csv", "required": True},
        "mapping": {"format": "pickle", "required": True},
    },
    outputs={
        "mapping": {"format": "pickle", "schema": {"type": "artifact", "artifact_type": "cipher_mapping"}},
    },
    description="Refine substitution mapping by matching decrypted ciphertexts to known plaintexts and correcting errors",
    tags=["cryptography", "substitution-cipher", "refinement", "generic"],
    version="1.0.0",
)
def refine_substitution_mapping(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    text_column: str = "text",
    cipher_column: str = "ciphertext",
    difficulty_column: str = "difficulty",
    difficulty_level: int = 1,
    match_threshold: float = 0.9,
    padding_margin: int = 50,
) -> str:
    """
    Refine the initial frequency-based mapping by:
    1. Finding a small subset of ciphertexts covering all cipher characters
    2. Decrypting them with the current mapping
    3. Matching to known plaintexts (>threshold accuracy)
    4. Noting and correcting character mistakes

    Based on approach from top Kaggle solution by lemonkoala.

    Args:
        text_column: Column containing plaintext
        cipher_column: Column containing ciphertext
        difficulty_column: Column indicating difficulty level
        difficulty_level: Which difficulty to process
        match_threshold: Minimum character accuracy to confirm a match
        padding_margin: Characters to skip from cipher edges to avoid padding
    """
    plaintext_df = pd.read_csv(inputs["plaintext_data"])
    test_df = pd.read_csv(inputs["ciphertext_data"])

    with open(inputs["mapping"], "rb") as f:
        mapping = pickle.load(f)

    cipher_df = test_df[test_df[difficulty_column] == difficulty_level].reset_index(drop=True)
    if len(cipher_df) == 0:
        # No ciphertexts for this difficulty; pass mapping through unchanged
        os.makedirs(os.path.dirname(outputs["mapping"]) or ".", exist_ok=True)
        with open(outputs["mapping"], "wb") as f:
            pickle.dump(mapping, f)
        return f"refine_substitution_mapping: no ciphertexts for difficulty {difficulty_level}"

    decrypt_map = dict(mapping["decrypt_map"])
    encrypt_map = {v: k for k, v in decrypt_map.items()}

    # Build translation table for decryption
    trans_from = ''.join(decrypt_map.keys())
    trans_to = ''.join(decrypt_map.values())
    trans_table = str.maketrans(trans_from, trans_to)

    # Find minimal subset of ciphertexts covering all cipher characters
    all_cipher_chars = set(decrypt_map.keys())
    cipher_texts = cipher_df[cipher_column].astype(str).values

    alphabet_per_cipher = []
    for text in cipher_texts:
        inner = text[padding_margin:-padding_margin] if len(text) > 2 * padding_margin else text
        alphabet_per_cipher.append(set(inner) & all_cipher_chars)

    # Greedy set-cover: pick ciphertexts that add the most uncovered characters
    subset_indices = [0]
    covered = set(alphabet_per_cipher[0]) if len(alphabet_per_cipher) > 0 else set()

    while covered != all_cipher_chars:
        uncovered = all_cipher_chars - covered
        found = False
        for idx, chars in enumerate(alphabet_per_cipher):
            if chars & uncovered:
                subset_indices.append(idx)
                covered |= chars
                found = True
                break
        if not found:
            break

    # Prepare plaintext lookup with padding info
    plaintext_df["_length"] = plaintext_df[text_column].astype(str).str.len()
    plaintext_df["_padded_length"] = (np.ceil(plaintext_df["_length"] / 100) * 100).astype(int)

    # Match subset ciphertexts to plaintexts, collect corrections
    all_corrects = set()
    # Map cipher_char -> correct_plain_char (direct, no lookup needed)
    cipher_corrections = {}

    for cipher_idx in subset_indices:
        cipher_row = cipher_df.iloc[cipher_idx]
        cipher_text = str(cipher_row[cipher_column])
        decrypted = cipher_text.translate(trans_table)
        cipher_len = len(cipher_text)

        candidates = plaintext_df[plaintext_df["_padded_length"] == cipher_len]

        for _, pt_row in candidates.iterrows():
            pt_text = str(pt_row[text_column])
            pt_len = len(pt_text)
            padding_left = math.floor((cipher_len - pt_len) / 2)
            unpadded = decrypted[padding_left:padding_left + pt_len]
            cipher_region = cipher_text[padding_left:padding_left + pt_len]

            if len(unpadded) != pt_len:
                continue

            # Calculate character-level accuracy
            correct_count = sum(1 for d, p in zip(unpadded, pt_text) if d == p)
            score = correct_count / pt_len if pt_len > 0 else 0

            if score >= match_threshold:
                for d, p, c in zip(unpadded, pt_text, cipher_region):
                    if d == p:
                        all_corrects.add(d)
                    else:
                        # Direct mapping: cipher character c should decrypt to p
                        cipher_corrections[c] = p
                break

    # Apply all corrections as a batch (avoids cascading update bugs)
    corrections_made = 0
    for cipher_char, correct_plain in cipher_corrections.items():
        if decrypt_map.get(cipher_char) != correct_plain:
            decrypt_map[cipher_char] = correct_plain
            corrections_made += 1

    refined_mapping = {
        "decrypt_map": decrypt_map,
        "cipher_chars": ''.join(decrypt_map.keys()),
        "plain_chars": ''.join(decrypt_map.values()),
        "difficulty_level": difficulty_level,
        "corrections_made": corrections_made,
    }

    os.makedirs(os.path.dirname(outputs["mapping"]) or ".", exist_ok=True)
    with open(outputs["mapping"], "wb") as f:
        pickle.dump(refined_mapping, f)

    return (f"refine_substitution_mapping: verified {len(all_corrects)} chars, "
            f"corrected {corrections_made} using {len(subset_indices)} samples")


@contract(
    inputs={
        "plaintext_data": {"format": "csv", "required": True},
        "ciphertext_data": {"format": "csv", "required": True},
        "mapping": {"format": "pickle", "required": True},
    },
    outputs={
        "matches": {"format": "csv"},
    },
    description="Decrypt ciphertexts using refined substitution mapping and match to known plaintexts",
    tags=["cryptography", "substitution-cipher", "matching", "generic"],
    version="1.0.0",
)
def decrypt_and_match_ciphertexts(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    text_column: str = "text",
    cipher_column: str = "ciphertext",
    id_column: str = "ciphertext_id",
    index_column: str = "index",
    difficulty_column: str = "difficulty",
    difficulty_level: int = 1,
) -> str:
    """
    Decrypt all ciphertexts for the given difficulty level and match each
    to its corresponding plaintext.

    Uses length-based candidate filtering for efficiency:
    - Ciphertext length = padded_length of matching plaintext (padded to multiples of 100)
    - Only compares ciphertexts to plaintexts with matching padded_length
    - Checks if plaintext is a substring of the decrypted ciphertext

    Args:
        text_column: Column containing plaintext
        cipher_column: Column containing ciphertext
        id_column: Column containing ciphertext identifier
        index_column: Column containing plaintext index (target)
        difficulty_column: Column indicating difficulty level
        difficulty_level: Which difficulty to process
    """
    plaintext_df = pd.read_csv(inputs["plaintext_data"])
    test_df = pd.read_csv(inputs["ciphertext_data"])

    with open(inputs["mapping"], "rb") as f:
        mapping = pickle.load(f)

    cipher_df = test_df[test_df[difficulty_column] == difficulty_level].reset_index(drop=True)
    if len(cipher_df) == 0:
        empty_df = pd.DataFrame(columns=[id_column, index_column])
        os.makedirs(os.path.dirname(outputs["matches"]) or ".", exist_ok=True)
        empty_df.to_csv(outputs["matches"], index=False)
        return f"decrypt_and_match_ciphertexts: no ciphertexts for difficulty {difficulty_level}"

    decrypt_map = mapping["decrypt_map"]
    trans_table = str.maketrans(
        ''.join(decrypt_map.keys()),
        ''.join(decrypt_map.values())
    )

    # Prepare plaintext lookup grouped by padded_length
    plaintext_df["_length"] = plaintext_df[text_column].astype(str).str.len()
    plaintext_df["_padded_length"] = (np.ceil(plaintext_df["_length"] / 100) * 100).astype(int)

    length_groups = {}
    for padded_len, group in plaintext_df.groupby("_padded_length"):
        length_groups[int(padded_len)] = list(zip(
            group[text_column].astype(str).values,
            group[index_column].values
        ))

    # Decrypt and match each ciphertext
    matches = []
    unmatched = 0

    for _, row in cipher_df.iterrows():
        cipher_text = str(row[cipher_column])
        decrypted = cipher_text.translate(trans_table)
        cipher_len = len(cipher_text)

        candidates = length_groups.get(cipher_len, [])
        found = False

        for pt_text, pt_index in candidates:
            if pt_text in decrypted:
                matches.append({
                    id_column: row[id_column],
                    index_column: int(pt_index),
                })
                found = True
                break

        if not found:
            unmatched += 1

    matches_df = pd.DataFrame(matches)
    os.makedirs(os.path.dirname(outputs["matches"]) or ".", exist_ok=True)
    matches_df.to_csv(outputs["matches"], index=False)

    return (f"decrypt_and_match_ciphertexts: matched {len(matches)}/{len(cipher_df)} "
            f"({unmatched} unmatched) for difficulty {difficulty_level}")


@contract(
    inputs={
        "ciphertext_data": {"format": "csv", "required": True},
    },
    outputs={
        "submission": {"format": "csv"},
    },
    description="Assemble final cipher submission from matched results and test data IDs",
    tags=["submission", "cryptography", "generic"],
    version="1.0.0",
)
def create_cipher_submission(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "ciphertext_id",
    target_column: str = "index",
    default_value: int = 0,
) -> str:
    """
    Combine matched cipher results into a submission file.

    Uses ciphertext_data (test.csv) to get all ciphertext IDs, then
    reads match files from any input slot starting with 'matches' and
    assigns matched indices. Unmatched entries use the default_value.

    Args:
        id_column: Column name for ciphertext identifier
        target_column: Column name for the prediction target (plaintext index)
        default_value: Value for ciphertexts that could not be matched
    """
    test_df = pd.read_csv(inputs["ciphertext_data"])
    sub = pd.DataFrame({
        id_column: test_df[id_column],
        target_column: default_value,
    })

    # Collect all match files (input keys starting with "matches")
    all_matches = []
    for key, path in inputs.items():
        if key.startswith("matches") and os.path.exists(path):
            try:
                match_df = pd.read_csv(path)
                if len(match_df) > 0:
                    all_matches.append(match_df)
            except pd.errors.EmptyDataError:
                # Skip empty files
                pass

    if all_matches:
        combined = pd.concat(all_matches, ignore_index=True)
        combined = combined.drop_duplicates(subset=[id_column], keep="first")
        match_map = dict(zip(combined[id_column], combined[target_column]))
        sub[target_column] = sub[id_column].map(
            lambda x: match_map.get(x, default_value)
        )

    sub[target_column] = sub[target_column].astype(int)

    os.makedirs(os.path.dirname(outputs["submission"]) or ".", exist_ok=True)
    sub[[id_column, target_column]].to_csv(outputs["submission"], index=False)

    matched_count = sum(1 for v in sub[target_column] if v != default_value)
    total = len(sub)
    return (f"create_cipher_submission: {matched_count}/{total} matched "
            f"({matched_count/total*100:.1f}%)")


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "build_substitution_mapping": build_substitution_mapping,
    "refine_substitution_mapping": refine_substitution_mapping,
    "decrypt_and_match_ciphertexts": decrypt_and_match_ciphertexts,
    "create_cipher_submission": create_cipher_submission,
}