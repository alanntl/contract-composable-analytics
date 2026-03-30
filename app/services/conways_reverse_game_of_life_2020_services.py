"""
Conway's Reverse Game of Life 2020 - SLEGO Services
====================================================
Competition: https://www.kaggle.com/competitions/conways-reverse-game-of-life-2020
Problem Type: Combinatorial Optimization (Reverse Cellular Automaton)
Target: start_0..start_624 (625 binary cells on 25x25 toroidal grid)
ID Column: id

This is NOT a standard ML problem. Given the end state of Conway's Game of Life
after `delta` steps on a 25x25 toroidal grid, find a starting configuration that
produces that end state.

Top solution approaches (from Kaggle notebooks):
1. SAT solver + simulated annealing (C++/Kissat) - #2 prize solution
2. Genetic algorithm on GPU (PyTorch) - top 10 solution
3. This implementation: Genetic algorithm + hill climbing on CPU (NumPy)

Competition-specific services:
- solve_conway_reverse: GA-based reverse Game of Life solver
- validate_conway_solution: Validate solution quality using training data
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slego_contract import contract

# =============================================================================
# IMPORTS FROM COMMON MODULES
# =============================================================================
try:
    from services.io_utils import load_data as _load_data, save_data as _save_data
except ImportError:
    from io_utils import load_data as _load_data, save_data as _save_data


# =============================================================================
# CONSTANTS
# =============================================================================
N = 25          # Grid dimension
N_CELLS = N * N  # 625 total cells


# =============================================================================
# GAME OF LIFE SIMULATION (Vectorized NumPy)
# =============================================================================

def _count_neighbors(grids):
    """Count live neighbors for each cell using toroidal (wrap-around) boundaries.

    Uses numpy roll for fully vectorized batch computation.

    Args:
        grids: numpy array of shape (..., 25, 25) with binary values (0/1)

    Returns:
        Neighbor counts array of same shape, values 0-8
    """
    n = np.zeros_like(grids, dtype=np.int8)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            n = n + np.roll(np.roll(grids, dx, axis=-2), dy, axis=-1)
    return n


def _forward_step(grids):
    """Apply one Game of Life step to a batch of grids.

    Rules:
    - Live cell with 2-3 neighbors survives (stasis)
    - Dead cell with exactly 3 neighbors becomes alive (reproduction)
    - All other cells die (overpopulation/underpopulation)

    Args:
        grids: shape (..., 25, 25) int8 array

    Returns:
        Next state, same shape
    """
    neighbors = _count_neighbors(grids)
    return ((neighbors == 3) | ((grids == 1) & (neighbors == 2))).astype(np.int8)


def _forward(grids, delta):
    """Apply delta Game of Life steps to a batch of grids.

    Args:
        grids: shape (..., 25, 25) int8 array
        delta: number of forward steps (1-5)

    Returns:
        State after delta steps, same shape
    """
    for _ in range(delta):
        grids = _forward_step(grids)
    return grids


# =============================================================================
# GENETIC ALGORITHM SOLVER
# =============================================================================

def _solve_puzzle(target, delta, rng, pop_size=200, n_gen=20, n_best=20,
                  n_elite=5, mutation_rate=0.003, alive_rate=0.2,
                  n_refine_flips=300):
    """Solve one Conway's reverse puzzle using genetic algorithm + hill climbing.

    Approach inspired by top Kaggle solutions:
    1. Initialize population with random grids + strategic seeds
    2. Evaluate fitness by forward-simulating and comparing to target
    3. Select best, crossover, mutate → new generation
    4. Refine best solution with hill climbing (random cell flips)

    Args:
        target: (25, 25) int8 array - the end state to reverse
        delta: number of Game of Life steps to reverse
        rng: numpy RandomState for reproducibility
        pop_size: population size for genetic algorithm
        n_gen: number of generations
        n_best: number of best individuals to keep as parents
        n_elite: number of elites to preserve unchanged
        mutation_rate: probability of flipping each cell
        alive_rate: fraction of alive cells in random initialization
        n_refine_flips: number of random flip attempts for hill climbing

    Returns:
        (solution, error_count) tuple
    """
    # --- Initialize population ---
    pop = (rng.random((pop_size, N, N)) > (1 - alive_rate)).astype(np.int8)

    # Strategic seeds (from solution notebook insights)
    pop[0] = target.copy()                                     # End state itself
    pop[1] = np.zeros((N, N), dtype=np.int8)                   # Empty board
    if pop_size > 2:
        pop[2] = _forward_step(target.copy()[np.newaxis])[0]   # Forward(target)
    if pop_size > 3:
        # Checkerboard pattern
        checker = np.zeros((N, N), dtype=np.int8)
        checker[::2, ::2] = 1
        checker[1::2, 1::2] = 1
        pop[3] = checker

    best_solution = target.copy()
    best_errors = N_CELLS

    # --- Genetic algorithm loop ---
    for gen in range(n_gen):
        # Evaluate fitness (fully vectorized batch forward simulation)
        results_fwd = _forward(pop, delta)
        target_exp = np.broadcast_to(target, pop.shape)
        errors = np.sum(results_fwd != target_exp, axis=(1, 2))

        # Track best
        gen_best = np.argmin(errors)
        if errors[gen_best] < best_errors:
            best_errors = int(errors[gen_best])
            best_solution = pop[gen_best].copy()
            if best_errors == 0:
                return best_solution, 0

        # Select best parents
        best_idx = np.argsort(errors)[:n_best]
        parents = pop[best_idx]

        # Crossover: random mask combining two parents
        dad_idx = rng.randint(0, n_best, size=pop_size)
        mom_idx = rng.randint(0, n_best, size=pop_size)
        mask = rng.random((pop_size, N, N)) > 0.5
        new_pop = np.where(mask, parents[dad_idx], parents[mom_idx]).astype(np.int8)

        # Mutation: randomly flip cells
        mutations = rng.random(new_pop.shape) < mutation_rate
        new_pop = np.where(mutations, 1 - new_pop, new_pop).astype(np.int8)

        # Elitism: preserve best individuals unchanged
        new_pop[:n_elite] = parents[:n_elite]
        pop = new_pop

    # --- Hill climbing refinement ---
    if best_errors > 0 and n_refine_flips > 0:
        current = best_solution.copy()
        current_errors = best_errors

        for _ in range(n_refine_flips):
            i, j = rng.randint(0, N), rng.randint(0, N)
            current[i, j] = 1 - current[i, j]

            fwd = _forward(current[np.newaxis], delta)[0]
            new_errors = int(np.sum(fwd != target))

            if new_errors <= current_errors:
                current_errors = new_errors
                if current_errors < best_errors:
                    best_errors = current_errors
                    best_solution = current.copy()
                if current_errors == 0:
                    break
            else:
                current[i, j] = 1 - current[i, j]  # Revert

    return best_solution, best_errors


def _solve_batch(args):
    """Solve a batch of puzzles. Designed for use with ProcessPoolExecutor.

    Args:
        args: tuple of (targets, deltas, puzzle_ids, params_dict)

    Returns:
        List of (puzzle_id, solution_flat, error_count) tuples
    """
    targets, deltas, puzzle_ids, params = args
    rng = np.random.RandomState(params.get('random_state', 42))

    pop_size = params.get('population_size', 200)
    n_gen = params.get('n_generations', 20)
    n_best = max(5, int(pop_size * params.get('n_best_ratio', 0.1)))
    n_elite = params.get('n_elite', 5)
    mutation_rate = params.get('mutation_rate', 0.003)
    alive_rate = params.get('random_alive_rate', 0.2)
    n_refine = params.get('n_refine_flips', 300)

    results = []
    for idx in range(len(targets)):
        sol, err = _solve_puzzle(
            targets[idx], int(deltas[idx]), rng,
            pop_size=pop_size, n_gen=n_gen, n_best=n_best,
            n_elite=n_elite, mutation_rate=mutation_rate,
            alive_rate=alive_rate, n_refine_flips=n_refine,
        )
        results.append((int(puzzle_ids[idx]), sol.flatten(), err))
    return results


# =============================================================================
# SLEGO SERVICES
# =============================================================================

@contract(
    inputs={
        "test_data": {"format": "csv", "required": True},
        "sample_submission": {"format": "csv", "required": False},
    },
    outputs={
        "submission": {"format": "csv"},
        "metrics": {"format": "json"},
    },
    description="Solve Conway's Reverse Game of Life using genetic algorithm with hill climbing",
    tags=["combinatorial-optimization", "simulation", "genetic-algorithm", "conway", "cellular-automaton"],
    version="1.0.0",
)
def solve_conway_reverse(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    n_generations: int = 20,
    population_size: int = 200,
    mutation_rate: float = 0.003,
    n_best_ratio: float = 0.1,
    n_elite: int = 5,
    random_alive_rate: float = 0.2,
    n_refine_flips: int = 300,
    n_workers: int = 1,
    batch_size: int = 500,
    random_state: int = 42,
) -> str:
    """
    Solve reverse Conway's Game of Life puzzles using genetic algorithm.

    For each test puzzle (end state + delta), finds a starting configuration
    that, after applying Game of Life rules 'delta' times on a 25x25 toroidal
    grid, produces the given end state.

    Approach (inspired by top Kaggle solutions):
    1. Genetic algorithm with population-based search
    2. Strategic initialization (end state, zeros, forward-of-end, checkerboard)
    3. Hill climbing refinement with random cell flips

    G1 Compliance: Generic reverse cellular automaton solver
    G3 Compliance: Reproducible with random_state parameter
    G4 Compliance: All hyperparameters exposed as parameters

    Parameters:
        n_generations: GA generations per puzzle
        population_size: GA population size
        mutation_rate: Probability of flipping each cell during mutation
        n_best_ratio: Fraction of population to keep as parents
        n_elite: Number of best individuals preserved unchanged
        random_alive_rate: Fraction of alive cells in random initialization
        n_refine_flips: Number of hill climbing flip attempts after GA
        n_workers: Number of parallel workers (1 = sequential)
        batch_size: Puzzles per batch for parallel processing
        random_state: Random seed for reproducibility
    """
    start_time = time.time()

    # Load test data
    test_df = _load_data(inputs["test_data"])
    n_puzzles = len(test_df)
    print(f"  Loaded {n_puzzles} puzzles")

    # Extract puzzle data
    ids = test_df["id"].values
    deltas_arr = test_df["delta"].values
    stop_cols = [f"stop_{i}" for i in range(N_CELLS)]
    grids = test_df[stop_cols].values.reshape(-1, N, N).astype(np.int8)

    # Solver parameters
    params = {
        'n_generations': n_generations,
        'population_size': population_size,
        'mutation_rate': mutation_rate,
        'n_best_ratio': n_best_ratio,
        'n_elite': n_elite,
        'random_alive_rate': random_alive_rate,
        'n_refine_flips': n_refine_flips,
        'random_state': random_state,
    }

    # Create batches
    n_batches = max(1, (n_puzzles + batch_size - 1) // batch_size)
    batches = []
    for b in range(n_batches):
        s = b * batch_size
        e = min((b + 1) * batch_size, n_puzzles)
        bp = params.copy()
        bp['random_state'] = random_state + b
        batches.append((grids[s:e], deltas_arr[s:e], ids[s:e], bp))

    # Process batches
    all_results = []
    total_errors = 0
    perfect = 0

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_solve_batch, batch): i
                       for i, batch in enumerate(batches)}
            completed = 0
            for future in as_completed(futures):
                batch_results = future.result()
                all_results.extend(batch_results)
                for _, _, err in batch_results:
                    total_errors += err
                    if err == 0:
                        perfect += 1
                completed += 1
                elapsed = time.time() - start_time
                pct = completed / n_batches * 100
                print(f"  [{completed}/{n_batches}] {pct:.0f}% done, "
                      f"{perfect} perfect, {elapsed:.0f}s elapsed")
    else:
        for i, batch in enumerate(batches):
            batch_results = _solve_batch(batch)
            all_results.extend(batch_results)
            for _, _, err in batch_results:
                total_errors += err
                if err == 0:
                    perfect += 1
            elapsed = time.time() - start_time
            pct = (i + 1) / n_batches * 100
            print(f"  [{i+1}/{n_batches}] {pct:.0f}% done, "
                  f"{perfect} perfect, {elapsed:.0f}s elapsed")

    # Sort by puzzle ID and create submission DataFrame
    all_results.sort(key=lambda x: x[0])

    start_cols = [f"start_{i}" for i in range(N_CELLS)]
    submission_data = {"id": [r[0] for r in all_results]}
    solutions = np.array([r[1] for r in all_results])
    for i in range(N_CELLS):
        submission_data[start_cols[i]] = solutions[:, i].astype(int)

    submission_df = pd.DataFrame(submission_data)

    # Save submission
    os.makedirs(os.path.dirname(outputs["submission"]) or ".", exist_ok=True)
    submission_df.to_csv(outputs["submission"], index=False)

    # Save metrics
    elapsed = time.time() - start_time
    error_rate = total_errors / (n_puzzles * N_CELLS) if n_puzzles > 0 else 0.0

    metrics = {
        "competition": "conways-reverse-game-of-life-2020",
        "total_puzzles": n_puzzles,
        "perfect_solutions": perfect,
        "perfect_rate": round(perfect / n_puzzles, 4) if n_puzzles > 0 else 0.0,
        "total_cell_errors": int(total_errors),
        "error_rate": round(error_rate, 6),
        "elapsed_seconds": round(elapsed, 1),
        "params": params,
    }

    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return (f"solve_conway_reverse: {perfect}/{n_puzzles} perfect, "
            f"error_rate={error_rate:.4f}, elapsed={elapsed:.1f}s")


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
    },
    outputs={
        "metrics": {"format": "json"},
    },
    description="Validate Conway's Game of Life forward simulation correctness on training data",
    tags=["validation", "simulation", "conway", "cellular-automaton"],
    version="1.0.0",
)
def validate_conway_solution(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    n_samples: int = 1000,
    random_state: int = 42,
) -> str:
    """
    Validate forward simulation against training data.

    Loads training data, runs Game of Life forward from known start states,
    and compares with known stop states. This verifies the simulation is correct.

    G1 Compliance: Generic validation for any cellular automaton simulation.
    G4 Compliance: Sample size parameterized.

    Parameters:
        n_samples: Number of training samples to validate (use -1 for all)
        random_state: Random seed for sampling
    """
    train_df = _load_data(inputs["train_data"])

    if 0 < n_samples < len(train_df):
        train_df = train_df.sample(n=n_samples, random_state=random_state)

    deltas_arr = train_df["delta"].values
    start_cols = [f"start_{i}" for i in range(N_CELLS)]
    stop_cols = [f"stop_{i}" for i in range(N_CELLS)]

    starts = train_df[start_cols].values.reshape(-1, N, N).astype(np.int8)
    stops = train_df[stop_cols].values.reshape(-1, N, N).astype(np.int8)

    total_errors = 0
    perfect = 0

    # Validate per delta group for efficiency (batch forward by same delta)
    for delta in sorted(set(deltas_arr)):
        mask = deltas_arr == delta
        batch_starts = starts[mask]
        batch_stops = stops[mask]
        results = _forward(batch_starts, int(delta))
        batch_errors = np.sum(results != batch_stops, axis=(1, 2))
        total_errors += int(np.sum(batch_errors))
        perfect += int(np.sum(batch_errors == 0))

    n_total = len(starts)
    metrics = {
        "n_samples": n_total,
        "perfect_simulations": perfect,
        "perfect_rate": round(perfect / n_total, 4) if n_total > 0 else 0.0,
        "total_errors": int(total_errors),
        "error_rate": round(total_errors / (n_total * N_CELLS), 6) if n_total > 0 else 0.0,
    }

    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return f"validate_conway: {perfect}/{n_total} perfect, forward sim verified"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
}


# =============================================================================
# PIPELINE SPECIFICATION
# =============================================================================

PIPELINE_SPEC = [
    {
        "service": "solve_conway_reverse",
        "inputs": {
            "test_data": "conways-reverse-game-of-life-2020/datasets/test.csv",
            "sample_submission": "conways-reverse-game-of-life-2020/datasets/sample_submission.csv",
        },
        "outputs": {
            "submission": "conways-reverse-game-of-life-2020/submission.csv",
            "metrics": "conways-reverse-game-of-life-2020/artifacts/metrics.json",
        },
        "params": {
            "n_generations": 20,
            "population_size": 200,
            "mutation_rate": 0.003,
            "n_best_ratio": 0.1,
            "n_elite": 5,
            "random_alive_rate": 0.2,
            "n_refine_flips": 300,
            "n_workers": 1,
            "random_state": 42,
        },
        "module": "conways_reverse_game_of_life_2020_services",
    }
]

INFERENCE_SPEC = PIPELINE_SPEC  # Same as training for this competition


def run_pipeline(base_path: str, verbose: bool = True):
    """Run the Conway's Reverse Game of Life pipeline."""
    for i, step in enumerate(PIPELINE_SPEC, 1):
        service_name = step["service"]
        service_fn = SERVICE_REGISTRY.get(service_name)
        if not service_fn:
            print(f"Error: Service {service_name} not found")
            continue

        res_in = {k: os.path.join(base_path, v) for k, v in step["inputs"].items()}
        res_out = {k: os.path.join(base_path, v) for k, v in step["outputs"].items()}

        if verbose:
            print(f"[{i}/{len(PIPELINE_SPEC)}] {service_name}...")

        try:
            result = service_fn(inputs=res_in, outputs=res_out, **step.get("params", {}))
            if verbose:
                print(f"  OK - {result}")
        except Exception as e:
            if verbose:
                print(f"  FAILED - {e}")
            raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Conway's Reverse Game of Life Solver")
    parser.add_argument("--base-path", default="storage", help="Base path for data")
    parser.add_argument("--validate", action="store_true", help="Run validation on training data")
    args = parser.parse_args()

    if args.validate:
        validate_conway_solution(
            inputs={"train_data": os.path.join(args.base_path, "conways-reverse-game-of-life-2020/datasets/train.csv")},
            outputs={"metrics": os.path.join(args.base_path, "conways-reverse-game-of-life-2020/artifacts/validation_metrics.json")},
        )
    else:
        run_pipeline(args.base_path)