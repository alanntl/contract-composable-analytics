"""
Santa 2022 - The Christmas Card Conundrum - Contract-Composable Analytics Services
============================================================
Competition: https://www.kaggle.com/competitions/santa-2022
Problem Type: Optimization (TSP-variant / Robotic Arm Path Planning)
Goal: Minimize total cost (reconfiguration cost + color cost)

This is an optimization competition where a robotic arm with 8 links
must visit all 257x257 pixels minimizing movement and color change costs.

Competition-specific services:
- load_santa_image: Load and convert image data to numpy array
- generate_snake_path: Generate path using snake/zigzag pattern
- compress_arm_path: Compress path by moving multiple links simultaneously
- apply_2opt_optimization: Apply 2-opt local optimization
- create_santa_submission: Create submission file from path
"""

import numpy as np
import pandas as pd
from functools import reduce
from itertools import product
from typing import Dict, List, Optional, Tuple
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contract import contract

try:
    from services.io_services import load_data, save_data
except ImportError:
    def load_data(path: str) -> pd.DataFrame:
        if path.endswith('.csv'):
            return pd.read_csv(path)
        elif path.endswith('.parquet'):
            return pd.read_parquet(path)
        else:
            return pd.read_csv(path)

    def save_data(df: pd.DataFrame, path: str, index: bool = False) -> None:
        if path.endswith('.csv'):
            df.to_csv(path, index=index)
        elif path.endswith('.parquet'):
            df.to_parquet(path, index=index)


# =============================================================================
# HELPER FUNCTIONS (Internal - not exposed as services)
# =============================================================================

def _cartesian_to_array(x: int, y: int, shape: Tuple[int, int, int] = (257, 257, 3)) -> Tuple[int, int]:
    """Transform cartesian coordinates to array indices."""
    m, n = shape[:2]
    i = (n - 1) // 2 - y
    j = (n - 1) // 2 + x
    return i, j


def _array_to_cartesian(i: int, j: int, shape: Tuple[int, int, int] = (257, 257, 3)) -> Tuple[int, int]:
    """Transform array indices to cartesian coordinates."""
    m, n = shape[:2]
    y = (n - 1) // 2 - i
    x = j - (n - 1) // 2
    return x, y


def _df_to_image(df: pd.DataFrame) -> np.ndarray:
    """Convert dataframe to 3D image array."""
    side = int(len(df) ** 0.5)
    return df.set_index(['x', 'y']).to_numpy().reshape(side, side, -1)


def _get_position(config: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Get the position (endpoint) of an arm configuration."""
    return reduce(lambda p, q: (p[0] + q[0], p[1] + q[1]), config, (0, 0))


def _rotate_link(vector: Tuple[int, int], direction: int) -> Tuple[int, int]:
    """Rotate a single link one step in the given direction."""
    x, y = vector
    if direction == 1:  # counter-clockwise
        if y >= x and y > -x:
            x -= 1
        elif y > x and y <= -x:
            y -= 1
        elif y <= x and y < -x:
            x += 1
        else:
            y += 1
    elif direction == -1:  # clockwise
        if y > x and y >= -x:
            x += 1
        elif y >= x and y < -x:
            y += 1
        elif y < x and y <= -x:
            x -= 1
        else:
            y -= 1
    return (x, y)


def _rotate(config: List[Tuple[int, int]], i: int, direction: int) -> List[Tuple[int, int]]:
    """Rotate link i of the configuration by one step."""
    config = config.copy()
    config[i] = _rotate_link(config[i], direction)
    return config


def _get_direction(u: Tuple[int, int], v: Tuple[int, int]) -> int:
    """Get rotation direction from vector u to vector v."""
    direction = np.sign(np.cross(u, v))
    if direction == 0 and np.dot(u, v) < 0:
        direction = 1
    return int(direction)


def _reconfiguration_cost(from_config: List, to_config: List) -> float:
    """Cost of reconfiguring the arm (sqrt of number of links rotated)."""
    diffs = np.abs(np.asarray(from_config) - np.asarray(to_config)).sum(axis=1)
    if not (diffs <= 1).all():
        return 1e6  # Invalid movement
    return np.sqrt(diffs.sum())


def _color_cost(from_pos: Tuple[int, int], to_pos: Tuple[int, int],
                image: np.ndarray, color_scale: float = 3.0) -> float:
    """Cost of color change between two positions."""
    return np.abs(image[to_pos] - image[from_pos]).sum() * color_scale


def _step_cost(from_config: List, to_config: List, image: np.ndarray) -> float:
    """Total cost of one step (reconfiguration + color)."""
    from_pos = _cartesian_to_array(*_get_position(from_config), image.shape)
    to_pos = _cartesian_to_array(*_get_position(to_config), image.shape)
    return _reconfiguration_cost(from_config, to_config) + _color_cost(from_pos, to_pos, image)


def _total_cost(path: List, image: np.ndarray) -> float:
    """Compute total cost of a path."""
    return reduce(
        lambda cost, pair: cost + _step_cost(pair[0], pair[1], image),
        zip(path[:-1], path[1:]),
        0,
    )


def _get_path_to_point(config: List[Tuple[int, int]], point: Tuple[int, int]) -> List:
    """Find a path from current config to target point."""
    path = [config]
    for i in range(len(config)):
        link = config[i]
        base = _get_position(config[:i])
        relbase = (point[0] - base[0], point[1] - base[1])
        position = _get_position(config[:i+1])
        relpos = (point[0] - position[0], point[1] - position[1])
        radius = reduce(lambda r, link: r + max(abs(link[0]), abs(link[1])), config[i+1:], 0)

        if radius == 1 and relpos == (0, 0):
            config = _rotate(config, i, 1)
            if _get_position(config) == point:
                path.append(config)
                break
            else:
                continue

        while np.max(np.abs(relpos)) > radius:
            direction = _get_direction(link, relbase)
            config = _rotate(config, i, direction)
            path.append(config)
            link = config[i]
            base = _get_position(config[:i])
            relbase = (point[0] - base[0], point[1] - base[1])
            position = _get_position(config[:i+1])
            relpos = (point[0] - position[0], point[1] - position[1])
            radius = reduce(lambda r, link: r + max(abs(link[0]), abs(link[1])), config[i+1:], 0)

    return path


def _get_path_to_configuration(from_config: List, to_config: List) -> List:
    """Find a path from one configuration to another."""
    path = [from_config]
    config = from_config.copy()
    while config != to_config:
        for i in range(len(config)):
            config = _rotate(config, i, _get_direction(config[i], to_config[i]))
        path.append(config)
    return path


def _compress_path(path: List) -> List:
    """Compress path by moving multiple links simultaneously."""
    r = [[] for _ in range(8)]
    for p in path:
        for i in range(8):
            if len(r[i]) == 0 or r[i][-1] != p[i]:
                r[i].append(p[i])
    mx = max([len(x) for x in r])

    for rr in r:
        while len(rr) < mx:
            rr.append(rr[-1])

    result = list(zip(*r))
    return [list(item) for item in result]


def _get_path_to_point_compressed(config: List[Tuple[int, int]], point: Tuple[int, int]) -> List:
    """Find compressed path from current config to target point."""
    path = _get_path_to_point(config, point)
    return _compress_path(path)


# =============================================================================
# Contract-Composable Analytics SERVICES
# =============================================================================

@contract(
    inputs={"image_data": {"format": "csv", "required": True}},
    outputs={"image_array": {"format": "pickle", "required": True}},
    description="Load Santa 2022 image data and convert to numpy array",
    tags=["santa-2022", "optimization", "io"],
    version="1.0.0"
)
def load_santa_image(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
) -> str:
    """
    Load Santa 2022 image data and convert to numpy array.

    Inputs:
        image_data: Path to image.csv file
    Outputs:
        image_array: Path to save numpy array (pickle)

    Returns:
        Status message
    """
    import pickle

    df = load_data(inputs['image_data'])
    image = _df_to_image(df)

    os.makedirs(os.path.dirname(outputs['image_array']), exist_ok=True)
    with open(outputs['image_array'], 'wb') as f:
        pickle.dump(image, f)

    return f"Loaded image with shape {image.shape}"


@contract(
    inputs={"image_array": {"format": "pickle", "required": True}},
    outputs={"path_array": {"format": "pickle", "required": True}},
    description="Generate path using snake/zigzag pattern for better color continuity",
    tags=["santa-2022", "optimization", "path-planning"],
    version="1.0.0"
)
def generate_snake_path(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    use_compression: bool = True,
) -> str:
    """
    Generate path using snake/zigzag pattern for better color continuity.

    This reorders points to follow a snake pattern (alternating direction
    on each row) which reduces color change cost significantly.

    Inputs:
        image_array: Path to numpy image array (pickle)
    Outputs:
        path_array: Path to save path array (pickle)

    Returns:
        Status message with path statistics
    """
    import pickle
    from tqdm import tqdm

    with open(inputs['image_array'], 'rb') as f:
        image = pickle.load(f)

    # Origin configuration
    origin = [(64, 0), (-32, 0), (-16, 0), (-8, 0), (-4, 0), (-2, 0), (-1, 0), (-1, 0)]
    n = origin[0][0] * 2  # 128

    # Generate snake pattern points (y varies, x alternates direction)
    points = []
    for x in range(-n, n + 1):
        if x % 2 == 0:
            for y in range(n, -n - 1, -1):
                points.append((y, x))
        else:
            for y in range(-n, n + 1):
                points.append((y, x))

    # Build path
    path = [origin]
    path_func = _get_path_to_point_compressed if use_compression else _get_path_to_point

    for i, p in enumerate(tqdm(points, desc="Generating path")):
        config = path[-1]
        new_path = path_func(config, p)
        path.extend(new_path[1:])

    # Return to origin configuration
    path.extend(_get_path_to_configuration(path[-1], origin)[1:])

    # Convert to numpy array
    path_array = np.array(path)

    os.makedirs(os.path.dirname(outputs['path_array']), exist_ok=True)
    with open(outputs['path_array'], 'wb') as f:
        pickle.dump(path_array, f)

    # Calculate initial cost
    cost = _total_cost(path, image)

    return f"Generated path with {len(path)} steps, initial cost: {cost:.2f}"


@contract(
    inputs={"image_array": {"format": "pickle", "required": True}},
    outputs={"path_array": {"format": "pickle", "required": True}},
    description="Generate path using greedy nearest-neighbor with color consideration",
    tags=["santa-2022", "optimization", "path-planning", "greedy"],
    version="1.0.0"
)
def generate_greedy_color_path(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    use_compression: bool = True,
    block_size: int = 8,
) -> str:
    """
    Generate path using greedy nearest-neighbor with color consideration.

    Divides the image into blocks and uses a greedy approach within each block,
    ordering pixels by color similarity to reduce color change cost.

    Inputs:
        image_array: Path to numpy image array (pickle)
    Outputs:
        path_array: Path to save path array (pickle)

    Parameters:
        use_compression: Whether to compress arm movements (default: True)
        block_size: Size of blocks for local ordering (default: 8)

    Returns:
        Status message with path statistics
    """
    import pickle
    from tqdm import tqdm

    with open(inputs['image_array'], 'rb') as f:
        image = pickle.load(f)

    origin = [(64, 0), (-32, 0), (-16, 0), (-8, 0), (-4, 0), (-2, 0), (-1, 0), (-1, 0)]
    n = 128

    # Create list of all pixels with their colors
    pixels = []
    for x in range(-n, n + 1):
        for y in range(-n, n + 1):
            i, j = _cartesian_to_array(x, y, image.shape)
            color = image[i, j]
            pixels.append((x, y, color[0], color[1], color[2]))

    # Sort pixels into blocks and order by color within each block
    def get_block_key(px):
        bx = (px[0] + n) // block_size
        by = (px[1] + n) // block_size
        # Alternate direction in each row of blocks (snake pattern)
        if bx % 2 == 0:
            return (bx, by, px[2] + px[3] + px[4])  # Sort by color sum
        else:
            return (bx, -by, px[2] + px[3] + px[4])

    pixels.sort(key=get_block_key)
    points = [(p[0], p[1]) for p in pixels]

    # Build path
    path = [origin]
    path_func = _get_path_to_point_compressed if use_compression else _get_path_to_point

    for p in tqdm(points, desc="Generating color-aware path"):
        config = path[-1]
        new_path = path_func(config, p)
        path.extend(new_path[1:])

    path.extend(_get_path_to_configuration(path[-1], origin)[1:])
    path_array = np.array(path)

    os.makedirs(os.path.dirname(outputs['path_array']), exist_ok=True)
    with open(outputs['path_array'], 'wb') as f:
        pickle.dump(path_array, f)

    cost = _total_cost(path, image)
    return f"Generated color-aware path with {len(path)} steps, cost: {cost:.2f}"


@contract(
    inputs={"image_array": {"format": "pickle", "required": True}},
    outputs={"path_array": {"format": "pickle", "required": True}},
    description="Generate path using Hilbert curve ordering with color refinement",
    tags=["santa-2022", "optimization", "path-planning", "hilbert"],
    version="1.0.0"
)
def generate_hilbert_color_path(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    use_compression: bool = True,
) -> str:
    """
    Generate path using Hilbert curve ordering with color refinement.

    Uses space-filling Hilbert curve combined with color similarity
    for locally optimal pixel ordering.

    Inputs:
        image_array: Path to numpy image array (pickle)
    Outputs:
        path_array: Path to save path array (pickle)

    Returns:
        Status message with path statistics
    """
    import pickle
    from tqdm import tqdm

    def hilbert_d2xy(n, d):
        """Convert Hilbert curve index d to (x, y) coordinates."""
        x = y = 0
        s = 1
        while s < n:
            rx = 1 & (d // 2)
            ry = 1 & (d ^ rx)
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            d //= 4
            s *= 2
        return x, y

    with open(inputs['image_array'], 'rb') as f:
        image = pickle.load(f)

    origin = [(64, 0), (-32, 0), (-16, 0), (-8, 0), (-4, 0), (-2, 0), (-1, 0), (-1, 0)]
    n = 128
    size = 2 * n + 1  # 257

    # Use 512x512 Hilbert curve and map to 257x257
    hilbert_n = 512
    points_hilbert = []

    for d in range(hilbert_n * hilbert_n):
        hx, hy = hilbert_d2xy(hilbert_n, d)
        # Map from [0, 511] to [-128, 128]
        x = int(hx * size / hilbert_n) - n
        y = int(hy * size / hilbert_n) - n
        x = max(-n, min(n, x))
        y = max(-n, min(n, y))
        if (x, y) not in [(p[0], p[1]) for p in points_hilbert[-1:]] if points_hilbert else True:
            points_hilbert.append((x, y))

    # Deduplicate while preserving order
    seen = set()
    points = []
    for p in points_hilbert:
        if p not in seen and -n <= p[0] <= n and -n <= p[1] <= n:
            seen.add(p)
            points.append(p)

    # Add any missing points
    all_points = set((x, y) for x in range(-n, n + 1) for y in range(-n, n + 1))
    missing = all_points - seen
    # Add missing points sorted by position
    missing_sorted = sorted(missing, key=lambda p: (p[0] + p[1], p[0]))
    points.extend(missing_sorted)

    # Build path
    path = [origin]
    path_func = _get_path_to_point_compressed if use_compression else _get_path_to_point

    for p in tqdm(points, desc="Generating Hilbert path"):
        config = path[-1]
        new_path = path_func(config, p)
        path.extend(new_path[1:])

    path.extend(_get_path_to_configuration(path[-1], origin)[1:])
    path_array = np.array(path)

    os.makedirs(os.path.dirname(outputs['path_array']), exist_ok=True)
    with open(outputs['path_array'], 'wb') as f:
        pickle.dump(path_array, f)

    cost = _total_cost(path, image)
    return f"Generated Hilbert path with {len(path)} steps, cost: {cost:.2f}"


@contract(
    inputs={
        "path_array": {"format": "pickle", "required": True},
        "image_array": {"format": "pickle", "required": True}
    },
    outputs={"optimized_path": {"format": "pickle", "required": True}},
    description="Apply 2-opt local optimization to improve the path",
    tags=["santa-2022", "optimization", "2opt", "local-search"],
    version="1.0.0"
)
def apply_2opt_optimization(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    max_k: int = 50,
    n_passes: int = 2,
) -> str:
    """
    Apply 2-opt local optimization to improve the path.

    2-opt swaps edges to reduce total cost. For each position i,
    we test if swapping edges (v_i, v_{i+1}) and (v_{i+k}, v_{i+k+1})
    with (v_i, v_{i+k}) and (v_{i+1}, v_{i+k+1}) reduces cost.

    Inputs:
        path_array: Path to input path array (pickle)
        image_array: Path to numpy image array (pickle)
    Outputs:
        optimized_path: Path to save optimized path array (pickle)

    Parameters:
        max_k: Maximum distance for 2-opt swaps (default: 50)
        n_passes: Number of optimization passes (default: 2)

    Returns:
        Status message with optimization results
    """
    import pickle

    with open(inputs['path_array'], 'rb') as f:
        path = pickle.load(f)
    with open(inputs['image_array'], 'rb') as f:
        image = pickle.load(f)

    optimized_path = path.copy()
    initial_cost = _total_cost(list(optimized_path), image)

    total_improvements = 0

    for pass_num in range(n_passes):
        for k in range(2, min(max_k + 1, len(optimized_path) - 1)):
            count = 0
            for i in range(len(optimized_path) - k - 1):
                # Current cost of two edges
                current_cost = _step_cost(list(optimized_path[i]), list(optimized_path[i+1]), image)
                current_cost += _step_cost(list(optimized_path[i+k]), list(optimized_path[i+k+1]), image)

                # New cost if we swap
                new_cost = _step_cost(list(optimized_path[i]), list(optimized_path[i+k]), image)
                new_cost += _step_cost(list(optimized_path[i+1]), list(optimized_path[i+k+1]), image)

                if new_cost < current_cost:
                    # Perform the 2-opt swap (reverse the segment between i+1 and i+k)
                    optimized_path[i+1:i+k+1] = np.flip(optimized_path[i+1:i+k+1], axis=0)
                    count += 1
                    total_improvements += 1

            if count > 0:
                print(f"Pass {pass_num+1}, k={k}: {count} improvements")

    # Remove consecutive duplicates
    mask = np.ones(len(optimized_path), dtype=bool)
    for i in range(1, len(optimized_path)):
        if np.array_equal(optimized_path[i], optimized_path[i-1]):
            mask[i] = False
    optimized_path = optimized_path[mask]

    final_cost = _total_cost(list(optimized_path), image)

    os.makedirs(os.path.dirname(outputs['optimized_path']), exist_ok=True)
    with open(outputs['optimized_path'], 'wb') as f:
        pickle.dump(optimized_path, f)

    improvement = initial_cost - final_cost
    return f"Optimized path: {len(optimized_path)} steps, cost: {final_cost:.2f} (improved by {improvement:.2f}, {total_improvements} swaps)"


@contract(
    inputs={"optimized_path": {"format": "pickle", "required": True}},
    outputs={"submission": {"format": "csv", "required": True}},
    description="Create submission CSV from path array",
    tags=["santa-2022", "optimization", "submission", "io"],
    version="1.0.0"
)
def create_santa_submission(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
) -> str:
    """
    Create submission CSV from path array.

    Converts the numpy path array to the expected submission format:
    configuration column with semicolon-separated arm vectors.

    Inputs:
        optimized_path: Path to optimized path array (pickle)
    Outputs:
        submission: Path to save submission CSV

    Returns:
        Status message
    """
    import pickle

    with open(inputs['optimized_path'], 'rb') as f:
        path = pickle.load(f)

    def config_to_string(config):
        return ';'.join([' '.join(map(str, vector)) for vector in config])

    submission = pd.Series(
        [config_to_string(config) for config in path],
        name="configuration",
    )

    os.makedirs(os.path.dirname(outputs['submission']), exist_ok=True)
    submission.to_csv(outputs['submission'], index=False, header=True)

    return f"Created submission with {len(submission)} configurations"


@contract(
    inputs={
        "path_array": {"format": "pickle", "required": True},
        "image_array": {"format": "pickle", "required": True}
    },
    outputs={"metrics": {"format": "json", "required": True}},
    description="Compute and save the total cost of a path",
    tags=["santa-2022", "optimization", "metrics"],
    version="1.0.0"
)
def compute_path_cost(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
) -> str:
    """
    Compute and save the total cost of a path.

    Inputs:
        path_array: Path to path array (pickle)
        image_array: Path to numpy image array (pickle)
    Outputs:
        metrics: Path to save metrics JSON

    Returns:
        Status message with cost
    """
    import pickle
    import json

    with open(inputs['path_array'], 'rb') as f:
        path = pickle.load(f)
    with open(inputs['image_array'], 'rb') as f:
        image = pickle.load(f)

    cost = _total_cost(list(path), image)

    metrics = {
        'total_cost': cost,
        'path_length': len(path),
        'image_shape': list(image.shape),
    }

    os.makedirs(os.path.dirname(outputs['metrics']), exist_ok=True)
    with open(outputs['metrics'], 'w') as f:
        json.dump(metrics, f, indent=2)

    return f"Path cost: {cost:.2f}"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "load_santa_image": load_santa_image,
    "generate_snake_path": generate_snake_path,
    "generate_greedy_color_path": generate_greedy_color_path,
    "generate_hilbert_color_path": generate_hilbert_color_path,
    "apply_2opt_optimization": apply_2opt_optimization,
    "create_santa_submission": create_santa_submission,
    "compute_path_cost": compute_path_cost,
}


if __name__ == "__main__":
    # Test the services
    print("Santa 2022 services loaded successfully")
    print(f"Available services: {list(SERVICE_REGISTRY.keys())}")
