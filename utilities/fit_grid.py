import numpy as np
import random
from typing import Tuple
from scipy.spatial.distance import pdist, squareform


def grid_spacing(points: np.ndarray) -> float:
    """
    Estimate the grid spacing based on an array of 2D points.

    -> For each point, compute the distances to all other points
    -> Take the 1st and 2nd smallest distances (excluding itself) for every point.
    -> Return the median of these distances as the grid spacing.

    Args:
        points (np.ndarray): Array of shape (N, 2) containing the 2D coordinates.

    Returns:
        float: Estimated grid spacing.
    """
    # Compute pairwise distance matrix
    dist_matrix = squareform(pdist(points))
    # Sort distances for each point; first column is zero (self-distance)
    sorted_dists = np.sort(dist_matrix, axis=1)
    # Extract the 1st and 2nd distances (except itself) for each point
    candidate_dists = sorted_dists[:, 1:3].flatten()
    return float(np.median(candidate_dists))


def find_neighbors(
    points: np.ndarray,
    target_point: np.ndarray,
    spacing: float,
    spacing_tolerance: float = 0.2,
) -> tuple[bool, np.ndarray, np.ndarray]:
    """
    Find two neighbor points for a given 'target_point' within a specified distance range.

    The function computes distances from 'target_point' to all points in 'points' and
    selects those within [(1-spacing_tolerance)*spacing, (1+spacing_tolerance)*spacing].
    It then picks the first one as neighbor_1 and, among the others, finds the
    candidate with the smallest dot product with the unit vector toward neighbor1.
    Finally, it may swap neighbor1 and neighbor_2 depending on the dot product value.

    Args:
        points (np.ndarray): Array of shape (N, 2) of 2D points.
        target_point (np.ndarray): Reference point of shape (2,).
        spacing (float): Estimated grid spacing.
        spacing_tolerance (float, optional): Tolerance of the grid spacing which
            determines bounds of distance range. Defaults to 0.2 (20% around spacing).

    Returns:
        tuple[bool, np.ndarray, np.ndarray]:  Tuple (flip_flag, neighbor_1, neighbor_2)
    """
    FLIP_FLAP_THRESHOLD = 0.1
    flip_flag = False
    neighbor_1 = np.array([])
    neighbor_2 = np.array([])

    lower_bound = spacing * (1 - spacing_tolerance)
    upper_bound = spacing * (1 + spacing_tolerance)

    # Compute distances from 'target_point' to all points.
    distances = np.sqrt(
        (points[:, 1] - target_point[1]) ** 2 + (points[:, 0] - target_point[0]) ** 2
    )
    valid_indices = np.where((distances > lower_bound) & (distances < upper_bound))[0]
    if valid_indices.size < 3:
        return flip_flag, neighbor_1, neighbor_2

    # First valid neighbor
    neighbor_1 = points[valid_indices[0], :]
    # Unit vector from 'target_point' to neighbor1
    u = neighbor_1 - target_point
    u /= np.linalg.norm(u)

    # For the remaining candidate neighbors:
    candidates = points[valid_indices[1:], :] - target_point
    candidate_norms = np.linalg.norm(candidates, axis=1, keepdims=True)
    candidates_unit = candidates / candidate_norms

    # Compute dot products between u and each candidate unit vector.
    dot_products = np.clip(np.dot(u, candidates_unit.T), -1, 1)
    min_idx = np.argmin(dot_products)
    # Note: neighbor2 comes from valid_indices[min_idx+1] (because we skipped index 0)
    neighbor_2 = points[valid_indices[min_idx + 1], :]

    flip_flag = dot_products[min_idx] < FLIP_FLAP_THRESHOLD

    # Unit vector from 'target_point' to neighbor2
    v = candidates_unit[min_idx]

    # Swap neighbors if neighbor2 is more horizontal than neighbor1.
    if np.abs(v[0]) < np.abs(u[0]):
        neighbor_1, neighbor_2 = neighbor_2.copy(), neighbor_1.copy()

    return flip_flag, neighbor_1, neighbor_2


def find_central_point(points: np.ndarray, spacing: float) -> np.ndarray:
    """
    Randomly search for a central point that has valid neighbors.

    The function shuffles the points (non-deterministically) and checks each candidate
    in random order using `find_neighbors`. The first candidate that returns a True flip_flag
    is selected as the central point. If none is found after checking all points, a ValueError is raised.

    :param points: Array of shape (N, 2) containing 2D points.
    :param spacing: Estimated grid spacing.
    :return: A central point as a 1D array of shape (2,).
    :raises ValueError: If no valid central point is found.
    """
    n_points = points.shape[0]
    indices = np.arange(n_points)
    np.random.shuffle(indices)
    for idx in indices:
        candidate = points[idx, :]
        flip_flag, _, _ = find_neighbors(
            points=points, target_point=candidate, spacing=spacing
        )
        if flip_flag:
            return candidate
    raise ValueError("Did not find a central point with valid neighbors.")


def fit_grid(
    points: np.ndarray, spacing: float = None, central_point: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a grid to the given 2D points using a least-squares adjustment.

    The algorithm:
      1. Computes an initial grid spacing (if not provided) and a central point.
      2. Determines the grid orientation using the neighbors of the central point.
      3. Rotates the points into a grid-aligned coordinate system.
      4. Snaps the points to the nearest grid intersections.
      5. Refines the grid parameters using least squares.
      6. Returns the fitted points in world coordinates and a version of the grid
         (rotated by 180° to match a desired orientation).

    :param points: Array of shape (N, 2) of 2D points.
    :param spacing: Pre-computed grid spacing (optional).
    :param central_point: A central point (optional).
    :return: Tuple (world_points, grid_points) where both are arrays of shape (N, 2).
    """
    MAX_ITERATIONS = 100
    num_points = points.shape[0]

    # Compute grid spacing and central point if not provided.
    if spacing is None:
        spacing = grid_spacing(points=points)
    if central_point is None:
        central_point = find_central_point(points=points, spacing=spacing)

    # Determine local grid orientation using neighbors of the central point.
    _, neighbor1, _ = find_neighbors(points, central_point, spacing)
    delta = neighbor1 - central_point
    theta = np.arctan2(delta[1], delta[0])

    # Rotate all points by theta
    R_initial = np.array(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    )
    shifted_points = points - central_point
    grid_aligned = shifted_points.dot(R_initial)

    # Snap grid coordinates to the nearest multiple of spacing.
    grid_aligned[:, 0] = np.round(grid_aligned[:, 0] / spacing) * spacing
    grid_aligned[:, 1] = np.round(grid_aligned[:, 1] / spacing) * spacing

    # Prepare data for least-squares modification.
    x_coords = grid_aligned[:, 0].reshape(-1, 1)
    y_coords = grid_aligned[:, 1].reshape(-1, 1)
    orig_x = points[:, 0].reshape(-1, 1)
    orig_y = points[:, 1].reshape(-1, 1)
    scale = 1.0
    # Parameter vector: [-theta, offset_x, offset_y, scale]
    params = np.array([-theta, central_point[0], central_point[1], scale]).reshape(
        -1, 1
    )

    step_norm = np.inf
    iteration = 0
    # Iteratively modify the grid parameters.
    while step_norm > 1e-7 and iteration < MAX_ITERATIONS:
        theta_current = params[0, 0]
        offset = params[1:3, 0].reshape(2, 1)

        # Compute rotated coordinates for Jacobian evaluation.
        jix = np.cos(theta_current) * x_coords - np.sin(theta_current) * y_coords
        jiy = np.sin(theta_current) * x_coords + np.cos(theta_current) * y_coords

        # Residuals between model and original coordinates.
        res_x = scale * jix + offset[0] - orig_x
        res_y = scale * jiy + offset[1] - orig_y

        # Build Jacobian matrices for x and y.
        Jx = np.zeros((num_points, 4))
        Jy = np.zeros((num_points, 4))
        Jx[:, 0] = scale * jix[:, 0]
        Jx[:, 1] = 1
        Jx[:, 3] = jix[:, 0]
        Jy[:, 0] = scale * jix[:, 0]
        Jy[:, 2] = 1
        Jy[:, 3] = jiy[:, 0]

        J = np.vstack((Jx, Jy))
        residual = np.vstack((res_x, res_y))
        # Solve the least-squares problem.
        delta_params = np.linalg.lstsq(J, residual, rcond=None)[0]
        params += delta_params
        step_norm = np.sqrt(np.sum(delta_params**2))
        iteration += 1

    refined_theta = params[0, 0]
    refined_offset = params[1:3, 0]
    R_refined = np.array(
        [
            [np.cos(refined_theta), -np.sin(refined_theta)],
            [np.sin(refined_theta), np.cos(refined_theta)],
        ]
    )
    A = np.diag([scale, scale]).dot(R_refined)

    # Map grid-aligned points to world coordinates.
    world_points = grid_aligned.dot(A) + refined_offset
    # Compute error (optional, for debugging)
    error = np.sqrt(
        (world_points[:, 0] - points[:, 1]) ** 2
        + (world_points[:, 1] - points[:, 0]) ** 2
    )

    # Rotate grid-aligned points by 180° to match the grid orientation.
    R_180 = np.array([[np.cos(np.pi), np.sin(np.pi)], [-np.sin(np.pi), np.cos(np.pi)]])
    grid_points = grid_aligned.dot(R_180)

    return world_points, grid_points
