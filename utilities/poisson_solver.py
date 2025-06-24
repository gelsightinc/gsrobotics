import numpy as np
from scipy import fftpack
import math


def poisson_dct_neumann(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """
    Perform 2D integration using a Poisson solver with non-homogeneous
    Neuman boundary conditions.

    Args:
        gx (np.ndarray): x-direction gradient array
        gy (np.ndarray): y-direction gradient array

    Returns:
        np.ndarray: reconstructed depth map.
    """

    # Ensure that gx and gy have the same shape.
    assert gx.shape == gy.shape, "gx and gy must have the same shape"

    # Get number of rows and columns.
    num_rows, num_cols = gx.shape

    # Create index arrays for x-direction differentiation
    next_col_ids = np.r_[1:num_cols, num_cols - 1]
    prev_col_ids = np.r_[0 : num_cols - 1, num_cols - 2]

    # Compute differences for x direction
    gxx = gx[:, next_col_ids] - gx[:, prev_col_ids]

    # Create index arrays for y-direction differentiation
    next_row_ids = np.r_[1:num_rows, num_rows - 1]
    prev_row_ids = np.r_[0 : num_rows - 1, num_rows - 2]

    # Compute differences for y direction
    gyy = gy[next_row_ids, :] - gy[prev_row_ids, :]

    # Calculate divergence (Right hand side of Poisson equation)
    div = gxx + gyy

    ### Right hand side of the boundary condition
    b = np.zeros(gx.shape)
    # Top boundary without corners
    b[0, 1:-1] = -gy[0, 1:-1]
    # Bottom boundary without corners
    b[-1, 1:-1] = gy[-1, 1:-1]
    # Left boundary without corners
    b[1:-1, 0] = -gx[1:-1, 0]
    # Right boundary without corners
    b[1:-1, -1] = gx[1:-1, -1]

    # constant factors
    factor = np.sqrt(2)
    factor_inv = 1 / factor

    # Corner adjustmets with normalization factor
    b[0, 0] = factor_inv * (-gy[0, 0] - gx[0, 0])
    b[0, -1] = factor_inv * (-gy[0, -1] + gx[0, -1])
    b[-1, 0] = factor_inv * (gy[-1, 0] - gx[-1, 0])
    b[-1, -1] = factor_inv * (gy[-1, -1] + gx[-1, -1])

    # Adjust divergence 'div' at boundaries to take into account correction 'b'
    div[0, 1:-1] -= b[0, 1:-1]
    div[-1, 1:-1] -= b[-1, 1:-1]
    div[1:-1, 0] -= b[1:-1, 0]
    div[1:-1, -1] -= b[1:-1, -1]

    # Apply scaling factor to boudnary corners
    div[0, 0] -= factor * b[0, 0]
    div[0, -1] -= factor * b[0, -1]
    div[-1, 0] -= factor * b[-1, 0]
    div[-1, -1] -= factor * b[-1, -1]

    # Convert adjusted divergence 'f' to frequency space using DCT transfom.
    div_dct = fftpack.dct(fftpack.dct(div, norm="ortho").T, norm="ortho").T

    # Create the denominator corresponding to Laplacian operator eigenvalues
    x, y = np.meshgrid(np.arange(1, div.shape[1] + 1), np.arange(1, div.shape[0] + 1))

    denom = 4 * (
        (np.sin(0.5 * math.pi * x / (div.shape[1]))) ** 2
        + (np.sin(0.5 * math.pi * y / (div.shape[0]))) ** 2
    )

    # Solve in the frequency space
    depth_dct = -div_dct / denom

    # Apply Inverse DCT to trasnform back to spacial space.
    depth = fftpack.idct(fftpack.idct(depth_dct, norm="ortho").T, norm="ortho").T

    # Subtract mean to get centered depth around zero
    depth -= depth.mean()

    return depth
