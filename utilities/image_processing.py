import cv2
import numpy as np
from scipy.interpolate import griddata
from typing import Optional


def normalize_array(array: np.ndarray, min_divider: float = None) -> np.ndarray:
    """
    Normalizes array of data. Desnt have to be image data.

    Args:
        array (np.ndarray): array to normalize

    Returns:
        np.ndarray: normalized result.
    """
    diff = array.max() - array.min()
    if min_divider:
        diff = max(min_divider, diff)

    normalized = (array - array.min()) / diff
    return normalized


def create_kernel(kernel_size: int = 5) -> np.ndarray:
    """
    Create a square-shaped kernel of ones for image processing operations.

    Args:
        kernel_size (int, optional): Size of the kernel. Defaults to 5.

    Returns:
        np.ndarray: Kernel array.
    """

    return np.ones(shape=(kernel_size, kernel_size), dtype=np.uint8)


def mask_from_range(image: np.ndarray, lower: int = 0, upper: int = 70) -> np.ndarray:
    """
    Create a binary mask from grayscale image based on the given intensity range.

    Args:
        image (np.ndarray): Image provided in grayscale.
        lower (int, optional): Lower threshold value. Defaults to 0.
        upper (int, optional): Upper threshold value. Defaults to 70.

    Returns:
        np.ndarray: Binary mask from grayscale image.
    """
    return cv2.inRange(src=image, lowerb=lower, upperb=upper)


def dilate(image: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    """
    Dilate input image using a square kernel.

    Args:
        image (np.ndarray): Input image
        kernel_size (int, optional): Size of square dilation kernel. Defaults to 5.
        iterations (int, optional): Number of iterations in dilate algorithm. Defaults to 1.

    Returns:
        np.ndarray: Dilated image.
    """
    kernel = create_kernel(kernel_size=kernel_size)
    return cv2.dilate(src=image, kernel=kernel, iterations=iterations)


def erode(image: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    """
    Erode input image using a square kernel.

    Args:
        image (np.ndarray): Input image
        kernel_size (int, optional): Size of square erosion kernel. Defaults to 5.
        iterations (int, optional): Number of iterations in erode algorithm. Defaults to 1.

    Returns:
        np.ndarray: Eroded image.
    """
    kernel = create_kernel(kernel_size=kernel_size)
    return cv2.erode(src=image, kernel=kernel, iterations=iterations)


def rescale(image: np.ndarray, scale: float = 1) -> np.ndarray:
    """
    Rescale input image while keeping its ratio.

    Args:
        image (np.ndarray): Input image
        scale (float, optional): Scaling size. Defaults to 1.

    Returns:
        np.ndarray: Scaled image.
    """
    if scale != 1.0:
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        image = cv2.resize(image, (new_width, new_height))

    return image


def crop_and_resize(
    image: np.ndarray,
    target_size: Optional[tuple[int, int]] = None,
    border_fraction: float = 0.15,
) -> np.ndarray:
    """
    Crop a fraction of the image along the borders while keeping its ratio, and optionally resize
    the cropped image if a target size is provided.

    Args:
        image (np.ndarray): Image to modify.
        target_size (Optional[tuple[int, int]]): Tuple (target_width, target_height) to which the
            image will be resized. If None, only cropping is occurs.
        border_fraction (float, optional): Fraction of the image dimensions to crop from each border.
            Is clamped to range [0, 0.49]
            Defaults to 0.15.

    Returns:
        np.ndarray: The modified image.
    """
    # clamp border fraction
    border_fraction = min(max(0, border_fraction), 0.49)
    # Calculate border sizes
    border_x = int(image.shape[0] * border_fraction)
    border_y = int(image.shape[1] * border_fraction)

    # Crop image
    modified_image = image[
        border_x : image.shape[0] - border_x, border_y : image.shape[1] - border_y
    ]

    # If a target size is provided, resize the cropped image
    if target_size is not None:
        modified_image = cv2.resize(modified_image, target_size)

    return modified_image


def matching_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Find row-wise intersection of two arrays A and B (equal rows).
    Based on stackoverflow solution:

    https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays

    Args:
        A (np.ndarray): first 2D array
        B (np.ndarray): second 2D array

    Returns:
        np.ndarray: row-wise intersection of A and B.
    """

    matches = [i for i in range(B.shape[0]) if np.any(np.all(A == B[i], axis=1))]
    if len(matches) == 0:
        return B[matches]
    return np.unique(B[matches], axis=0)


def interpolate_grad(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Interpolate image region defined by a binary mask using nearest neighbour.

    Function Interpolates values for image at pixels determined by binary mask.
    Mask is first dilated do define a neighbourhood of the masked area which
    serves as a source of 'known' values for the 'nearest' griddata interpolation algorithm.
    Interpolated values are cleaned from potential NaNs and then applied to the original
    image.


    Args:
        image (np.ndarray): 2D source image array
        mask (np.ndarray): Binary mask array (defines interpolation area)

    Returns:
        np.ndarray: Image with interpolation applied to the masked area.
    """

    # Create a new mask around the edge of the original mask
    dilated_mask = dilate(image=mask, kernel_size=3, iterations=2)
    mask_around = np.logical_and(dilated_mask > 0, mask == 0)

    # Indentify coordinates region for interpolation
    mask_around_trim = mask_around == 1

    # Generate coordinate grids
    row_range = np.arange(image.shape[0])
    col_range = np.arange(image.shape[1])
    col_grid, row_grid = np.meshgrid(col_range, row_range)

    # Sample positions and values for interpolation
    sample_rows = row_grid[mask_around_trim]
    sample_cols = col_grid[mask_around_trim]
    sample_points = np.vstack([sample_rows, sample_cols]).T
    sample_values = image[mask_around_trim]

    # Target points where interpolation will be applied
    target_rows = row_grid[mask != 0]
    target_cols = col_grid[mask != 0]
    target_points = np.vstack([target_rows, target_cols]).T

    # Interpolate data using scipy 'nearest' method
    interpolated_values = griddata(
        points=sample_points, values=sample_values, xi=target_points, method="nearest"
    )

    # Set potential Nan values to zero.
    interpolated_values[np.isnan(interpolated_values)] = 0.0

    # make copy of original image and apply interpolated values at masked region
    interpolated_image = image.copy()
    interpolated_image[mask != 0] = interpolated_values

    return interpolated_image


def remove_masked_area(
    gx: np.ndarray, gy: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate x and y gradient components at masked location.

    Args:
        gx (np.ndarray): The x-direction gradient array
        gy (np.ndarray): The y-direction gradient array
        mask (np.ndarray): A binary mask determining the region of interpolation

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Interpolated x-gradient (gx)
            - Interpolated y-gradient (gy)
    """

    gx_interpolate = interpolate_grad(image=gx.copy(), mask=mask)
    gy_interpolate = interpolate_grad(image=gy.copy(), mask=mask)

    return gx_interpolate, gy_interpolate


def add_fps_count_overlay(frame: np.ndarray, fps: float) -> None:
    """
    Adds overlay to frame using 'cv2.putText'. Proper values
    of text position, size and thickness are calculated based
    on the frame size and its comparison to reference values.

    Args:
        frame (MatLike): image frame for which we want to add overlay
    """

    height, width, _ = frame.shape
    # Reference resolution for which proper values of text were calculated
    baseline_height = 2464
    baseline_width = 3280

    # calculated scale factors
    scale_factor_height = height / baseline_height
    scale_factor_width = width / baseline_width

    # add text ovelay with parameter adjusted values
    cv2.putText(
        img=frame,
        text=f"FPS: {int(fps)}",
        org=(
            int(40 * scale_factor_width),
            int(140 * scale_factor_height),
        ),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=4 * scale_factor_height,
        color=(255, 255, 255),
        thickness=max(1, int(10 * scale_factor_height)),
        lineType=cv2.LINE_AA,
    )


def stack_label_above_image(
    image: np.ndarray, text: str, label_height: int
) -> np.ndarray:
    """
    This function adds text information to the image without compromising
    image area itself. It extends image at the top and adds text label
    with defined text and label high.

    Args:
        image (np.ndarray): Image to modify.
        text (str): Text to add above the image.
        label_height (int): Height in pixels for the Label area above the image.

    Returns:
        np.ndarray: Extended image with the label area.
    """
    image_width = image.shape[1]

    # Label with black background
    label = np.zeros((label_height, image_width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)
    thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (image_width - text_size[0]) // 2
    text_y = (label_height + text_size[1]) // 2

    cv2.putText(
        img=label,
        text=text,
        org=(text_x, text_y),
        fontFace=font,
        fontScale=font_scale,
        color=font_color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    # Return image with label stacked above
    return np.vstack((label, image))


def color_map_from_txt(path: str, is_bgr: bool = False) -> np.ndarray:
    """
    Loads color map from text file. Each line in text consists of three
    comma separated integer values describing RGB

    Args:
        path (str): path to txt file
        is_bgr (bool, optional): Defines if the order was in BGR format. Defaults to False.

    Returns:
        np.ndarray: returns loaded color map.
    """
    cmap = np.loadtxt(fname=path, dtype=int)
    if is_bgr:
        cmap = cmap[:, ::-1]
    return cmap


def apply_cmap(data: np.ndarray, cmap: np.ndarray) -> np.ndarray:
    """
    Applies color map in the form of np.ndarray where each column corresponds
    to one of RGB channels.

    Args:
        data (np.ndarray): data array for which color map needs to be applied
        cmap (np.ndarray): color map data

    Returns:
        np.ndarray: data with applied color map.
    """
    data_index = ((cmap.shape[0] - 1) * data).astype(np.int32)
    data_index = np.clip(data_index, 0, cmap.shape[0] - 1)
    data_cmapped = cmap[data_index]
    data_cmapped = data_cmapped.astype(np.uint)
    return data_cmapped


def trim_outliers(
    data: np.ndarray, lower_percentile: float = 1, upper_percentile: float = 99
) -> np.ndarray:
    """
    Clip the data so that values below the lower percentile and above the upper
    percentile are set to the threshold values.

    Parameters:
        data (np.ndarray): The input data array.
        lower_percentile (float): The lower percentile to use (default 1).
        upper_percentile (float): The upper percentile to use (default 99).

    Returns:
        np.ndarray: The clipped data.
    """
    lower = np.percentile(data, lower_percentile)
    upper = np.percentile(data, upper_percentile)
    return np.clip(data, lower, upper)
