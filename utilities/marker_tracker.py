import math
import copy
from typing import Any, Optional

import numpy as np
import cv2
from skimage.morphology import disk, closing
from skimage import measure
from scipy.ndimage import gaussian_filter, convolve
from utilities.normxcorr2 import normxcorr2
from enum import Enum
from utilities.fit_grid import fit_grid, grid_spacing
from utilities.logger import log_message

GRID_STYLE_ALL = "all"
GRID_STYLE_NO_BORDER = "no-border"


class MarkerTracker:
    """
    Class for tracking markers in an image using image processing techniques.

    This class processes a given marker image, extracts candidate marker locations,
    fits them to a grid, and tracks their positions.
    """

    def __init__(
        self,
        input_image: np.ndarray,
        grid_style: str = "no-border",
        do_plot: bool = False,
    ):
        """
        Initialize the MarkerTracker with a marker image and optional parameters.

        Args:
            input_image (np.ndarray): The input image containing markers.
            grid_style (str, optional): Grid style. Defaults to "no-border".
            do_plot (bool, optional): . Should it plot. Defaults to False.
        """

        marker_image = input_image.copy()
        self.grid_style = grid_style
        self.do_plot = do_plot

        filter_sigma: float = 20.0  # Standard deviation for Gaussian filter.
        binary_threshold: float = -0.05

        # Create new image by maximizing over color channel
        gray_image: np.ndarray = np.max(marker_image.astype(float), axis=2)
        img_height, img_width = gray_image.shape

        # Using SciPy's gaussian_filter for smoothing.
        smoothed = gaussian_filter(gray_image, sigma=filter_sigma)
        highpass = gray_image - smoothed

        # Create a binary mask using disk-shaped element
        selem = disk(3)
        binary_mask = highpass < binary_threshold
        closed_mask = closing(binary_mask, footprint=selem)
        label_image = measure.label(closed_mask)

        # Filter out connected components that that dont wall within a range
        min_blob_size: int = 20
        max_blob_size: int = 500
        clean_labels = np.zeros_like(label_image)
        for i in range(1, label_image.max() + 1):
            blob_size = (label_image == i).sum()
            if min_blob_size <= blob_size <= max_blob_size:
                clean_labels[label_image == i] = i

        # Generate a binary marker mask.
        marker_mask = np.where(clean_labels != 0, 1, 0)
        # Get region properties using scikit-image.
        region_props = measure.regionprops(clean_labels, intensity_image=gray_image)
        areas = np.array([prop.area for prop in region_props])
        centers = np.array([prop.centroid for prop in region_props])
        mean_intensities = np.array([prop.intensity_mean for prop in region_props])

        # Sort the centers.
        sorted_idx = self.sort_centers(centers)
        sorted_centers = centers[sorted_idx]
        sorted_areas = areas[sorted_idx]
        sorted_intensities = mean_intensities[sorted_idx]

        # Estimate grid spacing.
        grid_spacing_est = grid_spacing(points=sorted_centers)
        new_centers = sorted_centers.copy()
        new_areas = sorted_areas.copy()
        new_intensities = sorted_intensities.copy()

        # Depending on the grid style, either assign full grid coordinates or filter centers.
        if self.grid_style == GRID_STYLE_ALL:
            num_rows, num_cols, row_coords, col_coords = self.assign_coordinates(
                sorted_centers
            )
        else:
            # Exclude points that are too close to image boundaries and too near each other.
            min_boundary_distance = grid_spacing_est / 8
            min_spacing = grid_spacing_est / 2
            filtered_centers = []
            valid_indices = np.empty((0,), dtype=int)
            for idx in range(sorted_centers.shape[0]):
                point = sorted_centers[idx]
                if (
                    min_boundary_distance < point[1] < img_width - min_boundary_distance
                    and min_boundary_distance
                    < point[0]
                    < img_height - min_boundary_distance
                ):
                    if filtered_centers:
                        distances = np.linalg.norm(
                            point - np.array(filtered_centers), axis=1
                        )
                        if not np.any(distances < min_spacing):
                            filtered_centers.append(point)
                            valid_indices = np.append(valid_indices, idx)
                    else:
                        filtered_centers.append(point)
                        valid_indices = np.append(valid_indices, idx)

            new_centers = np.array(filtered_centers)
            new_areas = sorted_areas[valid_indices]
            new_intensities = sorted_intensities[valid_indices]

            # Fit centers to a grid
            grid_points, grid_weights = fit_grid(new_centers, grid_spacing_est)
            grid_coords = grid_weights / grid_spacing_est
            grid_coords[:, 1] -= np.min(grid_coords[:, 1])
            grid_coords[:, 0] -= np.min(grid_coords[:, 0])
            num_cols = int(np.max(grid_coords[:, 0]) + 1)
            num_rows = int(np.max(grid_coords[:, 1]) + 1)
            row_coords = np.round(grid_coords[:, 1]).astype(int)
            col_coords = np.round(grid_coords[:, 0]).astype(int)

        log_message(f"Detected grid: {num_rows} rows x {num_cols} cols")

        # Estimate dot radius
        marker_radii = np.sqrt(new_areas / np.pi)

        # Save marker tracking data.
        self.xdim: int = img_width
        self.ydim: int = img_height
        self.gridsz: tuple[int, int] = (num_cols, num_rows)
        self.grid_spacing: float = grid_spacing_est
        self.marker_mask: np.ndarray = marker_mask
        self.initial_marker_coord = [col_coords, row_coords]
        self.initial_marker_center: np.ndarray = new_centers
        self.marker_radius: np.ndarray = marker_radii
        self.marker_blackpt: np.ndarray = new_intensities
        self.marker_center: np.ndarray = new_centers
        self.marker_lastpos: np.ndarray = new_centers.copy()
        self.marker_currentpos: np.ndarray = new_centers.copy()

        # Optionally display detected markers.
        if self.do_plot:
            for i, center in enumerate(new_centers):
                cv2.circle(
                    marker_image,
                    (int(center[1]), int(center[0])),
                    radius=int(marker_radii[i]),
                    color=(0, 255, 0),
                )
            cv2.imshow(
                "Detected Markers",
                cv2.resize(
                    marker_image, (marker_image.shape[1] * 2, marker_image.shape[0] * 2)
                ),
            )
            cv2.waitKey()

    def track_markers(self, frame: np.ndarray) -> None:
        """
        Track markers in the given frame using normalized cross-correlation and mean-shift.

        Args:
            frame (np.ndarray): The current frame for marker tracking.
        """
        settings: dict[str, Any] = {
            "toplot": 5,
            "meanshift_max_iter": 3,
            "meanshift_min_step": 1,
            "template_factor": 1.5,
        }

        # Calculate maximum across all three color channels
        # the black dots will still be black in the grayscale image
        gray_frame = np.max(frame, axis=2)
        frame_height, frame_width = gray_frame.shape

        # Check that frame size matches the marker model.
        scale: float = 1.0
        if frame_height != self.ydim or frame_width != self.xdim:
            log_message(
                f"Frame size {frame_width}x{frame_height} differs from model size {self.xdim}x{self.ydim}"
            )
            log_message("Resize the model or images so they match.\n")
            exit(-1)
            scale = np.mean([frame_width / self.xdim, frame_height / self.ydim])

        # Update last marker positions.
        num_markers = len(self.marker_lastpos)
        self.marker_lastpos = copy.deepcopy(self.marker_currentpos)

        # Make template
        median_radius_scaled = scale * np.median(self.marker_radius)
        template_size = round(settings["template_factor"] * median_radius_scaled)
        x_grid, y_grid = np.meshgrid(
            np.arange(-template_size, template_size + 1),
            np.arange(-template_size, template_size + 1),
        )
        # Template: binary mask for points outside the circle.
        template = (x_grid**2 + y_grid**2) > (median_radius_scaled**2)
        # Smooth the template using a small Gaussian filter.
        template = gaussian_filter(template.astype(float), sigma=1)

        # Compute normalized cross-correlation.
        correlation = normxcorr2(template, gray_frame)
        half_temp = math.floor(template.shape[0] / 2)
        corr_height, corr_width = correlation.shape
        correlation = correlation[
            half_temp : corr_height - half_temp, half_temp : corr_width - half_temp
        ]

        # Optionally show the frame.
        plot_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 255)]
        if settings["toplot"] == 1:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                cv2.destroyWindow("Frame")

        # Mean-shift to update marker positions.
        pdists = np.zeros((num_markers, settings["meanshift_max_iter"]))
        last_positions = copy.deepcopy(self.marker_lastpos)
        marker_radii = self.marker_radius
        new_marker_positions = np.zeros(last_positions.shape)
        for marker_idx in range(num_markers):
            pos = scale * last_positions[marker_idx].copy()
            region_radius = scale * marker_radii[marker_idx]

            x0 = int(np.clip(round(pos[1] - region_radius), 0, frame_width - 1))
            x1 = int(np.clip(round(pos[1] + region_radius), 0, frame_width - 1))
            y0 = int(np.clip(round(pos[0] - region_radius), 0, frame_height - 1))
            y1 = int(np.clip(round(pos[0] + region_radius), 0, frame_height - 1))
            # Build meshgrid for the ROI.
            x_vals, y_vals = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))

            if settings["toplot"] == 1:
                cv2.rectangle(frame, (x0, y0), (x1, y1), plot_colors[0])
                enlarged_frame = cv2.resize(
                    frame, (frame.shape[1] * 2, frame.shape[0] * 2)
                )
                cv2.imshow("Frame", enlarged_frame)

            for itr in range(settings["meanshift_max_iter"]):
                weights = correlation[y0 : y1 + 1, x0 : x1 + 1] ** 2
                weights /= np.sum(weights)
                last_pos = pos.copy()
                pos[1] = np.sum(x_vals * weights)
                pos[0] = np.sum(y_vals * weights)

                x0 = int(np.clip(round(pos[1] - region_radius), 0, frame_width - 1))
                x1 = int(np.clip(round(pos[1] + region_radius), 0, frame_width - 1))
                y0 = int(np.clip(round(pos[0] - region_radius), 0, frame_height - 1))
                y1 = int(np.clip(round(pos[0] + region_radius), 0, frame_height - 1))
                x_vals, y_vals = np.meshgrid(
                    np.arange(x0, x1 + 1), np.arange(y0, y1 + 1)
                )
                shift_distance = np.linalg.norm(pos - last_pos)
                pdists[marker_idx, itr] = shift_distance

                if settings["toplot"] == 1:
                    cv2.rectangle(enlarged_frame, (x0, y0), (x1, y1), plot_colors[itr])
                if shift_distance < settings["meanshift_min_step"]:
                    break

            # Store the updated marker position.
            new_marker_positions[marker_idx] = pos / scale

        # Stop if mean-shift did not converge
        if np.max(pdists[:, -1] > settings["meanshift_min_step"]):
            import pdb

            pdb.set_trace()
            log_message("Mean-shift did not converge")

        self.marker_currentpos = new_marker_positions
        new_positions_scaled = scale * new_marker_positions
        if settings["toplot"] == 2:
            for c in range(num_markers):
                pt_start = (
                    int(scale * self.marker_center[c, 1]),
                    int(scale * self.marker_center[c, 0]),
                )
                pt_end = (
                    int(new_positions_scaled[c, 1]),
                    int(new_positions_scaled[c, 0]),
                )
                cv2.arrowedLine(frame, pt_start, pt_end, plot_colors[3])
                cv2.drawMarker(
                    frame,
                    int(scale * self.marker_center[c, 1]),
                    int(scale * self.marker_center[c, 0]),
                    color=(0, 0, 255),
                )
                cv2.drawMarker(
                    frame,
                    int(new_positions_scaled[c, 1]),
                    int(new_positions_scaled[c, 0]),
                    color=(0, 255, 0),
                )
            cv2.imshow(
                "Marker Frame",
                cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2)),
            )
            cv2.waitKey(5)

        self.marker_mask = self.create_markermask(
            frame, new_positions_scaled, marker_radii
        )

    def create_markermask(
        self, image: np.ndarray, centers: np.ndarray, radius: np.ndarray
    ) -> np.ndarray:
        """
        Create a binary marker mask with circles at marker centers.

        Args:
            image (np.ndarray): Input image.
            centers (np.ndarray): Marker coordinates.
            radius (np.ndarray): Marker radius.

        Returns:
            np.ndarray: Binary mask with markers drawn.
        """
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for i in range(len(centers)):
            cv2.circle(
                mask,
                (int(centers[i, 1]), int(centers[i, 0])),
                int(radius[i]),
                color=255,
                thickness=-1,
            )
        cv2.imshow("Marker Mask", mask)
        return mask

    def sort_centers(self, dot_centers: np.ndarray) -> np.ndarray:
        """
        Sort marker centers along grid lines based on their x and y coordinates.

        Args:
            dot_centers (np.ndarray): Array of marker centers (row, col).

        Returns:
            np.ndarray: Indices of the sorted marker centers.
        """
        # Lexicographically sort centers
        base_sorted_idx = np.lexsort((dot_centers[:, 1], dot_centers[:, 0]))
        sorted_centers = dot_centers[base_sorted_idx]

        # Extract the x and y coordinates of the sorted dot centers
        x_coords = sorted_centers[:, 0]
        y_coords = sorted_centers[:, 1]

        num_points = len(x_coords)
        grouped_indices = np.empty((0,), dtype=int)
        idx = 0

        # Group centers that are close in the x-dimension
        while idx < num_points - 1:
            group_indices = [base_sorted_idx[idx]]
            group_y = [y_coords[idx]]
            x_current = x_coords[idx]
            idx += 1
            while idx < num_points and (x_coords[idx] - x_current) < 10:
                group_indices.append(base_sorted_idx[idx])
                group_y.append(y_coords[idx])
                x_current = x_coords[idx]
                idx += 1
            if len(group_y) > 5:
                sorted_group = np.array(group_indices)[np.argsort(group_y)]
                grouped_indices = np.append(grouped_indices, sorted_group)
        return grouped_indices

    def assign_coordinates(
        self, dot_centers: np.ndarray
    ) -> tuple[int, int, np.ndarray, np.ndarray]:
        """
        Assign grid coordinates to dot centers based on proximity.

        Args:
            dot_centers (np.ndarray): Array of dot center coordinates.

        Returns:
            Tuple[int, int, np.ndarray, np.ndarray]:
                - Number of grid columns.
                - Number of grid rows.
                - Array of column indices.
                - Array of row indices.
        """
        x_coords = dot_centers[:, 0]
        y_coords = dot_centers[:, 1]
        num_points = len(y_coords)

        # Initialize grid coordinate arrays.
        row_indices = np.zeros(num_points, dtype=int)
        col_indices = np.zeros(num_points, dtype=int)

        col = 0
        row = 0
        point = 0
        while point < num_points - 1:
            row_indices[point] = row
            col_indices[point] = col
            x_current = x_coords[point]
            # Group centers that are close horizontally.
            while point < num_points - 1 and (x_coords[point + 1] - x_current) < 15:
                point += 1
                row += 1
                row_indices[point] = row
                col_indices[point] = col
                x_current = x_coords[point]
            point += 1
            col += 1
            row = 0
        # Ensure last point is assigned.
        if point == num_points - 1:
            row_indices[point] = row
            col_indices[point] = col

        num_cols = int(np.max(col_indices)) + 1
        num_rows = int(np.max(row_indices)) + 1
        return num_cols, num_rows, col_indices, row_indices

    def estimate_grid_spacing(self, centers: np.ndarray) -> float:
        """
        Estimate grid spacing from marker centers.

        Args:
            centers (np.ndarray): Marker centers.

        Returns:
            float: Estimated grid spacing (median of selected distances).
        """
        N = centers.shape[0]
        distances = np.zeros(4 * N)
        for i in range(N):
            point = centers[i, :]
            d = np.sqrt(
                (centers[:, 1] - point[1]) ** 2 + (centers[:, 0] - point[0]) ** 2
            )
            sorted_d = np.sort(d)
            distances[4 * i : 4 * i + 4] = sorted_d[1:5]
        return float(np.median(distances))
