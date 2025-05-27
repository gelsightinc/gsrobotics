import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
from utilities.poisson_solver import poisson_dct_neumaan
from utilities.image_processing import mask_from_range, remove_masked_area
from utilities.logger import log_message
from typing import Optional


class RGB2NormNet(nn.Module):
    def __init__(self):
        super(RGB2NormNet, self).__init__()
        input_size = 5
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 2)
        self.drop_layer = nn.Dropout(p=0.05)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc2(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc3(x))
        x = self.drop_layer(x)
        x = self.fc4(x)
        return x


class Reconstruction3D:
    """
    Class for reconstruction of a 3D surface map from an input image using a neural network.

    Attributes:
        device (torch.device): The computation target device ('cuda' or 'cpu').
        depth_map_zero (np.ndarray): Accumulated zero depth map with shape (image_height, image_width).
        depth_map_zero_counter (int): Counter for frames used to calculate initial zero-depth.
        net (Optional[torch.nn.Module]): Neural network model for depth estimation.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        use_gpu: bool = False,
    ) -> None:
        """
        Initialize the Reconstruction3D class.

        Args:
            image_width (int): Input image width.
            image_height (int): Input image height.
            use_gpu (bool, optional): If True, attempts to use a CUDA device if available.
                Defaults to False.
        """
        # Select computation device.
        self.device: torch.device = (
            torch.device("cuda")
            if use_gpu and torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.depth_map_zero_counter: int = 0
        self.depth_map_zero: np.ndarray = np.zeros((image_height, image_width))
        self.net: Optional[torch.nn.Module] = None

    def load_nn(self, net_path: str) -> Optional[torch.nn.Module]:
        """
        Load the neural network model from a file.

        This function loads the state dictionary from a checkpoint file,
        instantiates the model, moves it to the appropriate device, and stores it.

        Args:
            net_path (str): Path to the stored model state file (.pt).

        Returns:
            Optional[torch.nn.Module]: The loaded neural network model, or None if the file does not exist.
        """
        if not os.path.isfile(net_path):
            log_message(f"Error opening {net_path}. Path does not exist.")
            return None

        net = RGB2NormNet().float().to(self.device)
        state = torch.load(net_path, map_location=self.device)
        net.load_state_dict(state["state_dict"])
        self.net = net
        self.net.eval()
        return self.net

    def get_depthmap(
        self,
        image: np.ndarray,
        markers_threshold: Optional[tuple[int, int]] = None,
        contact_mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the depth map from an input image using the neural network.

        The method extracts normalized RGB values and pixel positions from a contact mask,
        uses the neural network to predict surface normals, and reconstructs a depth map
        via Poisson integration. Optionally, marker interpolation is applied.

        Args:
            image (np.ndarray): The input RGB image.
            markers_threshold (Optional[tuple[int, int]]): Tuple (min_mask_thresh, max_mask_thresh) which
                defines values to be masked. If None, no masking occures.
            contact_mask (Optional[np.ndarray], optional): A contact mask defining regions of interest.
                If None, the whole image is treated as the region of interest. Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                A tuple containing:
                - depth_map: The computed depth map.
                - contact_mask: Binary mask defining contact area.
                - gradient_x: Gradient map with respect to x (rows).
                - gradient_y: Gradient map with respect to y (columns).
        """

        # If no contact mask is provided, use the whole image.
        if contact_mask is None:
            contact_mask = np.ones(image.shape[:2], dtype=bool)

        image_height, image_width = image.shape[:2]

        if markers_threshold:
            # Convert to grayscale for mask computation.
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            marker_mask = mask_from_range(
                image=gray_image,
                lower=markers_threshold[0],
                upper=markers_threshold[1],
            )
            # Invert marker mask to get the contact mask.
            contact_mask = np.logical_not(marker_mask)

        # Initialize arrays for normal components and depth map.
        normal_x = np.zeros(image.shape[:2])
        normal_y = np.zeros(image.shape[:2])
        depth_map = np.zeros(image.shape[:2])

        # Get normalized RGB values from the image at contact mask.
        image_contact_normalized = image[np.where(contact_mask)] / 255.0

        # Get normalized pixel positions.
        px_positions_normalized = np.vstack(np.where(contact_mask)).T
        px_positions_normalized[:, 0] = px_positions_normalized[:, 0] / image_height
        px_positions_normalized[:, 1] = px_positions_normalized[:, 1] / image_width

        # Combine normalized RGB and normalized pixel positions into a feature matrix.
        features = np.column_stack((image_contact_normalized, px_positions_normalized))
        features_tensor = torch.from_numpy(features).float().to(self.device)

        # Run forward method on the NN model to obtain normals.
        with torch.no_grad():
            out = self.net(features_tensor)

        # Fill normal arrays with network output.
        normal_x[np.where(contact_mask)] = out[:, 0].cpu().detach().numpy()
        normal_y[np.where(contact_mask)] = out[:, 1].cpu().detach().numpy()

        # Calculate z-axis normal.
        normal_z = np.sqrt(1 - np.power(normal_x, 2) - np.power(normal_y, 2))
        normal_z[np.isnan(normal_z)] = np.nanmean(normal_z)

        gradient_x = -normal_x / normal_z
        gradient_y = -normal_y / normal_z

        # Adjust gradients if marker interpolation is enabled.
        if markers_threshold:
            gradient_x, gradient_y = remove_masked_area(
                gx=gradient_x, gy=gradient_y, mask=marker_mask
            )

        # Reconstruct depth from gradients using Poisson integration.
        depth_map = poisson_dct_neumaan(gx=gradient_x, gy=gradient_y)
        depth_map = np.reshape(depth_map, (image_height, image_width))

        # Update zero depth map for the first 50 frames.
        if self.depth_map_zero_counter < 50:

            self.depth_map_zero += depth_map
            if self.depth_map_zero_counter == 0:
                log_message("Zeroing depth. Please do not touch the sensor...")
            if self.depth_map_zero_counter == 49:

                self.depth_map_zero /= (
                    self.depth_map_zero_counter + 1
                )  # +1 to include current frame

        if self.depth_map_zero_counter == 50:
            log_message("Sensor is ready to use.")

        self.depth_map_zero_counter += 1

        # Subtract the accumulated zero depth for normalization.
        depth_map -= self.depth_map_zero

        return depth_map, contact_mask, gradient_x, gradient_y
