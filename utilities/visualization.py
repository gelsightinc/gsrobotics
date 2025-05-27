import numpy as np
import open3d
from typing import Optional


class Visualize3D:
    """
    Class designed to visualize 3D point cloud data generated from a depth map.

    This class uses the Open3D library to visualize a 3D point cloud.
    The depth map is used to set the z-coordinates of the points and to assign colors.

    Attributes:
        pointcloud_size_x (int): Number of points along the x-axis.
        pointcloud_size_y (int): Number of points along the y-axis.
        config (ConfigModel): Global configuration.
        save_path (str): Directory path used for saving point cloud data.
        file_counter (int): Counter used for naming saved point cloud files.
        pointcloud_X (np.ndarray): 2D grid array for x coordinates.
        pointcloud_Y (np.ndarray): 2D grid array for y coordinates.
        points (np.ndarray): Array of 3D points of shape (pointcloud_size_x * pointcloud_size_y, 3).
        pointcloud (open3d.geometry.PointCloud): Open3D point cloud object.
        visualizer (open3d.visualization.Visualizer): Open3D visualizer instance.
    """

    def __init__(
        self,
        pointcloud_size_x: int,
        pointcloud_size_y: int,
        save_path: str,
        window_width: int,
        window_height: int,
    ) -> None:
        """
        Initialize the Visualize3D class.

        Args:
            pointcloud_size_x (int): Number of points along the x-axis.
            pointcloud_size_y (int): Number of points along the y-axis.
            save_path (str): Directory path to save the point cloud files.
            window_width (int): Width of pointcloud window
            window_height (int): Height of pointcloud window
        """
        self.pointcloud_size_x: int = pointcloud_size_x
        self.pointcloud_size_y: int = pointcloud_size_y
        self.save_path: str = save_path
        self.file_counter: int = 0

        # Initialize Open3D components (grid, point cloud, and visualizer)
        self.init_open3D(window_width=window_width, window_height=window_height)

    def init_open3D(self, window_width: int, window_height: int) -> None:
        """
        Initialize the 3D grid and Open3D visualization components.

        This method:
          - Creates a grid of x and y coordinates.
          - Computes an initial depth (z) map using np.sin as a placeholder.
          - Flattens the grid into an array of 3D points.
          - Sets up the Open3D point cloud and visualizer window.

        Args:
            window_width (int): width of window showing pointcloud
            window_height (int): hight of window showing pointcloud
        """

        # Create a range of x and y coordinates.
        pointcloud_range_x = np.arange(self.pointcloud_size_x)
        pointcloud_range_y = np.arange(self.pointcloud_size_y)

        # Create a meshgrid from x and y coordinates.
        self.pointcloud_X, self.pointcloud_Y = np.meshgrid(
            pointcloud_range_x, pointcloud_range_y
        )

        # Generate an initial depth map (z-coordinates) using a sine function as a placeholder.
        pointcloud_Z = np.sin(self.pointcloud_X)

        # Initialize the points array with zeros.
        self.points = np.zeros((self.pointcloud_size_x * self.pointcloud_size_y, 3))
        # Set x and y coordinates from the meshgrid.
        self.points[:, 0] = self.pointcloud_X.flatten()
        self.points[:, 1] = self.pointcloud_Y.flatten()

        # Set the z-coordinates for points based on the depth map.
        self.depth2points(pointcloud_Z)

        # Create an Open3D point cloud and assign the points.
        self.pointcloud = open3d.geometry.PointCloud()
        self.pointcloud.points = open3d.utility.Vector3dVector(self.points)

        # Initialize the visualizer, create a window, and add the point cloud.
        self.visualizer = open3d.visualization.Visualizer()

        self.visualizer.create_window(width=window_width, height=window_height)
        self.visualizer.add_geometry(self.pointcloud)

        # setup background color and initial camera view
        render_options = self.visualizer.get_render_option()
        render_options.background_color = np.array([0.05, 0.05, 0.05])
        view_control = self.visualizer.get_view_control()
        view_control.set_front([0, 0, -1])
        view_control.set_up([0, -1, 0])
        center = [np.mean(self.points[:, 0]), np.mean(self.points[:, 1]), 0]
        view_control.set_lookat(center)

    def depth2points(self, depth_map: np.ndarray) -> None:
        """
        Update the z-coordinates of the point cloud based on the given depth map.

        Args:
            depth_map (np.ndarray): A 2D array representing the depth map.
        """
        self.points[:, 2] = depth_map.flatten()

    def update(
        self,
        depth_map: np.ndarray,
        gradient_x: Optional[np.ndarray] = None,
        gradient_y: Optional[np.ndarray] = None,
    ):
        """
        Update the point cloud with a new depth map, determine colors,
        update the visualization, and optionally save the point cloud.

        This method:
          - Updates the z-coordinates of the points using the new depth map.
          - Computes the gradients of the depth map (if not provided) to derive color information.
          - Updates the Open3D point cloud and visualizer.
          - Saves the point cloud to a file if a save path is provided.

        Args:
            depth_map (np.ndarray): A new depth map (2D array) to update the point cloud.
            gradient_x (Optional[np.ndarray]): Optional gradient in the x-direction.
            gradient_y (Optional[np.ndarray]): Optional gradient in the y-direction.
        """
        # Update the z-coordinates of the points based on the new depth map.
        self.depth2points(depth_map)

        # Compute gradients of the depth map along the x and y axes if not provided.
        if gradient_x is None or gradient_y is None:
            gradient_x, gradient_y = np.gradient(depth_map)

        # Scale the gradients with an arbitrary factor of 0.5.
        # gradient_x, gradient_y = gradient_x * 0.5, gradient_y * 0.5

        # Create single-channel color arrays based on the gradients.
        colors_x = np.clip(0.5 * gradient_x + 0.5, 0, 1)
        colors_y = np.clip(0.5 * gradient_y + 0.5, 0, 1)
        # Flatten the color arrays to match the number of points.
        colors_x_flat = colors_x.flatten()
        colors_y_flat = colors_y.flatten()

        # Create an RGB color array:
        colors = np.zeros((self.points.shape[0], 3))
        colors[:, 0] = colors_x_flat
        colors[:, 1] = colors_y_flat
        colors[:, 2] = (colors_x_flat + colors_y_flat) / 2

        # Update the point cloud's points and colors.
        self.pointcloud.points = open3d.utility.Vector3dVector(self.points)
        self.pointcloud.colors = open3d.utility.Vector3dVector(colors)

        # Refresh the visualizer.
        self.visualizer.update_geometry(self.pointcloud)
        self.visualizer.poll_events()
        self.visualizer.update_renderer()

        # Save the current point cloud to a file if a save path is provided.
        if self.save_path != "":
            filename = f"{self.save_path}/pc_{self.file_counter}.pcd"
            open3d.io.write_point_cloud(filename, self.pointcloud)

        # Increment the file counter.
        self.file_counter += 1

    def save_pointcloud(self) -> None:
        """
        Save the current point cloud to a file.

        The file is saved using the current counter value as a name postfix.
        """
        filename = f"{self.save_path}/pointcloud_{self.file_counter}.pcd"
        open3d.io.write_point_cloud(filename, self.pointcloud)
