import numpy as np
from datetime import datetime
from utilities.logger import log_message
import os


class MarkerDataLogger:
    """
    Class to log and save marker positions over time.

    Each call to add_frame() appends the current marker positions to the data.
    When save_data() is called, the accumulated data is saved at desired path
    in .npy and/or .csv format.
    """

    def __init__(self, num_markers: int = None) -> None:
        """
        Initialize the MarkerDataLogger.

        Args:
            num_markers (int): The number of marker points expected per frame.
                If None, data from first frame will define size.
        """

        self.num_markers = num_markers
        self.frames: list[np.ndarray] = []

    def add_frame(self, positions: np.ndarray) -> None:
        """
        Append a new frame of marker positions. If data shape does not match
            (num_markers, 2) then appending is ignored.
        """
        if not self.frames:
            self.frames: list[np.ndarray] = []
        if not self.num_markers:
            self.num_markers = positions.shape[0]

        if positions.shape != (self.num_markers, 2):
            log_message(
                f"Expected data shape ({self.num_markers}, 2), got {positions.shape}. Skipping"
            )

            return

        self.frames.append(positions.copy())

    def save_data(
        self, save_npy: bool = True, save_csv: bool = False, folder: str = "."
    ) -> None:
        """
        Save the accumulated marker data and clear the stored frames.

        Data is saved as a .npy file and/or .csv file with a filename
        marker_tracking_{date}.{ext} where date is formatted as YYYYMMDD_HHMMSS.
        The data is saved as a NumPy array of shape (num_frames, num_markers, 3).
        For CSV, the data is reshaped to (num_frames, num_markers*3).

        Args:
            save_npy (bool, optional): If True, save data in .npy format.
            save_csv (bool, optional): If True, save data in CSV format.
            folder (str, optional): The folder where files will be saved.
        """
        if not self.frames:
            log_message("No marker data to store.")
            return

        data = np.stack(self.frames, axis=0)

        # Create filename based on current date and time.
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"marker_tracking_{date_str}"

        if save_npy:
            numpy_filepath = os.path.join(folder, f"{filename}.npy")
            np.save(numpy_filepath, data)
            log_message(f"Saved data as NumPy: {numpy_filepath}")

        if save_csv:
            # Reshape data to 2D: each row is a frame, columns are flattened (num_markers*3)
            csv_data = data.reshape(data.shape[0], self.num_markers * 2)
            csv_filepath = os.path.join(folder, f"{filename}.csv")
            # Use fmt to specify formatting for each number (e.g. '%.4f')
            np.savetxt(
                csv_filepath,
                csv_data,
                delimiter=",",
                fmt="%.2f",
                header=f"Marker tracking data: {data.shape[0]} frames, {self.num_markers} markers (flattened as x,y for each marker)",
            )
            log_message(f"Saved data as CSV: {csv_filepath}")

        self.frames = None
