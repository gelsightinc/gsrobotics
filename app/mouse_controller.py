import pyautogui
import numpy as np
import time

class MouseController:
    def __init__(self, scaling_factor=150.0, min_threshold=0.05, max_speed=2000, smoothing_factor=0.5):
        """
        Initialize mouse controller
        
        Args:
            scaling_factor: Factor to scale the vector magnitude to pixel movement speed.
            min_threshold: Minimum vector magnitude to trigger movement.
            max_speed: Maximum speed in pixels per second.
            smoothing_factor: Factor for exponential moving average smoothing (0 = no smoothing, 1 = immediate).
        """
        # Disable pyautogui's fail-safe for smoother control during development/testing
        # Be cautious with this in production environments.
        pyautogui.FAILSAFE = False 
        
        self.scaling_factor = scaling_factor
        self.min_threshold = min_threshold
        self.max_speed = max_speed  # pixels per second
        self.smoothing_factor = smoothing_factor
        
        self.last_update_time = time.time()
        self.smoothed_dx = 0.0
        self.smoothed_dy = 0.0
        
    def calculate_average_vector(self, grid_vectors, contact_mask, grid_centers):
        """
        Calculate the weighted average vector within the contact area.
        Weights based on proximity to contact mask centroid.

        Args:
            grid_vectors (np.array): 7x9x2 array of marker displacement vectors.
            contact_mask (np.array): Binary mask of the contact area (imgh x imgw).
            grid_centers (np.array): 7x9x2 array of marker grid center coordinates (y, x).

        Returns:
            tuple: (average_vector, average_magnitude)
        """
        if contact_mask is None or not contact_mask.any():
            return np.zeros(2), 0.0

        # Find indices of markers within or near the contact mask
        contact_indices = []
        marker_coords_in_mask = []

        grid_rows, grid_cols = grid_vectors.shape[:2]

        # Efficiently check which grid centers fall within the mask bounding box first
        rows, cols = np.where(contact_mask > 0)
        if not rows.size > 0:
             return np.zeros(2), 0.0 # No contact
        min_r, max_r = np.min(rows), np.max(rows)
        min_c, max_c = np.min(cols), np.max(cols)

        relevant_vectors = []
        relevant_centers = []

        for r in range(grid_rows):
            for c in range(grid_cols):
                center_y, center_x = grid_centers[r, c]
                # Check if marker center is within bounding box of contact area
                if min_r <= center_y <= max_r and min_c <= center_x <= max_c:
                     # More precise check: Is the marker center pixel actually in the mask?
                     cy, cx = int(round(center_y)), int(round(center_x))
                     if 0 <= cy < contact_mask.shape[0] and 0 <= cx < contact_mask.shape[1] and contact_mask[cy, cx]:
                        relevant_vectors.append(grid_vectors[r, c])
                        relevant_centers.append(grid_centers[r, c])


        if not relevant_vectors:
            return np.zeros(2), 0.0

        relevant_vectors = np.array(relevant_vectors)
        # relevant_centers = np.array(relevant_centers) # Not needed for simple average

        # --- Simple Average Calculation ---
        average_vector = np.mean(relevant_vectors, axis=0)
        average_magnitude = np.linalg.norm(average_vector)

        # --- Optional: Weighted Average Calculation (more complex) ---
        # # Calculate centroid of the contact mask for weighting
        # M = cv2.moments(contact_mask)
        # if M["m00"] != 0:
        #     cX = int(M["m10"] / M["m00"])
        #     cY = int(M["m01"] / M["m00"])
        #     contact_centroid = np.array([cY, cX])

        #     # Calculate distances from each relevant marker to the contact centroid
        #     distances = np.linalg.norm(relevant_centers - contact_centroid, axis=1)
            
        #     # Create weights (closer markers get higher weight) - inverse distance squared
        #     weights = 1.0 / (distances**2 + 1e-6) # Add epsilon to avoid division by zero
        #     weights /= np.sum(weights) # Normalize weights

        #     # Calculate weighted average vector
        #     average_vector = np.sum(relevant_vectors * weights[:, np.newaxis], axis=0)
        #     average_magnitude = np.linalg.norm(average_vector)
        # else: # Fallback to simple average if centroid calculation fails
        #      average_vector = np.mean(relevant_vectors, axis=0)
        #      average_magnitude = np.linalg.norm(average_vector)
        # --- End Optional Weighted Average ---


        return average_vector, average_magnitude


    def update_mouse(self, vector, magnitude):
        """
        Move mouse based on the calculated average vector and magnitude.

        Args:
            vector (np.array): 2D vector [dy, dx] from gel coordinates.
            magnitude (float): Magnitude of the vector.
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        if dt == 0: # Avoid division by zero if updates are too fast
             return
        self.last_update_time = current_time

        if magnitude < self.min_threshold:
            # Apply decay to smoothed velocity when below threshold
            self.smoothed_dx *= (1.0 - self.smoothing_factor * 5 * dt) # Faster decay when still
            self.smoothed_dy *= (1.0 - self.smoothing_factor * 5 * dt)
            # Ensure velocity goes to zero eventually
            if abs(self.smoothed_dx) < 1e-3: self.smoothed_dx = 0
            if abs(self.smoothed_dy) < 1e-3: self.smoothed_dy = 0

        else:
            # Gel vector dy corresponds to screen -dy (up/down)
            # Gel vector dx corresponds to screen dx (left/right)
            target_dx_speed = vector[1] * self.scaling_factor * magnitude
            target_dy_speed = -vector[0] * self.scaling_factor * magnitude # Flip y-axis

            # Apply smoothing (Exponential Moving Average)
            self.smoothed_dx = (1.0 - self.smoothing_factor) * self.smoothed_dx + self.smoothing_factor * target_dx_speed
            self.smoothed_dy = (1.0 - self.smoothing_factor) * self.smoothed_dy + self.smoothing_factor * target_dy_speed


        # Limit speed
        current_speed = np.sqrt(self.smoothed_dx**2 + self.smoothed_dy**2)
        if current_speed > self.max_speed:
            scale = self.max_speed / current_speed
            move_dx = self.smoothed_dx * scale * dt
            move_dy = self.smoothed_dy * scale * dt
            # Update smoothed values to reflect clipping
            self.smoothed_dx *= scale
            self.smoothed_dy *= scale
        elif current_speed < 1.0 and magnitude < self.min_threshold : # Prevent tiny drifts when still
             move_dx = 0
             move_dy = 0
             self.smoothed_dx = 0
             self.smoothed_dy = 0
        else:
            move_dx = self.smoothed_dx * dt
            move_dy = self.smoothed_dy * dt

        # Move mouse relatively if there's movement
        if abs(move_dx) > 0.1 or abs(move_dy) > 0.1:
             try:
                 # Using moveRel for delta movement, tends to be smoother
                 pyautogui.moveRel(int(round(move_dx)), int(round(move_dy)), duration=0) 
             except pyautogui.FailSafeException:
                 print("Mouse movement blocked by fail-safe (moved to screen corner).")
             except Exception as e:
                 print(f"Error moving mouse: {e}") 