#!usr/bin/python3.8

import cv2
import numpy as np
from gelsight import gsdevice
from markertracker import MarkerTracker
from scipy.interpolate import griddata
from gs_utils import poisson_dct_neumaan, demark, Visualize3D

def find_marker(gray):
    """Find marker mask from grayscale image"""
    mask = cv2.inRange(gray, 0, 70)
    return mask

class GridFlowTracker:
    def __init__(self, imgw=320, imgh=240):
        self.imgw = imgw
        self.imgh = imgh
        self.base_frame = None
        self.marker_tracker = None
        self.grid_vectors = None
        self.marker_centers = None
        self.initial_positions = None
        self.block_assignments = None
        self.vis3d = None
        self.mmpp = 0.0634  # mm per pixel for GelSight Mini
        self.dm_zero = np.zeros((imgh, imgw))  # Note: height first, then width
        self.dm_zero_counter = 0
        
    def initialize(self, frame):
        """Initialize the tracker with a base frame"""
        # Convert frame to float32 for processing
        img = np.float32(frame) / 255.0
        
        # Initialize marker tracker
        self.marker_tracker = MarkerTracker(img)
        
        # Store initial marker positions
        self.base_frame = frame.copy()
        self.marker_centers = self.marker_tracker.initial_marker_center
        self.initial_positions = self.marker_centers.copy()
        
        # Create grid structure
        self._create_grid()
        
        # Initialize 3D visualization
        self._init_3d_visualization()
        
    def _init_3d_visualization(self):
        """Initialize 3D visualization"""
        self.vis3d = Visualize3D(self.imgh, self.imgw, self.mmpp, '')
        
    def _update_3d_visualization(self, frame):
        """Update 3D visualization with current frame"""
        if self.vis3d is None:
            return
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find marker mask
        markermask = find_marker(gray)
        
        # Calculate gradients
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Dilate marker mask to include pixels around markers
        dilated_mm = cv2.dilate(markermask, np.ones((3,3), np.uint8), iterations=2)
        
        # Interpolate gradients at marker locations
        gx_interp, gy_interp = demark(gx, gy, dilated_mm)
        
        # Reconstruct depth map
        depth_map = poisson_dct_neumaan(gx_interp, gy_interp)
        
        # Ensure depth map has correct dimensions
        if depth_map.shape != (self.imgh, self.imgw):
            depth_map = cv2.resize(depth_map, (self.imgw, self.imgh))
        
        # Remove initial zero depth
        if self.dm_zero_counter < 50:
            self.dm_zero += depth_map
            if self.dm_zero_counter == 49:
                self.dm_zero /= self.dm_zero_counter
        if self.dm_zero_counter == 50:
            print('Ok to touch me now!')
        self.dm_zero_counter += 1
        depth_map = depth_map - self.dm_zero
        
        # Update 3D visualization
        self.vis3d.update(depth_map)
        
    def _create_grid(self):
        """Create 7x9 grid structure from detected markers"""
        # Get marker centers
        centers = self.marker_centers
        
        # Sort centers by x then y coordinates (OpenCV coordinate system)
        sorted_indices = np.lexsort((centers[:, 0], centers[:, 1]))
        sorted_centers = centers[sorted_indices]
        
        # Reshape into 7x9 grid
        self.grid_centers = sorted_centers.reshape(7, 9, 2)
        
        # Initialize block assignments
        self.block_assignments = np.zeros((7, 9), dtype=int)
        
        # Assign block indices
        block_idx = 0
        for i in range(0, 6, 2):  # Step by 2 to create non-overlapping blocks
            for j in range(0, 8, 2):
                # Assign block index to each marker in the block
                self.block_assignments[i:i+2, j:j+2] = block_idx
                block_idx += 1
        
        # Initialize grid vectors
        self.grid_vectors = np.zeros((7, 9, 2))
        
    def update(self, frame):
        """Update tracking with new frame"""
        if self.marker_tracker is None:
            return
            
        # Convert frame to float32
        img = np.float32(frame) / 255.0
        
        # Track markers
        self.marker_tracker.track_markers(img)
        
        # Get current marker positions
        current_pos = self.marker_tracker.marker_currentpos
        
        # Calculate flow vectors for each grid cell
        for i in range(7):
            for j in range(9):
                idx = i * 9 + j
                if idx < len(current_pos):
                    # Calculate displacement from initial position
                    displacement = current_pos[idx] - self.initial_positions[idx]
                    self.grid_vectors[i,j] = displacement
        
        # Update 3D visualization
        self._update_3d_visualization(frame)
                
    def create_vector_heatmap(self):
        """Create a heatmap of the vector field"""
        # Create a grid for interpolation
        x = np.linspace(0, self.imgw, self.imgw)
        y = np.linspace(0, self.imgh, self.imgh)
        X, Y = np.meshgrid(x, y)
        
        # Get all valid vector points and their positions
        points = []
        values = []
        for i in range(7):
            for j in range(9):
                if i * 9 + j < len(self.marker_tracker.marker_currentpos):
                    # Get the center position
                    center = self.grid_centers[i,j]
                    points.append([center[1], center[0]])  # Swap x,y for OpenCV
                    
                    # Calculate magnitude and phase
                    vector = self.grid_vectors[i,j]
                    magnitude = np.linalg.norm(vector)
                    phase = np.arctan2(vector[0], vector[1])  # Note: swapped x,y for correct angle
                    
                    # Normalize phase to [0, 1] range
                    phase_norm = (phase + np.pi) / (2 * np.pi)
                    values.append([phase_norm, magnitude])
        
        if not points or not values:
            return None
            
        points = np.array(points)
        values = np.array(values)
        
        # Interpolate phase and magnitude separately
        phase_grid = griddata(points, values[:,0], (X, Y), method='cubic', fill_value=0)
        mag_grid = griddata(points, values[:,1], (X, Y), method='cubic', fill_value=0)
        
        # Normalize magnitude to [0, 1] range
        if np.max(mag_grid) > 0:
            mag_grid = mag_grid / np.max(mag_grid)
        
        # Create HSV image
        hsv = np.zeros((self.imgh, self.imgw, 3), dtype=np.uint8)
        hsv[..., 0] = phase_grid * 180  # Hue: phase (0-180)
        hsv[..., 1] = mag_grid * 255    # Saturation: magnitude
        hsv[..., 2] = 255               # Value: full brightness
        
        # Convert to BGR
        heatmap = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return heatmap
        
    def visualize(self, frame):
        """Visualize tracking results on frame"""
        vis_frame = frame.copy()
        
        # Draw flow vectors
        for i in range(7):
            for j in range(9):
                # Get the current position of the marker
                idx = i * 9 + j
                if idx < len(self.marker_tracker.marker_currentpos):
                    current_pos = self.marker_tracker.marker_currentpos[idx]
                    # Convert coordinates to OpenCV format
                    start_x, start_y = int(self.initial_positions[idx][1]), int(self.initial_positions[idx][0])
                    end_x, end_y = int(current_pos[1]), int(current_pos[0])
                    
                    # Only draw if there's significant movement
                    if np.linalg.norm(self.grid_vectors[i,j]) > -1.0:
                        # Draw the vector
                        cv2.arrowedLine(vis_frame, (start_x,start_y), (end_x,end_y), (0,0,255), 2, tipLength=0.2)
                        
                        # Draw marker at current position
                        cv2.circle(vis_frame, (end_x,end_y), 3, (255,0,0), -1)
                    
        return vis_frame

def main():
    # Initialize parameters
    imgw = 320
    imgh = 240
    USE_MINI_LIVE = True
    
    # Initialize camera
    if USE_MINI_LIVE:
        gs = gsdevice.Camera("GelSight Mini")
        gs.connect()
    else:
        cap = cv2.VideoCapture('data/mini_example.avi')
        
    # Initialize tracker
    tracker = GridFlowTracker(imgw, imgh)
    
    # Get initial frame
    if USE_MINI_LIVE:
        frame = gs.get_raw_image()
    else:
        ret, frame = cap.read()
        if not ret:
            print("Error reading video")
            return
            
    # Initialize tracker with first frame
    tracker.initialize(frame)
    
    try:
        while True:
            # Get new frame
            if USE_MINI_LIVE:
                frame = gs.get_image()
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                    
            # Update tracker
            tracker.update(frame)
            
            # Visualize results
            vis_frame = tracker.visualize(frame)
            
            # Create and show vector heatmap
            #heatmap = tracker.create_vector_heatmap()
            #f heatmap is not None:
            #    cv2.imshow('Vector Field Heatmap', cv2.resize(heatmap, (2*heatmap.shape[1], 2*heatmap.shape[0])))
            
            # Show results
            cv2.imshow('Grid Flow Tracking', cv2.resize(vis_frame, (2*vis_frame.shape[1], 2*vis_frame.shape[0])))
            
            # Handle keypresses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset base frame
                tracker.initialize(frame)
                
    except KeyboardInterrupt:
        print('Interrupted!')
    finally:
        if USE_MINI_LIVE:
            gs.stop_video()
        else:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 