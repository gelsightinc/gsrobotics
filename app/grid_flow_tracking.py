#!usr/bin/python3.8

import cv2
import numpy as np
from gelsight import gsdevice
from markertracker import MarkerTracker
from scipy.interpolate import griddata
from gs_utils import poisson_dct_neumaan, demark, Visualize3D
from gs3drecon import Reconstruction3D
import os

def find_marker(gray):
    """Find marker mask from grayscale image"""
    mask = cv2.inRange(gray, 0, 70)
    return mask

class GridFlowTracker:
    def __init__(self, dev, imgw=320, imgh=240):
        self.dev = dev
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

        # Path to 3d model
        path = '.'

        # Set the camera resolution
        mmpp = 0.0634  # mini gel 18x24mm at 240x320
        self.mmpp = mmpp

        # the device ID can change after unplugging and changing the usb ports.
        # on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
        net_file_path = 'nnmini.pt'

        ''' Load neural network '''
        model_file_path = path
        net_path = os.path.join(model_file_path, net_file_path)
        print('net path = ', net_path)
        GPU = False
        if GPU:
            gpuorcpu = "cuda"
        else:
            gpuorcpu = "cpu"

        self.nn = Reconstruction3D(dev)
        self.net = self.nn.load_nn(net_path, gpuorcpu)

        print(f"Mini GelSight resolution: {self.mmpp} mm/pixel")
        # Print dimensions in pixels and mm
        print(f"Image dimensions in pixels: {self.imgh}x{self.imgw}")
        print(f"Image dimensions in mm: {self.imgh * self.mmpp:.1f}x{self.imgw * self.mmpp:.1f}")
        
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
        self.vis3d = Visualize3D(self.imgw, self.imgh, self.mmpp, '')

        
    
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
        
        # Get depth map using neural network
        self.depth_map = self.nn.get_depthmap(frame, True, cm=None)
        
 
                
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
        
    def find_contact_area(self, depth_map):
        """Find contact area from depth map and fit ellipse"""
        # Threshold the depth map to find contact regions
        # Use Otsu's method to find the threshold automatically
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, contact_mask = cv2.threshold(depth_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up the mask with morphological operations
        kernel = np.ones((2,2), np.uint8)
        #contact_mask = cv2.morphologyEx(contact_mask, cv2.MORPH_CLOSE, kernel)
        #contact_mask = cv2.morphologyEx(contact_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(contact_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
            
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit an ellipse to the contour if it has enough points
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            return largest_contour, ellipse
        
        return largest_contour, None
        
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
        
        # Find and draw contact area
        contour, ellipse = self.find_contact_area(self.depth_map)
        if contour is not None:
            pass
            # Draw the contour
            #cv2.drawContours(vis_frame, [contour], -1, (0,255,0), 2)
            
            if ellipse is not None:
                # Draw the ellipse
                center, axes, angle = ellipse
                center = tuple(map(int, center))
                axes = tuple(map(int, axes))
                cv2.ellipse(vis_frame, center, axes, angle, 0, 360, (255,255,0), 2)
                
                # Draw major and minor axes
                major_angle = np.deg2rad(angle)
                minor_angle = major_angle + np.pi/2
                
                # Major axis
                major_dx = np.cos(major_angle) * axes[0]
                major_dy = np.sin(major_angle) * axes[0]
                pt1 = (int(center[0] - major_dx), int(center[1] - major_dy))
                pt2 = (int(center[0] + major_dx), int(center[1] + major_dy))
                cv2.line(vis_frame, pt1, pt2, (0,255,255), 2)
                
                # Minor axis
                minor_dx = np.cos(minor_angle) * axes[1]
                minor_dy = np.sin(minor_angle) * axes[1]
                pt1 = (int(center[0] - minor_dx), int(center[1] - minor_dy))
                pt2 = (int(center[0] + minor_dx), int(center[1] + minor_dy))
                cv2.line(vis_frame, pt1, pt2, (0,255,255), 2)
                
                # Add text with measurements
                major_axis_mm = axes[0] * self.mmpp * 2  # Convert to mm
                minor_axis_mm = axes[1] * self.mmpp * 2  # Convert to mm
                area_mm2 = np.pi * major_axis_mm * minor_axis_mm / 4  # Ellipse area
                
                cv2.putText(vis_frame, f'Major: {major_axis_mm:.1f}mm', 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                cv2.putText(vis_frame, f'Minor: {minor_axis_mm:.1f}mm', 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                cv2.putText(vis_frame, f'Area: {area_mm2:.1f}mm2', 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        
        # Update 3D visualization
        self.vis3d.update(self.depth_map)
        
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
    tracker = GridFlowTracker(gs, imgw, imgh)
    
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
            cv2.imshow('Grid Flow Tracking', vis_frame)
            
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