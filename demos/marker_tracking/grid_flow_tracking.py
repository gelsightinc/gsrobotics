#!usr/bin/python3.8

import cv2
import numpy as np
from gelsight import gsdevice
from markertracker import MarkerTracker

class GridFlowTracker:
    def __init__(self, imgw=320, imgh=240, grid_size=3):
        self.imgw = imgw
        self.imgh = imgh
        self.grid_size = grid_size
        self.base_frame = None
        self.marker_tracker = None
        self.grid_vectors = None
        self.grid_centers = None
        
    def initialize(self, frame):
        """Initialize the tracker with a base frame"""
        # Convert frame to float32 for processing
        img = np.float32(frame) / 255.0
        
        # Initialize marker tracker
        self.marker_tracker = MarkerTracker(img)
        
        # Store initial marker positions
        self.base_frame = frame.copy()
        self.marker_centers = self.marker_tracker.initial_marker_center
        
        # Create grid structure
        self._create_grid()
        
    def _create_grid(self):
        """Create NxN grid structure from detected markers"""
        # Get marker centers
        centers = self.marker_centers
        
        # Sort centers by y then x coordinates
        sorted_indices = np.lexsort((centers[:, 1], centers[:, 0]))
        sorted_centers = centers[sorted_indices]
        
        # Reshape into grid
        n_markers = len(sorted_centers)
        grid_shape = (n_markers // self.grid_size, self.grid_size)
        self.grid_centers = sorted_centers.reshape(grid_shape[0], grid_shape[1], 2)
        
        # Initialize grid vectors
        self.grid_vectors = np.zeros((grid_shape[0], grid_shape[1], 2))
        
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
        for i in range(self.grid_centers.shape[0]):
            for j in range(self.grid_centers.shape[1]):
                base_pos = self.grid_centers[i,j]
                curr_pos = current_pos[i*self.grid_size + j]
                self.grid_vectors[i,j] = curr_pos - base_pos
                
    def visualize(self, frame):
        """Visualize tracking results on frame"""
        vis_frame = frame.copy()
        
        # Draw grid lines
        for i in range(self.grid_centers.shape[0]):
            for j in range(self.grid_centers.shape[1]):
                # Draw grid cell
                if j < self.grid_centers.shape[1]-1:
                    pt1 = tuple(self.grid_centers[i,j].astype(int))
                    pt2 = tuple(self.grid_centers[i,j+1].astype(int))
                    cv2.line(vis_frame, pt1, pt2, (0,255,0), 1)
                if i < self.grid_centers.shape[0]-1:
                    pt1 = tuple(self.grid_centers[i,j].astype(int))
                    pt2 = tuple(self.grid_centers[i+1,j].astype(int))
                    cv2.line(vis_frame, pt1, pt2, (0,255,0), 1)
                    
                # Draw flow vectors
                if self.grid_vectors is not None:
                    start = tuple(self.grid_centers[i,j].astype(int))
                    end = tuple((self.grid_centers[i,j] + self.grid_vectors[i,j]).astype(int))
                    cv2.arrowedLine(vis_frame, start, end, (0,0,255), 2, tipLength=0.2)
                    
        return vis_frame

def main():
    # Initialize parameters
    imgw = 320
    imgh = 240
    USE_MINI_LIVE = True
    GRID_SIZE = 3  # NxN grid size
    
    # Initialize camera
    if USE_MINI_LIVE:
        gs = gsdevice.Camera("GelSight Mini")
        gs.connect()
    else:
        cap = cv2.VideoCapture('data/mini_example.avi')
        
    # Initialize tracker
    tracker = GridFlowTracker(imgw, imgh, GRID_SIZE)
    
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