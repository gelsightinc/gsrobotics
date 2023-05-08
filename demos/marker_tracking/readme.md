# Marker Tracking Demos


There are 2 demos available using 2 different algorithms, optical flow and mean shift.

The demos can be run using the sample input video, mini_example.avi, in the data directory. 

To run the demos using data directly from the Gelsight MINI live set: 

    USE_MINI_LIVE = True
    
Both demos require at least one frame of data when the gel is not touching anything in order to find the initial marker positions.
You can check the marker calibration of the sample video by running markertracker.py


**Options**:

```
    SAVE_VIDEO_FLAG = True, save the video with markers overlayed 
    USE_MINI_LIVE = True, use the live stream from the Mini
```

### Optical Flow
* **optical_flow_marker_tracking.py**: Calculates the optical flow using the Lucas-Kanade method.

### Mean Shift
* **mean_shift_marker_tracking.py**:  Track the markers using an iterative mean-shift algorithm to shift the markers toward the highest density regions

   **Set Parameters**:

* templatefactor:  Default value = 1.5 In track_markers, this number gets multiplied by marker radius to define the region to mean-shift


   **Optional Parameters/Outputs**:

```
  SAVE_MARKER_FLAG = True, output a .csv file
  SHOW_3D = True, compute and display 3D data
  If SHOW_3D = True:
      GPU = True, use GPU to compute 3D data
      SAVE_3D_TMD = True, save each frame of 3D data as a .tmd file
      SAVE_3D_PCD = True, save each frame of 3D data as a .pcd file 
```

.csv file format

    Frame number, Row, Column, Ox, Oy, Oz, Cx, Cy, Cz

**Oz and Cz are only output if SHOW_3D is set to True**

```
    Ox, Oy, Oz: the x,y and z coordinate of each marker at frame 0
    Cx, Cy, Cz: the x,y and z coordinate of each marker at current frame
```



