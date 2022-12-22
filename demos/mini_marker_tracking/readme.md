# Marker Tracking Algorithm

## Requirement

* opencv
* pybind11
* numpy


For Ubuntu:

```
sudo apt-get install python-opencv
pip install pybind11 numpy
```

For Mac:

```
brew install opencv
pip install pybind11 numpy
```

## Example

```
make
python src/tracking.py
```

### Shear force test
* **Pink arrow**: the marker is out of boundary/not detected, but can be inferred from surrounding flow (assume same first derivative)

![](results/shear_flow.gif)

### Twist force test
![](results/twist_flow.gif)


## Configuration

Configuration based on different marker settings (marker number/color/size/interval)


### Step 1: Marker detection

The marker detection is in	`src/marker_detection.py`

Modify the code based on the marker color & size.

To verify, run

```
python src/tracking.py calibrate
```

And the mask should looks like:

![](results/mask.gif)


**Set Parameters**:

* `src/marker_detection/find_marker`: The kernel size in GaussianBlur, it depends on marker size. should could cover the whole marker.
* `src/marker_detection/find_marker`: change `yellowMin, yellowMax` based on markers' color in HSV space.
* `src/marker_detection/marker_center`: change the `areaThresh1, areaThresh2` for the minimal and maximal size of markers



### Step 2: Marker matching

**Set Parameters**

`src/setting.py`

* RESCALE: scale down
* N, M: the row and column of the marker array
* x0, y0: the coordinate of upper-left marker (in original size)
* dx, dy: the horizontal and vertical interval between adjacent markers (in original size)
* fps_: the desired frame per second, the algorithm will find the optimal solution in 1/fps seconds


## Output

The Matching Class has a function `get_flow`. It return the flow information:

```
flow = m.get_flow()

output: (Ox, Oy, Cx, Cy, Occupied) = flow
    Ox, Oy: N*M matrix, the x and y coordinate of each marker at frame 0
    Cx, Cy: N*M matrix, the x and y coordinate of each marker at current frame
    Occupied: N*M matrix, the index of the marker at each position, -1 means inferred. 
        e.g. Occupied[i][j] = k, meaning the marker mc[k] lies in row i, column j.
```

