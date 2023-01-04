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
python tracking.py
```

### Shear force test
* **Pink arrow**: the marker is out of boundary/not detected, but can be inferred from surrounding flow (assume same first derivative)

![](results/shear_flow.gif)

### Twist force test
![](results/twist_flow.gif)


## Configuration

Configuration based on different marker settings (marker number/color/size/interval)


### Step 1: Marker detection

The marker detection is in	`marker_detection.py`

Modify the code based on the marker color & size.

To verify, run

```
python tracking.py calibrate
```

And the mask should looks like:

![](results/mask.gif)


**Set Parameters**:

* `find_marker`: The kernel size in GaussianBlur, it depends on marker size. should could cover the whole marker.
* `find_marker`: change `yellowMin, yellowMax` based on markers' color in HSV space.
* `marker_center`: change the `areaThresh1, areaThresh2` for the minimal and maximal size of markers



### Step 2: Marker matching

**Set Parameters**

`setting.py`

* RESCALE: scale down
* N, M: the row and column of the marker array
* fps_: the desired frame per second, the algorithm will find the optimal solution in 1/fps seconds


## Output

The Matching Class has a function `get_flow`. It returns the flow information:

```
flow = m.get_flow()

output: (Ox, Oy, Cx, Cy, Occupied) = flow
    Ox, Oy: N*M matrix, the x and y coordinate of each marker at frame 0
    Cx, Cy: N*M matrix, the x and y coordinate of each marker at current frame
    Occupied: N*M matrix, the index of the marker at each position, -1 means inferred. 
        e.g. Occupied[i][j] = k, meaning the marker mc[k] lies in row i, column j.
```

