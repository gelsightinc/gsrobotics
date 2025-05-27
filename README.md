# Gelsight Mini

This respository contains python code to connect to the GelSight Mini hardware. The code demonstrates basic functionalities, such as viewing and saving data (e.g., images and video) from these devices, depth estimation, 3D point cloud and marker tracking. Any further information about Gelsight can be found on [gelsight.com](https://www.gelsight.com/)

On this page, you will find
 - [Demo Scripts](#demo-scripts) for
   + [Live View](#liveview-demo)
   + [Marker tracking](#marker-tracker-demo)
   + [3D Viewer](#3d-viewer-demo)
   + [Basic OpenCV Demo](#basic-opencv-demo)
 - Detailed [Installation Instructions](#installation-instructions) for
   + [Windows](#windows-installation-instructions)
   + [Linux](#linux-installation-instructions)
 - [CAD Models](#cad-models) for
   + [GelSight Mini Case](#gelsight-mini-case)
   + [Schunk Gripper Adapter](#schunk-adapter)
   + [Panda Gripper Adapter](#panda-adapter)
   + [Kuka Adapter](#kuka-adapter)
  - [Frequently Asked Questions](#frequently-asked-questions)

# Demo Scripts

## LiveView Demo

1. Run `demo_liveview.py` in PyCharm. You should see a window app for live view session. On top bar you have device selection option. This is especially helpful if you have web cameras, or more than one GelSight Mini. Once a device is choosen, live feed should start.
<table>
<tr><td>
<img src="/docs/images/live_view_device_choice.png" alt="Demo Live View Device choice" height="400px">
</td>
</tr>
<tr><td>Select device in Live view app.</td>
</tr>
</table>

1. Once live feed starts you should be able to zoom in/out the view.
<table>
<tr><td>
<img src="/docs/images/live_view_zoom.png" alt="Demo Live View zoom view" height="400px">
</td>
</tr>
<tr><td>Zoom in/out the live view</td>
</tr>
</table>

1. Select **Data Folder** button to choose location of your captured screenshots/videos.
The default location is the Desktop.
<table>
<tr><td>
<img src="/docs/images/live_view_location_path.png" alt="Demo Live View path choice" height="400px">
</td>
</tr>
<tr><td>Choose location for saving images and videos.</td>
</tr>
</table>

1. Select **Save Image** button (or press **SPACEBAR**) to take screenshot of the current feed.
   File will be saved at **Data Folder** location as .png file.

1. Select **Start Recording** button to start feed capture. Then pres **Stop Recording** to save
   captured feed at **Data Folder** location as .mp4 file.

## Marker Tracker Demo

1. Run `demo_markertracker.py` in PyCharm. You should se a window app for marker tracking session. On top bar you have device selection option.
Once a device is choosen, live feed should start.
<table>
<tr><td>
<img src="/docs/images/marker_tracker_still.png" alt="Markertracker app sensor still" height="400px">
</td>
</tr>
<tr><td>Markertracker app with still surface</td>
</tr>
</table>

1. When applying force to the device surface markertracking app should detect shift in marker positions and draw movement vectors as an overlay.
<table>
<tr><td>
<img src="/docs/images/marker_tracker_pressed.png" alt="Markertracker app sensor pressed" height="400px">
</td>
</tr>
<tr><td>Markertracker app with pressed surface</td>
</tr>
</table>

1. You can record movement of the markers . By default data is saved in both files: .npy and .csv.

   Select **Data Folder** button to choose location of your captured markers movement. The default location one is desktop.

1. Select **Start Recording** to start recording of markers movement.

1. Select **Save Data** to store current sequence. Each row of data (except for the first description row)
   forms a sequence x, y pairs (position x, position y) for all markers.

1. Select **Reset Tracking** to reset current tracking (sometimes reset is needed when to large force is applied).

## 3D Viewer Demo

1. Run `demo_view3D.py`in PyCharm. You should se a window app for live view session. By default you should two windows. One should consist
   of three horizontally stacked views: camera feed, camera mask and depth estimation. The second window should show 3D pointcloud the 3D pointcloud.
   The camera ID is taken from the config settings.

<table>
<tr><td>
<img src="/docs/images/3d_still.png" alt="3D view sensor still" height="600px">
</td>
</tr>
<tr><td>3D view app sensor still</td>
</tr>
</table>

<table>
<tr><td>
<img src="/docs/images/3d_pressed.png" alt="3D view sensor still" height="600px">
</td>
</tr>
<tr><td>3D view app sensor pressed</td>
</tr>
</table>

1. You can rotate the point cloud by clicking with the left mouse button and dragging.  Zoom in and out with mouse scroll wheel.
2. To quit the app press 'q' on the keyboard. (Closing with 'x' button doesnt always work. Multiple presses sometimes are needed)

## Basic OpenCV Demo

This demo shows how to grab images from the GelSight Mini and display them using
OpenCV functions. 

1. Run `opencv_liveview-demo.py` in PyCharm. 

<table>
<tr><td>
<img src="/docs/images/opencv_liveview.png" alt="GelSight Mini OpenCV Liveview Demo" height="600px">
</td>
</tr>
<tr><td>Live view from GelSight Mini using OpenCV functions</td>
</tr>
</table>


## GS Config

Each of demo files uses **ConfigModel** class from config.py file that can be found in the root folder.
By default each demo script uses **default_config** from the `config.py` file. If you want to edit them, on option would be to adjust **default_config** variable in `config.py`.
Optionally you can provide path to valid .json config in the demo scripts.
An example `default_config.json` can be found in the root folder.

```json
{
  "default_camera_index": 0,
  "camera_width": 320,
  "camera_height": 240,
  "border_fraction": 0.15,
  "marker_mask_min": 0,
  "marker_mask_max": 70,
  "pointcloud_enabled": true,
  "pointcloud_window_scale": 3.0,
  "cv_image_stack_scale": 1.5,
  "nn_model_path": "./models/nnmini.pt",
  "cmap_txt_path": "./cmap.txt",
  "cmap_in_BGR_format": true,
  "use_gpu": false
}
```

### Description:

**default_camera_index**: `int` -> Used by demo_3d.py to determine which device should be used as an active camera

**camera_width**: `int` -> Default width of camera feed. Image width will be resized to target width.

**camera_height**: `int` -> Default height of camera feed. Image width will be resized to target height.

**border_fraction**: `float` -> Amount of crop around image edges. 0.15 means that 15% from top, bottom, left and right crop will be applied.

**marker_mask_min**: `int` -> Grayscale value (0-255) determining lower bound for masked out region from the image (marker spots removal)

**marker_mask_max**: `int` -> Grayscale value (0-255) determining upper bound for masked out region from the image (marker spots removal)

**pointcloud_enabled**:`bool` -> When enabled, 3D pointcloud window will be drawn in demo_3D.py script.

**pointcloud_window_scale**: `float` -> Determines the window size of 3D pointcloud in demo_3d.py script.

**cv_image_stack_scale**: `float` -> Determines the window size of horizontally stacked images in demo_3d.py script.

**nn_model_path**: `str` -> Path to normals estimation deep neural network model.

**cmap_txt_path**: `str` -> Path to colormap scheme useid int demo_3d.py script.

**cmap_in_BGR_format**: `bool` -> Determines color format of the cmap used in demo_3d.py script.

**use_gpu**: `bool` -> When enabled, GPU will be used (if available) for neural network model inference.

## Running scripts from terminal

You can run python scripts directly from the PyCharm terminal. Go to the bottom left corner and click `Terminal` button (Alt + F12)

<table>
<tr><td>
<img src="/docs/images/pycharm_terminal.png" alt="PyCharm terminal" style="height:300px; width:auto;">
</td>
</tr>
<tr><td>PyCharm terminal</td>
</tr>
</table>

PyCharm by default creates local envirnoment where it downloads all required packages. When running scripts using PyCharm run, this
environment is automatically used. In terminal you need to be sure that the local enviroment is activated. To activate it (assuming that
folder name is `.venv` and its placed at the root) you need to
type

#### Windows

```sh
# If using CMD
.\.venv\Scripts\activate

# If using PowerShell
.\.venv\Scripts\Activate.ps1
```

#### Linux

```sh
source .venv/bin/activate
```

Once the environment is activated you should see its name in the paranthesees. In this case it should look like:

<table>
<tr><td>
<img src="/docs/images/pycharm_venv_active.png" alt="PyCharm venv activated" style="height:200px; width:auto;">
</td>
</tr>
<tr><td>PyCharm .venv activated</td>
</tr>
</table>

For example, to run `demo_liveview.py` (assuming you are in the root folder) you need to type in terminal:

#### Windows

```sh
python demo_liveview.py
```

#### Linux

```sh
python demo_liveview.py
```

You can pass as an optional argument a path to the config file. For example you could use `default_config.js` that
is in the root folder:

#### Windows

```sh
python demo_liveview.py --gs-config default_config.json

```

#### Linux

```sh
python demo_liveview.py --gs-config default_config.json
```




# Installation Instructions

This section describes how to set up a python development environment on your
system for running the code examples in this repository. If you already have a
working python environment, clone the repository and run the [Demo
Scripts](#demo-scripts) described above.

## Windows Installation Instructions

#### Cloning the repository

On Windows, we recommend running the examples within the PyCharm development environment, however you are free to choose your environment and IDE. Here are instructions on how to set up PyCharm.

1. Download and install **[Git for Windows](https://git-scm.com/downloads/win)**
2. Download and install **[TortoiseGit](https://tortoisegit.org/)**. For most users, the correct version is **Windows installer (64-bit)**. Run the First Start wizard after installation and choose the default options.
3. Download and install **[Python 3.12](https://www.python.org/downloads/release/python-3129/)**. More recent versions of Python might require additional steps to install the packages used by this codebase. For most users, the correct version is **Windows installer (64-bit)**

4. Go to **[PyCharm](https://www.jetbrains.com/pycharm/download/?section=windows)** and download the **PyCharm Community** installer

5. Clone this repository
   1. Navigate to a folder on your machine where you would like to clone the repository
   2. On this repository page, click the green **Code** button at the top and copy the repository URL by clicking the button to the right of the URL.
   3. In Windows Explorer, right-click in the folder and choose Git Clone... On Windows 11, you will have to choose "Show More Options"
      <table>
          <tr><td>
      <img src="/docs/images/git-clone.png" alt="TortoiseGit menu" height="400px">
          </td> 
          <td>
              <img src="/docs/images/show-more-options.png" alt="Windows 11 contextual menu" height="400px">
          </td></tr>
          <tr><td>Git Clone using TortoiseGit</td>
          <td>Need to click Show More Options on Windows 11</td></tr>
      </table>
   4. Clone the repository to a folder on your local machine
      <table>
          <tr><td>
      <img src="/docs/images/git-clone-paths.png" alt="TortoiseGit settings" height="400px">
          </td> 
          </tr>
          <tr><td>Clone repository using TortoiseGit</td>
       </tr>
      </table>

#### Configuring PyCharm

1. Plug in the GelSight Mini device to a USB port on your computer. The lights should turn on.
1. Launch PyCharm and Open the gsrobotics folder you cloned following the instructions above. Choose **Trust Project** when prompted.
1. The first time you open the gsrobotics folder, PyCharm will prompt you to create the virtual environment. This will automatically install the packages listed in the `requirements.txt` file. Click **OK** to create the environment.
<table>
<tr><td>
<img src="/docs/images/create-virtual-environment.png" alt="Pycharm virtualenv" height="200px">
</td>
</tr>
<tr><td>Create the virutal environment in PyCharm</td>
</tr>
</table>
1. From the file navigator on the left, open the script `demo_liveview.py` in the root folder. Right-click in the file editor and choose **Run 'demo_liveview'**.
<table>
<tr><td>
<img src="/docs/images/run_demo_liveview.png" alt="Run demo liveview" height="400px">
</td>
</tr>
<tr><td>Run demo_liveview.py in PyCharm</td>
</tr>
</table>

## Linux Installation Instructions

### Cloning the repository

1. Download and install **[Git for Linux](https://git-scm.com/downloads/linux)**
2. Download and install **[Python 3.12](https://www.python.org/downloads/release/python-3129/)**. More recent versions of Python might require additional steps to install the packages used by this codebase.

3. Go to **[PyCharm](https://www.jetbrains.com/help/pycharm/installation-guide.html#toolbox)** instalation guide and choose prefered way of intalation of **PyCharm Community** on linux.

4. Clone this repository
   1. Navigate to a folder on your machine where you would like to clone the repository
   2. On this repository page, click the green **Code** button at the top and copy the repository URL by clicking the button to the right of the URL.
   3. In Linux desired location, right-click in the folder and choose **open terminal**.
   4. In the terminal type
   ```sh
   git clone repository_url
   ```

### Configuring PyCharm

On Linux configuration process is the same as on Windows. In PyCharm open new project and select destination of your downloaded repository.
In the same way as on Widnows, PyCharm will automatically ask you to create local environment and use `requirements.txt` as a source list of packages
to install.

## Device setup

The camera on the GelSight Mini is a USB camera.

If you need to adjust the camera settings for your application, you can change the camera parameters using any app or library that can control UVC cameras.

### Linux camera settings

A popular library is [v4l2-ctl](https://manpages.ubuntu.com/manpages/bionic/man1/v4l2-ctl.1.html).
To install this library on ubuntu run,

    sudo apt-get update
    sudo apt-get -y install v4l-utils

Refer to file config/mini_set_cam_params.sh present in this repository to view/edit all the available camera parameters.
You can list the devices by running:

    v4l2-ctl --list-devices

In most cases when you have one Mini connected to your computer, the device ID is usually 2, because the webcam on your computer is always on device ID 0.

### Windows camera settings

On Windows, you can use the AMCap app to configure the camera settings, see https://docs.arducam.com/UVC-Camera/Quick-Start-for-Different-Systems/Windows/

## Possible errors

### With cv2

    pip uninstall opencv-python-headless

### With Docker container

    sudo apt-get install libopenjp2-7
    sudo apt-get install qt5-default
    pip3 install opencv-python==4.1.2.30
    sudo apt-get install libopenexr-dev

# CAD Models
The following CAD models can help you mount a GelSight Mini on different grippers or create your own adapter. 

## GelSight Mini Case
The GelSight Mini Case model can be used to test designs of custom adapters and fixtures:
[Download GelSight Mini Case STP File](docs/CAD/gsmini_shell.zip)

## Schunk Adapter
This adapter is used to attach a GelSight Mini to a Schunk parallel jaw gripper:
[Download Schunk Gripper Adapter STP File](docs/CAD/schunk_adapter_gsmini.zip)
  <table>
          <tr>
          <td>
              <img src="/docs/images/schunk_adapter.jpg" alt="Schunk Gripper Adapter" height="400px">
          </td>
          </tr>
          <tr>
           <td>Schunk Gripper Adapter</td>
          </tr>
      </table>

## Panda Adapter
This adapter is used to attach a GelSight Mini to the parallel jaw gripper on a Franka Emica Panda robot:
[Download Panda Gripper Adapter STP File](docs/CAD/panda_adapter_gsmini.zip)
  <table>
          <tr>
          <td>
              <img src="/docs/images/panda_adapter.jpg" alt="Panda Gripper Adapter" height="400px">
          </td>
          </tr>
          <tr>
           <td>Panda Gripper Adapter</td>
          </tr>
      </table>

## Kuka Adapter
This adapter is used to attach a GelSight Mini to a Kuka robot:
[Download Kuka Adapter STP File](docs/CAD/kuka_adapter_gsmini.zip)
  <table>
          <tr>
          <td>
              <img src="/docs/images/kuka_adapter.jpg" alt="Kuka Adapter" height="400px">
          </td>
          </tr>
          <tr>
           <td>Kuka Adapter</td>
          </tr>
      </table>


# Frequently Asked Questions


#### 1. What is the mm to pixel conversion value?
   
When using the 3D model and 240x320 down sampled image, the resolution is 0.0634 mm/pixel.
 
#### 2. How was the 0.0632 mm/pixel conversion obtained? Should the height and width mm to pixel values be different?
   
The mm/pixel scale factor was obtained and calculated by scanning an object with known size (a ruler) and measuring the image distance in pixels.
The height and width values are the same because all the image processing algorithms work in pixel units to keep the axes at the same scale.
 
#### 3. What is the difference between the GelSight Mini and GelSight R1.5?
   
The GelSight R1.5 is an older research system for robotic tactile sensing that is not currently available for sale.
 
#### 4. Can we obtain a tactile map of the grasped objects with the sensor? In which format would this information be provided? What is the scale and resolution?
   
Yes, in the form of 3D point cloud data derived from 2D images. GS Mini has an 8MP camera. When using the 3D model and the 240x320 down sampled image, the resolution is 0.0634 mm/pixel.

#### 5. Can we obtain the pressure force applied in the sensors? In which format would this information be provided? What is the scale and resolution?
   
Currently height displacement can be obtained which can be used to train a model that outputs the pressure force applied.
 
#### 6. Can we get information about the texture of the objects in contact with the sensor? In which format would this information be provided?
   
It is possible to derive texture information from statistical analyses of the height map.
 
#### 7. How can we access the sensor’s readings in real-time? Do we need software for this, or can we use an SDK in a programming language (e.g., Python, C++, Java) with an API that allows us to access the sensor readings by code?
   
There are programming examples in Python which allow real-time images and processing of data.
 
#### 8. Are the sensors compatible with Windows and Ubuntu? Is there any ROS node available for interfacing with the sensor?
    
Programming is available in Python which is cross platform.
 
#### 9. Does the sensor require an additional power supply cable, or does it only use the USB-3 cable?
   
Power is supplied via the USB cable.
 
#### 10. What is the frequency of sensor reading?

The frame rate is 25 FPS.
 
#### 11. Does the sensor need a processing unit, or can it be connected directly to a computer?
    
It can be connected directly to a computer.
 
#### 12. What would be the price of a pair of sensors? Is there any special price if we buy a certain number of sensors? Is there any discount for universities and academic institutions?
    
A pair of Mini Systems would be $ USD 1,008.00, plus shipping fees and any applicable import fees. If you opted for the Mini Robotics Package, which is the Mini System plus a tracking marker gel, a pair would be $1,108.00 USD plus shipping and import fees. Unfortunately, there is no discount offered on our Mini products at this time.
 
#### 13. If we buy sensors, will they be delivered now or is there a waiting list? How long would it take to receive the sensors?

The standard lead time for Mini products is about 4 weeks, but we can often ship sooner. Transit time once shipped is usually 4-6 business days, subject to any customs delays.
 
#### 14. Is calibration typically performed for each sensor prior to purchase, or is it something the buyer needs to undertake?

The example code repository includes a single calibrated model that the user can reference after the sensor has been purchased.
 
#### 15. Does the GS Mini output consist solely of RGB camera data, or are depth maps and possible point cloud information provided directly through specific packages? Or would these need to be developed separately?

There are programming examples available in Python which allow real-time images and processing in the form of 3D point cloud data derived from 2D images.
 
#### 16. In addition to Python, is there a C++ version of the GS Mini demo that is available?
    
Only Python is supported by the GS Mini demo at this time.
 
#### 17. Have there been any studies done on the GS Mini reconstruction accuracy?
    
GS Mini is not a metrology device so there are no study results that can be shared regarding the accuracy of the 3D reconstruction.
 
#### 18. What is the distance between the camera lens and the top of the gel pad?
    
The lens moves up and down depending on the focus. Below is a diagram showing the location of the camera with respect to the housing.

#### 19. Is there any html/css/javascript code available to integrate the GS Mini web app?

The source code for the web app is not currently supported but it can be referenced by choosing view source from a browser.
 
#### 20. What is the durometer of the Mini gel?
    
The durometer of the Mini gel is 55 on the Shore 00 scale. The same applies for the Mini gel with trackers.
 
#### 21. What is the optimum force that should be applied to the Mini for image capture?
    
The optimal force depends on what is being measured. For example, it takes 80N to reach the bottom of a 0.5mm deep groove and less force to measure shallower features.
 
#### 22. What type of semi-specular coating is used for the Mini gel? Is it like the original type with aluminum flakes, or no?
    
It is a blend of black and white pigments, not metal flakes.
 
#### 23. What is the difference between Mini and other GelSight Products like Max and Mobile?
    
GelSight Mobile and Max systems are shop-floor handheld metrology systems. As such, they have quantified measurement performance in XY and Z as well as for different applications such as hole diameter measurements and roughness.  
 GelSight Mini was designed for robotic manipulation applications, so we do not have system accuracy information available, but you are welcome to perform your own studies
 
#### 24. What is the difference between GelSight Mini and DIGIT?
    
The Mini provides a better-quality image with more even illumination.  It also comes with the ability to generate 3D data, and marker gels can be used with Mini.  It is a tactile sensor designed for robotic and research applications.  Mini’s lens is sufficient to see a human fingerprint at up to 5-line pairs per millimeter or 0.1 mm.
DIGIT provides lower quality images.  It is designed for robotic in-hand manipulation and machine learning. The sensor is not designed for quantitative measurements.
DIGIT has 60Hz frame rate and Mini has 24Hz frame rate. 

For more info on DIGIT, please visit: [digit.ml](https://digit.ml/)
 
 
#### 25. In what instances/applications is it preferred to use tracker gel?
    
The tracker gel is useful for tasks that involve shear forces on the gel, such as insertion tasks. You can see a demo application using the marker gels here:  https://www.youtube.com/watch?v=vTa8u8-XOEU
 
#### 26. What is an estimated durability of Mini Gel – use cases like skin, water, heat?
    
The durability of the gel is directly related to the way it is used.  If used on a smooth surface, the gel can withstand 1000s of scans.  However, grasping rough surfaces can shear the gel, and pressing the gel into items with sharp edges can wear away the gel coating, and cut or tear the gel. 
 
#### 27. What is the preferred method of cleaning Mini Gel cartridge?
    
The standard method is to brush off debris with a foam swab. If the cartridge is dirty, it can be cleaned with a foam swab saturated with a small amount of IPA.
 
#### 28. How do we get the serial number of a Mini sensor?
    
There are python code examples in our github repository that demonstrate how to read the serial number from the camera description string:
```
     cam_desc = gsdevice.get_camera_desc("GelSight Mini")
    # Get serial from description
    match = re.search("[A-Z0-9]{4}-[A-Z0-9]{4}", cam_desc)
    devserial = "Unknown"
    if match:
        devserial = match.group()
        print("found mini: ", devserial)
 ```

#### 29. Here is a short version of GelSight Mini warranty?
    
1 Year limited warranty, not including gel.
 
#### 30. Here are some useful YouTube links for general usage of GelSight mini sensors.
    - [Mini introductory video](https://youtu.be/HIFA83COlcc)
    - [Sandpaper classification](https://youtu.be/EhvuZaydEW4)
    - [Hardness Demo](https://youtu.be/HnmVz8bAiyA?t=10s)
    - [Liquid identification](https://youtu.be/vTa8u8-XOEU)

# Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

# Contact

This package is under active development. Contact support@gelsight.com if have any questions / comments / suggestions.

```

```
