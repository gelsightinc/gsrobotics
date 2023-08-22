# SDK for GelSight robotic sensors


This respository contains python code to connect to the GelSight Mini. The code enables basic functionalities, such as
view and save data (images or video) from these devices, and example code to get 3D point cloud data derived from 2D images.

## Prerequisites

    Python 3.8 or above

## Install python libraries
    pip3 install .
    or 
    pip3 install . --upgrade

Note this step does not install the [ROS](http://wiki.ros.org/ROS/Installation) or [ROS2](https://docs.ros.org/en/foxy/index.html) related 
libraries required by the python scripts in examples/ros folder of this repository yet. Please follow the installation guide of those seperately. 


## Set paths 

    PYDIR=`pip3 show gelsight | grep -i location | cut -f2 -d" "`
    export PYTHONPATH=$PYDIR/gelsight:$PYTHONPATH


# GelSight Mini


## Windows setup

The script tries to find the correct camera device id on Windows.
You may need to change the following line in show3d.py and showimages.py

    dev = gsdevice.Camera("GelSight Mini")


## View raw 2D image
    cd examples
    python3 showimages.py


## View 3d point cloud data
    cd examples
    python3 show3d.py

To obtain the expected results from the algorithms implemented on the raw image, please set the default camera parameters present in mini_set_cam_params.sh.

## Mini Marker Tracking

There are multiple marker tracking demos.  Each uses a different marker tracking algorithm.
You can find all of them in the demo directory.

* marker_tracking:  contains demos using a mean shift algorithm and an optical flow algorithm

      cd demos/marker_tracking
      python3 mean_shift_marker_tracking.py
              or
      python3 optical_flow_marker_tracking.py

* mini_tracking_linux_V0: contains demo using compiled code for depth first search to run on Linux, to run:
  
      cd demos/mini_tracking_linux_V0
      python3 tracking.py

* mini_tracking_windows_V0: contains demo using compiled code for depth first search to run on Windows, to run:

      cd demos/mini_tracking_windows_V0
      python3.exe tracking.py


# Mini ROS and ROS2 

## Prerequisites

Install [ROS](http://wiki.ros.org/ROS/Installation) or [ROS2](https://docs.ros.org/en/foxy/index.html) , see [ROS Documentation](https://docs.ros.org/) for instructions.
The example code uses cv_bridge which can be installed using:
    
    sudo apt-get install ros-${ROS_DISTRO}-cv-bridge
    
For example, on Ubuntu 20, 

    To install cv-bridge for ROS
    
        sudo apt-get install ros-noetic-cv-bridge
        
    To install cv-bridge for ROS2
    
        sudo apt-get install ros-foxy-cv-bridge

The showimages examples publish to the ROS topic /gsmini_rawimg_0

The show3d examples publish to the ROS topic /pcd

They can be viewed in rviz or rviz2

## ROS examples
    source /opt/ros/noetic/setup.bash
    cd examples/ros
    roscore
    python3 showimages_ros.py
    python3 show3d_ros.py
    rviz -d mini_ros_3d_config.rviz

## ROS2 examples
    source /opt/ros/foxy/setup.bash
    cd examples/ros
    python3 showimages_ros2.py
    python3 show3d_ros2.py
    rviz2 -d mini_ros2_3d_config.rviz
    
## Device setup

The camera on the GelSight Mini is a USB camera. 

If you need to adjust the camera settings for your application, you can change the camera parameters using any app or library that can control UVC cameras. 

## Linux camera settings

A popular library is [v4l2-ctl](https://manpages.ubuntu.com/manpages/bionic/man1/v4l2-ctl.1.html).
To install this library on ubuntu run, 

    sudo apt-get update
    sudo apt-get -y install v4l-utils

Refer to file config/mini_set_cam_params.sh present in this repository to view/edit all the available camera parameters. 
You can list the devices by running:

    v4l2-ctl --list-devices

In most cases when you have one Mini connected to your computer, the device ID is usually 2, because the webcam on your computer is always on device ID 0.

## Windows camera settings

On Windows, you can use the AMCap app to configure the camera settings, see https://docs.arducam.com/UVC-Camera/Quick-Start-for-Different-Systems/Windows/


## Possible errors 
### With cv2
    pip uninstall opencv-python-headless
  
### With Docker container
    sudo apt-get install libopenjp2-7
    sudo apt-get install qt5-default
    pip3 install opencv-python==4.1.2.30
    sudo apt-get install libopenexr-dev


# Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


# Contact
This package is under active development. Contact debra_shure@gelsight.com if have any questions / comments / suggestions.


