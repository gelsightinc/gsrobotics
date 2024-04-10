# SDK for GelSight robotic sensors


This respository contains python code to connect to the GelSight Mini. The code enables basic functionalities, such as view and save data (images or video) from these devices, and example code to get 3D point cloud data derived from 2D images.

About to update instructions

## Prerequisites

### Windows

#### Cloning the repository

On Windows, we recommend running the examples within the PyCharm development environment. Here are instructions on how to set up PyCharm. 

1. Download and install **[Git for Windows](https://gitforwindows.org/)**
1. Download and install **[TortoiseGit](https://tortoisegit.org/)**. For most users, the correct version is 64-bit Windows. Run the First Start wizard after installation and choose the default options.
1. Download and install **[Python 3.10](https://www.python.org/downloads/release/python-31011/)**. More recent versions of Python might require additional steps to install the packages used by this codebase. 
    For most users, the correct version is **Windows installer (64-bit)**
1. Go to **[PyCharm](https://www.jetbrains.com/pycharm/download/?section=windows)** and download the PyCharm Community installer

1. Clone this repository
    1. Navigate to a folder on your machine where you would like to clone the repository
    2. On this page, click the green **Code** button at the top and copy the repository URL by clicking the button to the right of the URL.
    3. In Windows Explorer, right-click in the folder and choose Git Clone...  On Windows 11, you will have to choose "Show More Options"
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
1. The first time you open the gsrobotics folder, PyCharm will prompt you to create the virtual environment. This will automatically install the packages listed in the requirements.txt file. Click **OK** to create the environment.
          <table>
           <tr><td>
       <img src="/docs/images/create-virtual-environment.png" alt="TortoiseGit settings" height="200px">
           </td> 
           </tr>
           <tr><td>Create the virutal environment in PyCharm</td>
        </tr>
       </table>
1. From the file navigator on the left, open the script **showimages.py** in the examples folder. Right-click in the file editor and choose **Run 'showimages'**.
          <table>
           <tr><td>
       <img src="/docs/images/run-showimages.png" alt="TortoiseGit settings" height="400px">
           </td> 
           </tr>
           <tr><td>Run showimages.py in PyCharm</td>
        </tr>
       </table>
1. You should see a window with a live view from the camera in the GelSight Mini. Press the Mini into an object to see a live tactile image.
1. Congratulations, you are now able to program your GelSight Mini!

#### Live 3D Demo
1. These instructions assume you have successfully installed Python, PyCharm and run the **showimages.py** demo above.
1. Open the **show3d.py** script
1. Right-click in the file editor and choose **Run 'show3d'**. 


https://github.com/gelsightinc/gsrobotics/assets/44114954/85b3e123-730d-4dfa-a05a-983dfc1e5a78



1. Congratulations, you are now able to run the 3D demo using GelSight Mini!


### Linux

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


# More Examples



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
This package is under active development. Contact support@gelsight.com if have any questions / comments / suggestions.


