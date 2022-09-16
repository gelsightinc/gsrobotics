# SDK for GelSight robotic sensors


This respository contains python code, to connect to Gelsight robotic sensors that include, Mini, R1.5 and R1. The code enables basic functionalities, like 
view and save data (images or video) from these devices, and an example code to get 3D point cloud data derived from 2D images.

## Prerequisites
    Python 3.8
    Ubuntu 20.04

## Install python libraries
    pip3 install .
    or 
    pip3 install . --upgrade

Note this step does not install the [ROS](http://wiki.ros.org/ROS/Installation) or [ROS2](https://docs.ros.org/en/foxy/index.html) related 
libraries required by the python scripts in examples/ros folder of this repository yet. Please follow the installation guide of those seperately. 


## Set paths 
or run setup.sh script to set the following

    PYDIR=`pip3 show gelsight | grep -i location | cut -f2 -d" "`

    export LD_LIBRARY_PATH=$PYDIR/gelsightcore:$LD_LIBRARY_PATH

    export PYTHONPATH=$PYDIR/gelsightcore:$PYDIR/gelsight:$PYTHONPATH


# Mini

The instruction below pertain to GelSight Mini device. For R1.5 device, keep scrolling down.

## Device setup

The camera on Mini is a USB camera. You can change the camera parameters using any app or library that can control UVC cameras. On Ubuntu, one such popular library is [v4l2-ctl](https://manpages.ubuntu.com/manpages/bionic/man1/v4l2-ctl.1.html).
To install this library on ubuntu run, 

    sudo apt-get update
    sudo apt-get -y install v4l-utils

Refer to file config/mini_set_cam_params.sh present in this repository to view/edit all the available camera parameters. 
You can list the devices by running:

```v4l2-ctl --list-devices```


To set the camera parameters listed in mini_set_cam_params.sh file, run, 

```sudo ./config/mini_set_cam_params-ctl.sh 2```

Note the scripts takes the camera device id (0, or 1, or 2, or 3,.. etc), as the first argument. In most cases when you have one Mini connected to 
you computers, the device ID is usually 2, because the webcam on your computer is always on device ID 0.


## View raw 2D image
    cd examples
    python3 showimages.py -d mini


## View 3d point cloud data
    cd examples
    python3 show3d.py -d mini

To obtain the expected results from the algorithms implemented on the raw image, please set the default camera parameters present in mini_set_cam_params.sh.  

# Mini ROS and ROS2 

## Prerequisites

ROS or ROS2 should be intalled, see [ROS Documentation](https://docs.ros.org/) for instructions.
The example code uses cv_bridge which can be installed using:
    
    sudo apt-get install ros-${ROS_DISTRO}-cv-bridge

The showimages examples publish to the ROS topic /gsmini_rawimg_0

The show3d examples publish to the ROS topic /pcd

They can be viewed in rviz or rviz2

## ROS examples
    cd examples/ros
    python3 showimages_ros.py
    python3 show3d_ros.py -d mini

## ROS2 examples
    cd examples/ros
    python3 showimages_ros2.py
    python3 show3d_ros2.py -d mini
    


# R1.5

The instruction below pertain to GelSight R1.5 device. For R1 device, keep scrolling down.  

## Device setup
The R1.5 camera is read through a Raspberry Pi. Your Pi you received has been configured with the following settings:

Username: pi \
Password: gelsight
    
The address name of your specific Pi is shown on small white sticker, right above the ethernet port on the Pi. e.g.  gsr150xx.local

Add this address to your wired network: (for Ubuntu users)

1. Go to, Network Settings -> Wired connection
2 Add a new ethernet network by clicking the plus sign on the top right corner
3. In the field Name, put gsr150XX (name on the sticker without the .local)
4. In the same window, go to, IPv4 Setting -> Method -> select Link Local only
5. Save and exit

If you want to change the default address of Pi and set it to a static IP address, follow the instructions [here](https://raspberrypi-guide.github.io/networking/set-up-static-ip-address)


## Connect and see live data

Open a browser and go to link:

    http://gsr150XX.local:8080/?action=stream 

DO NOT forget to replace xx below with the actual ID number of the Raspberry Pi from here onwards.


R1.5 runs a service by default that streams video at 60 fps at a resolution of 240x320, once the device is turned on or restarted. Refresh the above webpage like to view live video feed from the device. 

To change the resolution or rate of streaming, you can ssh into the device and do the following:

    1. stop the current service:
        1. sudo systemctl stop mjpg_streamer_video.service 
    2. change the parameters and restart the service:
        1. edit /usr/bin/mjpg_streamer_video.sh
        2. sudo systemctl restart mjpg_streamer_video.service

Alternatively, you can stop the service as described above and use the script provided in the repository to connect and stream the video.

In a separate terminal:

    ./connect.sh gsr150XX.local 

Password is gelsight

    pi@gsr15demo.local's password: 
    Linux gsr15demo 5.10.17-v7l+ #1414 SMP Fri Apr 30 13:20:47 BST 2021 armv7l

    The programs included with the Debian GNU/Linux system are free software;
    the exact distribution terms for each program are described in the
    individual files in /usr/share/doc/*/copyright.

    Debian GNU/Linux comes with ABSOLUTELY NO WARRANTY, to the extent
    permitted by applicable law.
    MJPG Streamer Version: git rev: 310b29f4a94c46652b20c4b7b6e5cf24e532af39
     i: fps.............: 60
     i: resolution........: 480 x 640
     i: camera parameters..............:

    Sharpness 0, Contrast 0, Brightness 50
    Saturation 0, ISO 0, Video Stabilisation No, Exposure compensation 0
    Exposure Mode 'auto', AWB Mode 'off', Image Effect 'none'
    Metering Mode 'average', Colour Effect Enabled No with U = 128, V = 128
    Rotation 270, hflip No, vflip No
    ROI x 0.000000, y 0.000000, w 1.000000 h 1.000000
     o: www-folder-path......: ./www/
     o: HTTP TCP port........: 8080
     o: HTTP Listen Address..: (null)
     o: username:password....: disabled
     o: commands.............: enabled
     i: Starting Camera
    Encoder Buffer Size 81920


Pressing Ctrl/C will return you to the terminal prompt.
The video will still be running. See the steps below.


If for some reason you are not able to view the data then shutdown and restart device as follows. 
In a separate terminal:

    ssh pi@gsr150XX.local
    ps -aux | grep mjpg_str

       pi         739  5.1  0.0  80944  2008 ?        Sl   14:50   0:00 mjpg_streamer -i input_raspicam.so -fps 60 -x 480 -y 640 -rot 270 -quality 100 -awb off -awbgainR 1 -awbgainB 1 -o output_http.so -w ./www

    kill -9 739
    sudo shutdown now

Connect to the R1.5 again as shown in the steps above


## View raw 2D image
Once you are able to view the raw image, on the web link ```http://gsr150XX.local:8080/?action=stream```, you can now use python scripts to view/save images.

    cd examples
    python3 showimages.py -d gsr150XX.local

## View 3d point cloud data

    cd examples
    python3 show3d.py -d gsr150XX.local


# R1 Setup

The instruction below pertain to GelSight R1 device.

To setup camera in R1 sensor install the XIMEA Linux Software Packge from the official XIMEA website. Details can be found [here](https://www.ximea.com/support/wiki/apis/ximea_linux_software_package).

    wget https://www.ximea.com/downloads/recent/XIMEA_Linux_SP.tgz
    tar xzf XIMEA_Linux_SP.tgz
    cd package
    ./install

After you have installed the XIMEA package you can either use the GUI or use one of the python files (gs_ximea.py / gs_exaample.py). To use the python files you'd need to install the XIMEA python API. To do that just locate the XIMEA Software package that you have untarred (or unzipped). In the above example it's the folder named package.

    cd api/Python/v3

select the folder v3 or v2 depending on your python version and copy all the contents in the folder ximea to your python dist-packages folder.

    sudo cp -r ximea /usr/local/lib/python3.8/dist-packages

To know where the dist-packages folder, open python in a terminal and run

    import site; site.getsitepackages()
    
Test the camera using the XIMEA camera tools

    /opt/XIMEA/bin/xiCamTool

You might have to increase the USB buffer size to read the XIMEA camera if you get an error like this.

    Acquisition failed to start due to insufficient system resources

    HA_USB_Device::Data_Read_Bulk_Async error: -1 endpoint:x81 
    
    Check that /sys/module/usbcore/parameters/usbfs_memory_mb is set to 0.

Simply run this in a terminal to resolve the issue. More details on this can be found here.

    sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb >/dev/null <<<0
    

## Run the example to view the 3d data
    cd examples
    python3 showimages.py -d R1
    python3 show3d.py -d R1


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
This package is under active development. Contact debra_shure@gelsight.com or radhen@gelsight.com if have any questions / comments / suggestions.


