# SDK for GelSight robotic sensors

# Install the python code

    pip3 install .
    or 
    pip3 install . --upgrade

# Set paths

    PYDIR=`pip3 show gelsight | grep -i location | cut -f2 -d" "`

    export LD_LIBRARY_PATH=$PYDIR/gelsightcore:$LD_LIBRARY_PATH

    export PYTHONPATH=$PYDIR/gelsightcore:$PYDIR/gelsight:$PYTHONPATH


# R1.5 device setup

Your device has been configured with the following settings:

Username: pi

Password: gelsight
    
The address name of the R1.5 is on shown on small white sticker, right above the ethernet port on the Pi.

e.g.  gsr1500x.local

Add this address to your wired network:

    Go to, Network Settings -> Wired connection
    Add a new ethernet network by clicking the plus sign on the top right corner
    In the field Name, put gsr1500X (name on the sticker without the .local)
    In the same window, go to, IPv4 Setting -> Method -> select Link Local only
    Save and exit

# Setting up a static IP address

If you want to change it so that you are using a static IP address, follow the instructions here:

https://raspberrypi-guide.github.io/networking/set-up-static-ip-address


# R1.5

# Check the video is running

Open a browser and go to link  
  http://gsr1500x.local:8080/?action=stream


A service is running on the R1.5 that streams video once the device is turned on.  It is streaming video at 60 fps at a resolution of 240x320.

To change the resolution or rate of streaming, you can ssh into the device and do the following:

    1. stop the current service:
        1. sudo systemctl stop mjpg_streamer_video.service 
    2. change the parameters and restart the service:
        1. edit /usr/bin/mjpg_streamer_video.sh
        2. sudo systemctl restart mjpg_streamer_video.service

Alternatively, you can stop the service as described above and use the script provided in the repository to connect and stream the video.

In a separate terminal:

    ./connect.sh gsr15demo.local 

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


# Shutdown and restart device
In a separate terminal:

    ssh pi@gsr15demo.local
    ps -aux | grep mjpg_str

       pi         739  5.1  0.0  80944  2008 ?        Sl   14:50   0:00 mjpg_streamer -i input_raspicam.so -fps 60 -x 480 -y 640 -rot 270 -quality 100 -awb off -awbgainR 1 -awbgainB 1 -o output_http.so -w ./www

    kill -9 739
    sudo shutdown now

Connect to the R1.5 again as shown in the steps above

# Run the example to view the 3d data

    cd examples
    python3 show3d.py -d gsr15demo.local


# R1 Setup

Setup camera in R1 sensor

Install the XIMEA Linux Software Packge from the official XIMEA website. Details can be found here.

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
    
# Run the example to view the 3d data

    cd examples
    python3 show3d.py -d R1


# Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

# Contact

This package is under active development. Contact radhen@gelsight.com if have any questions / comments / suggestions.


