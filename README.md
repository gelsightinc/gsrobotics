SDK for GelSight robotic sensors

# Install the python code

pip3 install .

# Set paths

PYDIR=`pip3 show gelsight | grep -i location | cut -f2 -d" "`
export LD_LIBRARY_PATH=$PYDIR/gelsightcore:$LD_LIBRARY_PATH
export PYTHONPATH=$PYDIR/gelsightcore:$PYDIR/gelsight:$PYTHONPATH

# Connect to R1.5
# In a separate terminal:

./connect.sh <device> 
enter the password gelsight

For example:
    > ./connect gsr15demo.local    

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


# Check the video

open a browser and go to link  <device>:8080/?action=stream

e.g.  http://gsr15demo.local:8080/?action=stream


# Shutdown and restart device
# In a separate terminal:

ssh pi@gsr15demo.local
ps -aux | grep mjpg_str
kill -9 pid
sudo shutdown now

Connect to the R1.5 again as shown in the steps above

# Run the example to view the 3d data
cd examples
python3 show3d.py


# R1 Notes
# Be sure to set the following:

> sudo tee /sys/module/usbcore/paramters/usbfs_memory_mb >/dev/null <<<0

