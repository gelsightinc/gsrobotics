# tactile_sdk_local
SDK for GelSight robotic sensors


###############################
# install the python code
###############################
pip3 install .

##############################
# set paths
##############################
PYDIR=`pip3 show gelsight | grep -i location | cut -f2 -d" "`
export LD_LIBRARY_PATH=$PYDIR/gelsightcore:$LD_LIBRARY_PATH
export PYTHONPATH=$PYDIR/gelsightcore:$PYDIR/gelsight:$PYTHONPATH

#############################
# connect to R1.5
#############################
sh connect.sh 
enter the password gelsight

###########################
# check the stream
###########################
open a browser and go to link
gsr15demo.local:8080/?action=stream


################################
# shutdown and restart
###############################
ps -aux | grep mjpg_str
kill -9 pid
sudo shutdown now

Connect to the R1.5 again as shown above

################################
# run the example to show3d
################################
cd examples
python3 show3d.py



---------------------------------------------------------------
#################################
# Connect to R1 
#################################

> sudo tee /sys/module/usbcore/paramters/usbfs_memory_mb >/dev/null <<<0

