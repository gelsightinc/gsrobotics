# tactile_sdk_local
SDK for GelSight robotic sensors

###########################
# 3d reconstruction
###########################

gs_raspi_param.ini - set input/output directories

gs_ximea.py - c enter, s for single images, q to quit

find_bga_r1.py - finds circles puts it in circles directory

nn_train_bga.py - trains the nn using the bga

nn_rgb2nrm_test.py - view the results of the 3d reconstruction, realtime 3d


R1.5

###########################
#dot tracking
###########################
utils/live_ximea.py

R1   Nvidia and Haptix 14 devices out


Connect to R1

> sudo tee /sys/module/usbcore/paramters/usbfs_memory_mb >/dev/null <<<0

marker tracking

use tracking.py

Tune settings

setting.py - number of rows/cols dots

	   - x0, y0 : location of top left upper corner
	   - dx, dy : distance between dots


want to automatically find setting.py

