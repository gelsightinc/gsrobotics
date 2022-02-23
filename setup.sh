#!/bin/bash

PYDIR=`pip3 show gelsight | grep -i location | cut -f2 -d" "`

export LD_LIBRARY_PATH=$PYDIR/gelsightcore:$LD_LIBRARY_PATH

export PYTHONPATH=$PYDIR/gelsightcore:$PYDIR/gelsight:$PYTHONPATH

sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb >/dev/null <<<0
