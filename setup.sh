#!/bin/bash

PYDIR=`pip3 show gelsight | grep -i location | cut -f2 -d" "`

export PYTHONPATH=$PYDIR/gelsight:$PYTHONPATH
