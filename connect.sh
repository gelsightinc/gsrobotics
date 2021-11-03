#!/bin/bash

ssh pi@gsr15demo <<EOF
    mjpg_streamer -i "input_raspicam.so -fps 60 -x 480 -y 640 -rot 270 -quality 100 -awb off -awbgainR 1 -awbgainB 1" -o "output_http.so -w ./www"
EOF

