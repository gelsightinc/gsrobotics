#!/bin/bash

if [ "$#" -eq "0" ]
then
    echo "USAGE $0 <raspberry pi name or address>" && exit
else
    RASPPI=$1
fi

ssh -T pi@$RASPPI <<EOF
    mjpg_streamer -i "input_raspicam.so -fps 60 -x 240 -y 320 -rot 270 -quality 100 -awb off -awbgainR 1.0 -awbgainB 1.0" -o "output_http.so -w ./www"
EOF

