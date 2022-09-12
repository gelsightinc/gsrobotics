v4l2-ctl -d /dev/video$1 --list-ctrls

v4l2-ctl --device /dev/video$1 -c brightness=0
v4l2-ctl --device /dev/video$1 -c contrast=32
v4l2-ctl --device /dev/video$1 -c saturation=64
v4l2-ctl --device /dev/video$1 -c hue=0
v4l2-ctl --device /dev/video$1 -c white_balance_temperature_auto=0
v4l2-ctl --device /dev/video$1 -c gamma=140
v4l2-ctl --device /dev/video$1 -c gain=0
v4l2-ctl --device /dev/video$1 -c power_line_frequency=0
v4l2-ctl --device /dev/video$1 -c white_balance_temperature=4500
v4l2-ctl --device /dev/video$1 -c sharpness=3
v4l2-ctl --device /dev/video$1 -c backlight_compensation=0
v4l2-ctl --device /dev/video$1 -c exposure_auto=1
v4l2-ctl --device /dev/video$1 -c exposure_absolute=180
v4l2-ctl --device /dev/video$1 -c exposure_auto_priority=0
#v4l2-ctl --device /dev/video0 -c focus_absolute=1
#v4l2-ctl --device /dev/video0 -c focus_auto=1

