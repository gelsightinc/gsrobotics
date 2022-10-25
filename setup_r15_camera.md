# Setup camera in R1.5 sensor


In case you are starting with a fresh Raspberry Pi, follow the initial steps below to install the Raspbian OS on the micro SD card, otherwise skip to next section 

## Initial steps
1. [Download Raspberry Pi OS with desktop and recommended software](https://www.raspberrypi.org/downloads/raspberry-pi-os/)
2. [Download balenaEtcher](https://www.balena.io/etcher) (or Raspberry Pi Imager)
3. Run balenaEtcher to flash Rapbian OS into micro SD card 
4. Initialize Raspberry Pi
    1. Insert micro SD
    2. Connect raspberry pi zero to power, monitor, keyboard, mouse
    3. Turn on raspberry pi
    4. First configuration:
        1. Setup country, language, timezone
        2. Change password: gelsight
        3. Select Wifi
        4. Skip Update Software
        5. Reboot
    5. Second configuration:
        1. Applications (top left raspberry icon) => Preferences => Raspberry Pi Configuration
        2. Change System => Hostname: gsr1500X
        3. Enable Interface => Camera & SSH
        4. Click OK => Yes to reboot
5. Install mjpg_streamer:
    1. Open terminal: Ctrl + Alt + T
    2. `git clone https://github.com/jacksonliam/mjpg-streamer.git`
    3. `sudo apt-get update`
    4. `sudo apt-get install cmake libjpeg8-dev gcc g++ -y`
    5. `cd mjpg-streamer/mjpg-streamer-experimental/`
    6. `make`
    7. `sudo make install`


![R1.5 connections](https://github.com/gelsightinc/gsrobotics/blob/main/r15_connections.jpg?raw=true)

## Start streaming (linux)
1. Connect R1.5 to Raspberry Pi and your Linux computer as shows in the above image, and power up the Pi. 
2. Add raspberry pi network to Ubuntu:
    1. Go to, Network Settings -> Wired connection
    2. Add a new ethernet network by clicking the plus sign on the top right corner
    3. In the field Name, put **gsr1500X**. 
    4. In the same window, go to, IPv4 Setting -> Method -> select **Link Local only**
    5. Save and exit
3. SSH into raspberry pi from PC
    1. Open terminal: Ctrl + Alt + T
    2. ssh pi@gsr1500X.local 
    3. password: gelsight 
    4. Now you are ssh'd into raspberry pi. Run the below command in the same terminal window.
       1. `mjpg_streamer -i "input_raspicam.so -fps 60 -x 480 -y 640 -rot 270 -quality 100 -awb off -awbgainR 1 -awbgainB 1" -o "output_http.so -w ./www"`
    5. Open a browser (Google Chrome or Firefox), and copy and paste the link below to preview video streaming.  Remember to replace the x in gsr1500x with appropriate ID.
        1. http://gsr1500x.local:8080/?action=stream
    6. IP to view camera data directly on pi - http://localhost:8080/?action=stream


## Setup automatic video streaming on start (linux)

SSH into the Raspberry Pi. Open a terminal and run,

1. nano ~/run.sh
2. Type in: `mjpg_streamer -i "input_raspicam.so -fps 60 -x 480 -y 640 -rot 270 -quality 100 -awb off -awbgainR 1 -awbgainB 1" -o "output_http.so -w ./www"`
3. Save: Control + X
4. Save modified buffer? Y => Enter
5. chmod +x ~/run.sh
6. nano ~/.bashrc
7. Type in at the end: ./run.sh
8. Save: Control + X => Y => Enter


## Start streaming (Windows)

1. Connect R1.5 to Raspberry Pi and your Windows computer as shows in the above image, and power up the Pi.
2. Follow [this link](https://jarrodstech.net/how-to-raspberry-pi-ssh-on-windows-10/#comments) to ssh into Raspberry Pi using a Windows OS.
3. Open a browser (Google Chrome or Firefox), and copy and paste the link below to preview video streaming. Remember to replace the x in gsr1500x with appropriate ID.  
        1. http://gsr1500x.local:8080/?action=stream

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## Contact 
This package is under active development. Contact radhen@gelsight.com if have any questions / comments / suggestions. 


## License
All rights reserved to GelSight Inc. 
