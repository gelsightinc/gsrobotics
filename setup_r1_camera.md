# Setup camera in R1 sensor


If you haven't already, install the XIMEA Linux Software Packge from the official XIMEA website. Details can be 
[found here](https://www.ximea.com/support/wiki/apis/XIMEA_Linux_Software_Package#Installation). 

`wget https://www.ximea.com/downloads/recent/XIMEA_Linux_SP.tgz` 

Untar \
`tar xzf XIMEA_Linux_SP.tgz` \
`cd package` 
 
After you have installed the XIMEA package you can either use the GUI or use one of the python files (gs_ximea.py / 
gs_exaample.py). To use the python files you'd need to install the XIMEA python API. To do that just locate the XIMEA 
Software package that you have untarred (or unzipped). In the above example it's the folder named **package**.

`cd package/api/Python/v3`

select the folder v3 or v2 depending on your python version and copy all the contents in the folder **ximea** to your 
python dist-packages folder. 

`cp -r ximea /usr/local/lib/python3.8/dist-packages` 

To know where the dist-packages folder, open python in a terminal and run 

`import site; site.getsitepackages()`

You might have to increase the USB buffer size to read the XIMEA camera if you get an error like this.

`HA_USB_Device::Data_Read_Bulk_Async error: -1 endpoint:x81
Check that /sys/module/usbcore/parameters/usbfs_memory_mb is set to 0.`

Simply run this in a terminal to resolve the issue. More details on this can be [found here](https://www.ximea.com/support/wiki/apis/Linux_USB30_Support). 

`sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb >/dev/null <<<0`

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## Contact 
This package is under active development. Contact radhen@gelsight.com if have any questions / comments / suggestions. 


## License
All rights reserved to GelSight Inc. 
