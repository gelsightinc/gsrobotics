# Steps to use [Gelsight Mini](https://www.gelsight.com/) with [Pytouch](https://facebookresearch.github.io/PyTouch/) 
____


1. Locate and edit the configuration file for gelsight in [pytouch's source code](https://github.com/facebookresearch/pytouch). Usually its present at, 

```pytouch/sensors/gelsight.py file.```


2. Set values for SCALE, MEANS and STD. for your gelsight mini. SCALE defines the size of the image in pixels. MEAN and STD are the values by which image is normalized before fed into the machine learning models. 

 ```
class GelsightSensorDefaults:
    SCALES = [64, 64]
    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]
```

3. Initialize pytouch for mini [as shown here](https://facebookresearch.github.io/PyTouch/docs/tutorials/intro) by importing GelSightSensor


```from pytouch.sensors import GelSightSensor```


4. Follow Basic Tutorial and/or Task Tutorial from the [Pytouch official documentation page](https://facebookresearch.github.io/PyTouch/), to read the saved data and generate derived signals.  
