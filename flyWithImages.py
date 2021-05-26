"""
Demo of the ffmpeg based mambo vision code (basically flies around and saves out photos as it flies)

Author: Amy McGovern
"""
from pyparrot.Minidrone import Mambo
from pyparrot.DroneVision import DroneVision
import threading
import cv2
import time


# set this to true if you want to fly for the demo
testFlying = False

# the folder you would like to save images to...
folder = "images/";

class UserVision:
    def __init__(self, vision):
        self.index = 0
        self.vision = vision

    def save_pictures(self, args):
        print("in save pictures on image %d " % self.index)

        img = self.vision.get_latest_valid_picture()

        if (img is not None):
            filename = "test_image_%06d.png" % self.index
            #cv2.imwrite(filename, img)
            cv2.imwrite(folder + filename, img)
            self.index +=1
            #print(self.index)



# you will need to change this to the address of YOUR mambo
mamboAddr = "e0:14:d0:63:3d:d0"

# make my mambo object
# remember to set True/False for the wifi depending on if you are using the wifi or the BLE to connect
mambo = Mambo(mamboAddr, use_wifi=True)
print("trying to connect to mambo now")
success = mambo.connect(num_retries=3)
print("connected: %s" % success)

if (success):
    # get the state information
    print("sleeping")
    mambo.smart_sleep(1)
    mambo.ask_for_state_update()
    mambo.smart_sleep(1)

    print("Preparing to open vision")
    mamboVision = DroneVision(mambo, is_bebop=False, buffer_size=30)
    userVision = UserVision(mamboVision)
    mamboVision.set_user_callback_function(userVision.save_pictures, user_callback_args=None)
    success = mamboVision.open_video()
    print("Success in opening vision is %s" % success)

    if (success):
        # get the state information, ready for takeoff
        print("sleeping")
        mambo.smart_sleep(2)
        mambo.ask_for_state_update()
        mambo.smart_sleep(2)

        print("Vision successfully started!")

        # take off
        print("taking off!")
        mambo.safe_takeoff(5)

        # fly foward
        mambo.smart_sleep(1)
        mambo.fly_direct(roll=0, pitch=50, yaw=0, vertical_movement=0, duration= .75)
        mambo.smart_sleep(1)

        # rotate and take pictures
        for x in range(8):
            mambo.turn_degrees(45)
            mambo.smart_sleep(.1)

        mambo.smart_sleep(1)

        mambo.safe_land(5)
        mambo.smart_sleep(1)
        # done doing vision demo
        print("Ending the sleep and vision")
        mamboVision.close_video()

        mambo.smart_sleep(5)

    print("disconnecting")
    mambo.disconnect()
