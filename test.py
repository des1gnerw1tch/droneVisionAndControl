
from pyparrot.Minidrone import Mambo
# address is insignifigant when using WIFI
mamboAddr = "e0:14:d0:63:3d:d0"

# make my mambo object
# remember to set True/False for the wifi depending on if you are using the wifi or the BLE to connect
mambo = Mambo(mamboAddr, use_wifi=True)

print("trying to connect")
success = mambo.connect(num_retries=3)
print("connected: %s" % success)

if (success):
    # get the state information, ready for takeoff
    print("sleeping")
    mambo.smart_sleep(2)
    mambo.ask_for_state_update()
    mambo.smart_sleep(2)

    # take off
    print("taking off!")
    mambo.safe_takeoff(5)

    mambo.smart_sleep(1)
    mambo.fly_direct(roll=0, pitch=50, yaw=0, vertical_movement=0, duration= .75)
    mambo.smart_sleep(1)

    # rotate and take pictures
    for x in range(8):
        mambo.turn_degrees(45)
        #mambo.take_picture()
        print("take picture")
        #print(mambo.get_groundcam_pictures_names())
        mambo.smart_sleep(1)

    mambo.smart_sleep(1)

    # landing
    print("landing")
    mambo.safe_land(5)
    mambo.smart_sleep(5)

    print("disconnect")
    mambo.disconnect()
