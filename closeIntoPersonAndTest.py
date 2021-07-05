import os
import sys
from pyparrot.Minidrone import Mambo
from pyparrot.DroneVision import DroneVision
import threading
import cv2
import time
import os   # so we can use command line from python file
import subprocess   # so we can use command line in different directory than file
import csv

# README: this script will close into a single person, using object detection of persons body


def get_parent_dir(n=1):
    """returns the n-th parent dicrectory of the current
    working directory"""
    current_path = os.path.dirname(os.path.abspath(__file__))
    for _ in range(n):
        current_path = os.path.dirname(current_path)
    return current_path


src_path = os.path.join(get_parent_dir(1), "2_Training", "src")
utils_path = os.path.join(get_parent_dir(1), "Utils")

sys.path.append(src_path)
sys.path.append(utils_path)

import argparse
from keras_yolo3.yolo import YOLO, detect_video, detect_webcam
from PIL import Image
from timeit import default_timer as timer
from utils import load_extractor_model, load_features, parse_input, detect_object
import test
import utils
import pandas as pd
import numpy as np
from Get_File_Paths import GetFileList
import random
from Train_Utils import get_anchors

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set up folder names for default values
data_folder = os.path.join(get_parent_dir(n=1), "Data")

image_folder = os.path.join(data_folder, "Source_Images")

image_test_folder = os.path.join(image_folder, "Test_Images")

detection_results_folder = os.path.join(image_folder, "Test_Image_Detection_Results")
detection_results_file = os.path.join(detection_results_folder, "Detection_Results.csv")

model_folder = os.path.join(data_folder, "Model_Weights")

# TODO: replace with Samantha's new weights
model_weights = os.path.join(model_folder, "yolo.h5")
model_classes = os.path.join(model_folder, "coco_classes.txt")

anchors_path = os.path.join(src_path, "keras_yolo3", "model_data", "yolo_anchors.txt")

# TODO: replace with real person and mask labels
person_label = 0
mask_label = 1

FLAGS = None


def run_weights(_out_df):
    print("Started to run weights")
    if input_image_paths and not webcam_active:
        print(
            "Found {} input images: {} ...".format(
                len(input_image_paths),
                [os.path.basename(f) for f in input_image_paths[:5]],
            )
        )
        start = timer()
        text_out = ""

        # This is for images
        for i, img_path in enumerate(input_image_paths):
            print(img_path)
            prediction, image = detect_object(
                yolo,
                img_path,
                save_img=save_img,
                save_img_path=FLAGS.output,
                postfix=FLAGS.postfix,
            )
            y_size, x_size, _ = np.array(image).shape
            for single_prediction in prediction:
                print("referenced out_df")
                _out_df = _out_df.append(
                    pd.DataFrame(
                        [
                            [
                                os.path.basename(img_path.rstrip("\n")),
                                img_path.rstrip("\n"),
                            ]
                            + single_prediction
                            + [x_size, y_size]
                        ],
                        columns=[
                            "image",
                            "image_path",
                            "xmin",
                            "ymin",
                            "xmax",
                            "ymax",
                            "label",
                            "confidence",
                            "x_size",
                            "y_size",
                        ],
                    )
                )
        end = timer()
        print(
            "Processed {} images in {:.1f}sec - {:.1f}FPS".format(
                len(input_image_paths),
                end - start,
                len(input_image_paths) / (end - start),
            )
        )
        _out_df.to_csv(FLAGS.box, index=False)
        print("Finished run weights")


if __name__ == "__main__":
    #------------ STARTING DETECTOR, LOADING WEIGHTS -----------------
    # Delete all default flags
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    """
    Command line options
    """

    parser.add_argument(
        "--input_path",
        type=str,
        default=image_test_folder,
        help="Path to image/video directory. All subdirectories will be included. Default is "
        + image_test_folder,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=detection_results_folder,
        help="Output path for detection results. Default is "
        + detection_results_folder,
    )

    parser.add_argument(
        "--no_save_img",
        default=False,
        action="store_true",
        help="Only save bounding box coordinates but do not save output images with annotated boxes. Default is False.",
    )

    parser.add_argument(
        "--file_types",
        "--names-list",
        nargs="*",
        default=[],
        help="Specify list of file types to include. Default is --file_types .jpg .jpeg .png .mp4",
    )

    parser.add_argument(
        "--yolo_model",
        type=str,
        dest="model_path",
        default=model_weights,
        help="Path to pre-trained weight files. Default is " + model_weights,
    )

    parser.add_argument(
        "--anchors",
        type=str,
        dest="anchors_path",
        default=anchors_path,
        help="Path to YOLO anchors. Default is " + anchors_path,
    )

    parser.add_argument(
        "--classes",
        type=str,
        dest="classes_path",
        default=model_classes,
        help="Path to YOLO class specifications. Default is " + model_classes,
    )

    parser.add_argument(
        "--gpu_num", type=int, default=1, help="Number of GPU to use. Default is 1"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        dest="score",
        default=0.25,
        help="Threshold for YOLO object confidence score to show predictions. Default is 0.25.",
    )

    parser.add_argument(
        "--box_file",
        type=str,
        dest="box",
        default=detection_results_file,
        help="File to save bounding box results to. Default is "
        + detection_results_file,
    )

    parser.add_argument(
        "--postfix",
        type=str,
        dest="postfix",
        default="_catface",
        help='Specify the postfix for images with bounding boxes. Default is "_catface"',
    )

    parser.add_argument(
        "--is_tiny",
        default=False,
        action="store_true",
        help="Use the tiny Yolo version for better performance and less accuracy. Default is False.",
    )

    parser.add_argument(
        "--webcam",
        default=False,
        action="store_true",
        help="Use webcam for real-time detection. Default is False.",
    )

    FLAGS = parser.parse_args()

    save_img = not FLAGS.no_save_img

    file_types = FLAGS.file_types

    webcam_active = FLAGS.webcam

    if file_types:
        input_paths = GetFileList(FLAGS.input_path, endings=file_types)
    else:
        input_paths = GetFileList(FLAGS.input_path)

    # Split images and videos
    img_endings = (".jpg", ".jpeg", ".png")
    vid_endings = (".mp4", ".mpeg", ".mpg", ".avi")

    input_image_paths = []
    input_video_paths = []
    for item in input_paths:
        if item.endswith(img_endings):
            input_image_paths.append(item)
        elif item.endswith(vid_endings):
            input_video_paths.append(item)

    output_path = FLAGS.output
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if FLAGS.is_tiny and FLAGS.anchors_path == anchors_path:
        anchors_path = os.path.join(
            os.path.dirname(FLAGS.anchors_path), "yolo-tiny_anchors.txt"
        )

    anchors = get_anchors(anchors_path)
    # define YOLO detector
    yolo = YOLO(
        **{
            "model_path": FLAGS.model_path,
            "anchors_path": anchors_path,
            "classes_path": FLAGS.classes_path,
            "score": FLAGS.score,
            "gpu_num": FLAGS.gpu_num,
            "model_image_size": (416, 416),
        }
    )

    # Make a dataframe for the prediction outputs
    out_df = pd.DataFrame(
        columns=[
            "image",
            "image_path",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "label",
            "confidence",
            "x_size",
            "y_size",
        ]
    )

    # labels to draw on images
    class_file = open(FLAGS.classes_path, "r")
    input_labels = [line.rstrip("\n") for line in class_file.readlines()]
    print("Found {} input labels: {} ...".format(len(input_labels), input_labels))
    # -----------LOADING WEIGHTS DONE ---------------#

    # ------------ DRONE CONTROL ---------------------#

    # the folder you would like to save images to...
    saveFolder = "images/"

    # the path to Detector.py
    detectorPath = "/Users/zacharywalker-liang/Documents/Research/Drone/pyparrot/programs/my_programs/TrainYourOwnYOLO_Samantha/3_Inference"

    # path to test images
    testPath = "/Users/zacharywalker-liang/Documents/Research/Drone/pyparrot/programs/my_programs/TrainYourOwnYOLO_Samantha/Data/Source_Images/Test_Images/"

    # path to csv results
    resultsPath = "/Users/zacharywalker-liang/Documents/Research/Drone/pyparrot/programs" \
                  "/my_programs/TrainYourOwnYOLO_Samantha/Data/Source_Images/Test_Image_Detection_Results/Detection_Results.csv"


    class UserVision:
        def __init__(self, vision):
            self.index = 0
            self.vision = vision

        def get_latest_picture(self):
            return self.vision.get_latest_valid_picture()

        def save_pictures(self, args):
            # print("in save pictures on image %d " % self.index)

            img = self.vision.get_latest_valid_picture()
            if (img is not None):
                filename = "test_image_%06d.png" % self.index

                cv2.imwrite(saveFolder + filename, img)
                self.index += 1


    # you will need to change this to the address of YOUR mambo
    mamboAddr = "e0:14:d0:63:3d:d0"

    # make my mambo object
    # remember to set True/False for the wifi depending on if you are using the wifi or the BLE to connect
    mambo = Mambo(mamboAddr, use_wifi=True)
    print("trying to connect to mambo now")
    success = mambo.connect(num_retries=3)
    print("connected: %s" % success)

    # holds bounding box of a single person
    class Box:
        def __init__(self, min_x, max_x, min_y, max_y):
            self.min_x = min_x
            self.max_x = max_x
            self.min_y = min_y
            self.max_y = max_y

        def get_min_x(self):
            return self.min_x

        def get_max_x(self):
            return self.max_x

        def get_min_y(self):
            return self.min_y

        def get_max_y(self):
            return self.max_y

        def get_center_x(self):
            return (self.min_x + self.max_x) / 2

        def get_center_y(self):
            return (self.min_y + self.max_y) / 2

    # moves last image captured on drone into test directory (where weights will be run on the image)
    def last_image_to_test_dir():
        _mamboVision = DroneVision(mambo, is_bebop=False, buffer_size=30)
        _userVision = UserVision(mamboVision)
        img = _userVision.get_latest_picture()
        filename = "lastImage.png"
        cv2.imwrite(testPath + filename, img)

    # closes into a person to check if they have a mask on
    # TODO: Write this function
    def close_into_person(target):

        # calculates euclidean distance between two boxes center
        def calculate_distance(box1, box2):
            return pow(pow(box1.get_center_x() - box2.get_center_x(), 2) + pow(box1.get_center_y + box2.get_center_y(), 2), .5)

        last_image_to_test_dir()  # put last drone image into testing directory
        run_weights(out_df)  # run updated target weights
        img_width = None  # total image width
        img_height = None  # total image height
        row_length = None
        with open(resultsPath, newline='') as csvfile:
            reader1 = csv.reader(csvfile, delimiter=',', quotechar='|')
            row_length = 0
            for row in reader1:
                row_length += 1

        # open csv file, make reader to find closest person to target
        # CSV after use
        with open(resultsPath, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            line_count = 0
            closest_person = None  # biggest area of person so far
            least_distance = None

            for row in reader:
                line_count += 1
                if line_count != 1:  # makes sure first row is skipped
                    label = float(row[6])
                else:
                    label = -1  # throwaway label, this is because csv reads first row which is headers

                if label == 0:  # if object detected is a person
                    x_min = float(row[2])
                    y_min = float(row[3])
                    x_max = float(row[4])
                    y_max = float(row[5])
                    detected_person = Box(x_min, x_max, y_min, y_max)
                    area = abs(x_max - x_min) * abs(y_max - y_min)
                    if closest_person is None:  # if no person found yet, this person is biggest area
                        closest_person = detected_person  # TODO: beware, could be a reference problem
                    elif calculate_distance(target, detected_person) < least_distance:  # if this person is closest to target
                        least_distance = calculate_distance(target, detected_person)
                        closest_person = detected_person

                    # on last iteration, call our function to center drone
                    if line_count == row_length and closest_person is not None:
                        img_width = float(row[8])
                        img_height = float(row[9])
                        print("Person detected, centering drone....")
                        center_drone(closest_person, area, img_width, img_height)



    # surveys a scene, returns a person who does not have mask inside of bounding box
    def find_test_person():
        def mask_in_person(person, mask):
            mask_center_x = (mask.get_min_x() + mask.get_max_x()) / 2
            mask_center_y = (mask.get_min_y() + mask.get_max_y()) / 2
            if (person.get_min_x() < mask_center_x < person.get_max_x()
                    and person.get_min_y() < mask_center_y < person.get_max_y()):
                return True
            else:
                return False

        subject = None

        # finds the length of lines in csv file
        with open(resultsPath, newline='') as csvfile:
            reader1 = csv.reader(csvfile, delimiter=',', quotechar='|')
            row_length = 0  # length of lines in csv file
            for row in reader1:
                row_length += 1

        # finds a person where it can't tell if a mask is being worn
        with open(resultsPath, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            line_count = 0

            #  finds a person
            for row in reader:
                line_count += 1
                if line_count != 1:  # makes sure first row is skipped
                    label = float(row[6])
                else:
                    label = -1

                # if found person, look through csv again and try to find a mask on that person
                if label == person_label:  # if object detected is a person
                    found_mask = False
                    line_count2 = 0
                    for row2 in reader:  # go through csv again, line by line
                        line_count2 += 1
                        if line_count2 != 1:
                            label2 = float(row2[6])
                            if label == mask_label:  # if is mask label
                                subject = Box(row[2], row[4], row[3], row[5])  # person subject
                                if mask_in_person(subject, Box(row2[2], row2[4], row2[3], row2[5])):
                                    found_mask = True
                    if found_mask:
                        pass
                    else:
                        return subject

            return None  # person without mask not found

    # centers drone to person depth, depending on size of person box
    # TODO: add functionality to drop focus when mask is found on person, add functionality where if close enough,
    # TODO: and no mask/ incorrect mask, drone will print non / incorrect masked person found
    def center_drone(person, area, img_width, img_height):

        # PLANE CENTERING
        # normal value between -0.5 and 0.5, 0 is when drone centered, positive is when face is to right of drone
        x_center = (person.get_center_x() / img_width) - 0.5
        # normal value between -0.5 and 0.5, 0 is when drone centered, positive is when person is above drone
        y_center = ((person.get_center_y() / img_height) - 0.5) * -1

        print("x_center: " + str(x_center))
        print("y_center: " + str(y_center))

        # vertical centering
        if y_center > .1:
            mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=10, duration=.2)
            print("Adjusting drone, moving upwards")
            mambo.smart_sleep(.5)
        elif y_center < -.1:
            mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-20, duration=.2)
            print("Adjusting drone, moving downwards")
            mambo.smart_sleep(.5)
        else:
            print("Did not need to adjust drone vertically")

        # horizontal centering
        if x_center > .15:
            if x_center > .4:
                mambo.turn_degrees(40)
                print("Adjusting drone, rotating right macro")
            else:
                mambo.turn_degrees(20)
                print("Adjusting drone, rotating right micro")
            mambo.smart_sleep(.5)
        elif x_center < -.15:
            if x_center < -.4:
                mambo.turn_degrees(-40)
                print("Adjusting drone, rotating left macro")
            else:
                mambo.turn_degrees(-20)
                print("Adjusting drone, rotating micro")
            mambo.smart_sleep(.5)
        else:
            print("Did not need to adjust drone horizontally")

        # DEPTH CENTERING
        target_area = 14000  # pixel x pixel
        total_area = img_width * img_height

        # if normal area is above 0, it means drone is too close
        # if normal area is below 0, it means drone is too far
        normal_area = area - target_area  # value between -0.5 and 0.5

        print("normal area" + str(normal_area))
        BUFFER = 5000  # area within drone is safe

        if normal_area > BUFFER:
            mambo.fly_direct(roll=0, pitch=-50, yaw=0, vertical_movement=0, duration=.3)
            print("Adjusting drone, moving farther")
            mambo.smart_sleep(.5)
        elif normal_area < -BUFFER:
            mambo.fly_direct(roll=0, pitch=50, yaw=0, vertical_movement=0, duration=.3)
            print("Adjusting drone, moving closer")
            mambo.smart_sleep(.5)
        else:
            print("Did not need to adjust drone depth")

    # main program down here --
    if success:
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

        if success:
            # get the state information, ready for takeoff

            print("sleeping")
            mambo.smart_sleep(2)
            mambo.ask_for_state_update()
            mambo.smart_sleep(2)

            print("Vision successfully started!")

            # take off
            print("taking off!")
            mambo.safe_takeoff(5)

            print("flying upwards")
            # fly upwards for a little
            mambo.smart_sleep(4)
            mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=30, duration=.5)
            mambo.smart_sleep(2)

            focus = None

            # MAIN CONTROL ALGORITHM
            while True:
                if focus is not None:  # found a target to focus on with no mask
                    close_into_person(focus)
                else:  # did not find a target to focus on
                    last_image_to_test_dir()
                    run_weights(out_df)
                    focus = find_test_person()
                    if focus is None:  # no interesting person is found, rotate and try again
                        mambo.turn_degrees(20)

            mambo.smart_sleep(5)

            mambo.safe_land(5)
            mambo.smart_sleep(1)
            # done doing vision demo
            print("Ending the sleep and vision")
            mamboVision.close_video()
            mambo.smart_sleep(5)

        print("disconnecting")
        mambo.disconnect()

    # Close the current yolo session
    yolo.close_session()
