# Created by Samantha Wong, modified for Drone use by Zachary Walker-Liang

import numpy as np
import cv2
import random
from scipy.spatial import distance
import matplotlib.pyplot as plt
import csv
from datetime import datetime


def check(raw_img_file_path, csv_file_path, result_img_folder_path, person_label):
    num = 0
    with open(csv_file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        lines = 0
        for row in csv_reader:
            test = person_label
            if test == row[6]:
                num = num + 1
    person = np.zeros(shape=(num, 4))
    count = 0
    with open(csv_file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                if row[6] == person_label:
                    person[count] = [row[2], row[4], row[3], row[5]]
                    count += 1
                    line_count += 1
                else:
                    line_count += 1
    img = \
        cv2.imread(raw_img_file_path
                   )
    bottomcenter = []
    heights = []
    distances = np.zeros(shape=(num, num))
    person1 = []
    person2 = []
    notdist = []
    test = 0
    for count in range(len(person)):
        (x1, x2, y1, y2) = person[count]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255,
                      0), 3)
        xmiddle = int((x1 + x2) / 2)
        ymiddle = int(y2)
        middle = (xmiddle, ymiddle)
        bottomcenter.append(middle)
        cv2.putText(
            img,
            str(count),
            middle,
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
            )
        height = abs(y2 - y1)
        heights.append(height)
    for a in range(num):
        for b in range(a + 1, num):
            if a != b:
                distances[a][b] = distance.euclidean(bottomcenter[a],
                        bottomcenter[b])
    for c in range(num):
        for d in range(c, num):
            if (c != d) & (distances[c][d] <= heights[c]):
                person1.append(c)
                person2.append(d)
    for e in range(len(person1)):
        if person1[e] not in notdist:
            notdist.append(person1[e])
    for f in range(len(person2)):
        if person2[f] not in notdist:
            notdist.append(person2[f])
    notdist.sort()
    for h in notdist:
        (x1, x2, y1, y2) = person[h]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0,
                      0), 3)
    plt.figure(figsize=(20, 10))
    #plt.imshow(img)
    #plt.show()

    # Save plot as image to directory, added by Zachary Walker-Liang
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string = dt_string.replace("/", "-")
    dt_string = dt_string.replace(":", "$")
    print(dt_string)

    if len(notdist) > 0:
        plt.imsave((result_img_folder_path + '{}').format('socialDistancingFig' + dt_string + '.png'), img)
        print("Social distancing violation detected, saving image")
    else:
        print("No Social distancing violation found")


if __name__ == '__main__':
    check('/Users/zacharywalker-liang/Documents/Research/Drone/Examples/exampleGood.jpg',
          '/Users/zacharywalker-liang/Documents/Research/Drone/Examples/Detection_Results2.csv',
          '/Users/zacharywalker-liang/Documents/Research/Drone/Examples/', '2')


