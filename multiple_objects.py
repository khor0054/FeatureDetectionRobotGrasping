import torch
import cv2
import math
import numpy as np
from torch import nn
from detect import detect_corners_lines, show_detections
from outline_center import find_centroid
from skspatial.objects import Line, Point
from scipy.spatial import ConvexHull
from realsense_capture import capture
from grasp import initialize_open

mode = 2
print("\n>> Mode 2: Continuous pick and place.")
initialize_open()

focal = 618
depth = 0.46
max_actual = 0.11

max_grip = max_actual * (focal/depth)

weights = "/home/kaisherng/Documents/yolov5_obb/runs/train/cornell500/weights/best.pt"
directory = "/home/kaisherng/Documents/yolov5_obb/image_captures/capture.png"
# directory = "/home/kaisherng/Documents/images/"

# pred format: list of [X center, Y center, box length, box height, angle, confidence, class]

def grasp():
    while True:
        capture()
        # run detect.py and return list of detections
        pred, img = detect_corners_lines(weights, directory)

        # determine object centroid
        objectX, objectY, contours = find_centroid(img)

        if objectX == 0 or objectY == 0 or len(contours) == 0:
            corner, edge = 0, 0
            b = pred[0]

            for x in b:
                if x[6] == 0:
                    corner = corner + 1
                if x[6] == 1:
                    edge = edge + 1

            if corner == 0 and edge == 0:
                break

            print("\n>> Object(s) are still present.")
            print("\n>>", corner, "corner(s) and", edge, "edge(s) detected.")
            print("\n>> Poor contrast object. Finding estimated centroid.")
            all = find_all(pred)

            if len(all) == 0:
                print("\n>> No YOLO detections. Unable to grasp object.")


            if 0 < len(all) <= 2:
                print("\n>> Not enough detections for estimated centroid. Proceeding to grasp.")
                non_circular(pred, 320, 240, [], all)

            if len(all) > 2:
                hull = ConvexHull(all)
                objectX = int(np.mean(hull.points[hull.vertices, 0]))
                objectY = int(np.mean(hull.points[hull.vertices, 1]))

                if corner == 3:
                    print("\n>> Object is of triangular shape.")
                    triangular(pred, objectX, objectY, [], all)
                else:
                    print("\n>> Object is of non-triangular shape.")
                    non_circular(pred, objectX, objectY, [], all)

        else:
            corner, edge = 0, 0
            b = pred[0]

            for x in b:
                if x[6] == 0:
                    corner = corner + 1
                if x[6] == 1:
                    edge = edge + 1

            print("\n>>", corner, "corner(s) and", edge, "edge(s) detected.")

            # determine object shape
            if len(pred[0]) == 0:
                # if circular shape
                print("\n>> Object is of circular shape.")
                circular(pred, objectX, objectY, contours)

            if len(pred[0]) != 0:
                print("\n>> Object is of non-circular shape.")
                area = cv2.contourArea(contours)
                all = find_all(pred)

                if len(all) >= 3:
                    hull = cv2.convexHull(all)
                    bounded_area = cv2.contourArea(hull)

                    if 0.8*area <= bounded_area <= 1.2*area:
                        print("\n>> Detections are a good estimate of object shape.")
                    else:
                        print("\n>> Detections are a poor estimate of object shape.")

                if corner == 3:
                    print("\n>> Object is of triangular shape.")
                    triangular(pred, objectX, objectY, contours, all)
                else:
                    print("\n>> Object is of non-triangular shape.")
                    non_circular(pred, objectX, objectY, contours, all)

    print("\n>> No more objects detected.")


def circular(pred, objectX, objectY, c):
    left = tuple(c[c[:, :, 0].argmin()][0])
    right = tuple(c[c[:, :, 0].argmax()][0])
    all =[]

    (aX, aY) = left
    (bX, bY) = right

    distance = math.sqrt((aX - bX) ** 2 + (aY - bY) ** 2)
    if distance > max_grip:
        print("\n>> Object diameter is larger than gripper max opening. Attempting to grip from a point.")
        bX = aX+10
        bY = aY
        aX = aX-10

        left = (aX, aY)
        right = (bX, bY)

    show_detections(mode, objectX, objectY, c, pred, left, right, all, weights, directory)


def non_circular(pred, objectX, objectY, contours, all):
    a = pred[0]

    # create 1 new column with values 0
    pad_value = 0
    pad_func = nn.ConstantPad1d((0, 1, 0, 0), pad_value)
    a = pad_func(a)

    # calculate distance from centroid and add to 8th column
    for i in a:
        x = i[0]
        y = i[1]
        dist = math.sqrt((x - objectX) ** 2 + (y - objectY) ** 2)
        i[7] = dist

    # sort all detections by distance in ascending order
    a = a[a[:, 7].sort()[1]]
    b = a
    # remove corner detections
    a = a[a[:, 6] == 0]

    if len(a) >= 2:
        print("\n>> Searching for parallel pairs.")
        grip1 = a[0]
        grip2 = a[1]
        left = 0
        right = 0

        # compare angles
        for i in a:
            for j in a:
                if i[0] == j[0] and i[1] == j[1] and i[2] == j[2] and i[3] == j[3]:
                    continue
                angle = abs(math.degrees(i[4]) - math.degrees(j[4]))

                if angle < 25:
                    print("\n>> Parallel pair found.")
                    aX = int(i[0])
                    aY = int(i[1])
                    bX = int(j[0])
                    bY = int(j[1])
                    xdist = bX - aX
                    ydist = aY - bY

                    if xdist == 0 or ydist == 0:
                        continue
                    orientation = math.degrees(math.atan(ydist/xdist))     # angle of perpendicular distance

                    if orientation >= 180:
                        orientation = orientation - 180

                    diff = abs(orientation - math.degrees(i[4]))

                    if diff < 45:
                        print("\n>> Pair not graspable. Finding next suitable pair.")
                        continue

                    dist = math.sqrt(xdist*xdist + ydist*ydist)

                    if dist > max_grip:
                        print("\n>> Grasp point larger than gripper. Finding next suitable pair.")
                        continue

                    if i[2] < j[2]:
                        shorter = i
                        longer = j
                    else:
                        shorter = j
                        longer = i

                    point1 = (int(shorter[0]), int(shorter[1]))
                    y_diff = 0.5 * longer[2] * math.sin(longer[4])
                    x_diff = 0.5 * longer[2] * math.cos(longer[4])

                    if math.degrees(longer[4]) < 90:
                        point2 = (int(longer[0] + x_diff), int(longer[1] - y_diff))
                        point3 = (int(longer[0] - x_diff), int(longer[1] + y_diff))
                    else:
                        point2 = (int(longer[0] - x_diff), int(longer[1] - y_diff))
                        point3 = (int(longer[0] + x_diff), int(longer[1] + y_diff))

                    line = Line.from_points(point2, point3)
                    point = Point(point1)

                    point_projected = tuple(line.project_point(point))
                    grip1 = i
                    grip2 = j
                    left = point1
                    right = (int(point_projected[0]), int(point_projected[1]))
                    print("\n>> Grasp position optimized.\n")
                    break

            if left != 0 and right != 0:
                break

        if left == 0 and right == 0:
            print("\n>> No parallel pairs found. Attempting to pick at a point.\n")
            pred2, left, right = aim_one(b)
        else:
            # remove last column and some reformatting of tensor
            pred2 = torch.stack((grip1, grip2), 0)
            pred2 = [pred2[:, [0, 1, 2, 3, 4, 5, 6]]]
    else:
        print("\n>> Insufficient edges found. Attempting to pick at a point.\n")
        pred2, left, right = aim_one(b)

    show_detections(mode, objectX, objectY, contours, pred2, left, right, all, weights, directory)

def triangular(pred, objectX, objectY, contours, all):
    a = pred[0]
    a = a[a[:, 2].sort()[1]]

    edge = a[-1]
    angle = edge[4]
    x = objectX
    y = objectY
    dist = 50

    if angle > 0:
        xdiff = dist * math.sin(angle)
        ydiff = dist * math.cos(angle)
        print(xdiff, ydiff)
        left = (int(x - xdiff), int(y - ydiff))
        right = (int(x + xdiff), int(y + ydiff))
    if angle < 0:
        angle = abs(angle)
        xdiff = dist * math.sin(angle)
        ydiff = dist * math.cos(angle)
        left = (int(x - xdiff), int(y + ydiff))
        right = (int(x + xdiff), int(y - ydiff))
    if angle == 0:
        left = (int(x), int(y + 20))
        right = (int(x), int(y - 20))

    pred2 = torch.stack((edge, edge), 0)
    pred2 = [pred2[:, [0, 1, 2, 3, 4, 5, 6]]]

    show_detections(mode, objectX, objectY, contours, pred2, left, right, all, weights, directory)

def aim_one(b):
    i = b[0]
    for j in b:
        if j[6] == 0:
            i = j
            break

    x = i[0]
    y = i[1]
    dist = 20
    angle = i[4]
    if angle > 0:
        xdiff = dist * math.sin(angle)
        ydiff = dist * math.cos(angle)
        left = (int(x - xdiff), int(y - ydiff))
        right = (int(x + xdiff), int(y + ydiff))
    if angle < 0:
        angle = abs(angle)
        xdiff = dist * math.sin(angle)
        ydiff = dist * math.cos(angle)
        left = (int(x - xdiff), int(y + ydiff))
        right = (int(x + xdiff), int(y - ydiff))
    if angle == 0:
        left = (int(x), int(y + 20))
        right = (int(x), int(y - 20))

    pred = torch.stack((i, i), 0)
    pred = [pred[:, [0, 1, 2, 3, 4, 5, 6]]]

    return pred, left, right


def find_all(pred):
    a = pred[0]
    points = []
    for i in a:
        points.append((int(i[0]), int(i[1])))

        """
        if i[6] == 1:
            points.append((int(i[0]), int(i[1])))
        """

    points = np.array(points)

    return points


grasp()

