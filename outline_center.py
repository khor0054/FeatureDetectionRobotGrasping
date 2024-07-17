import cv2
import math
from grasp import execute_grasp, execute_pick_and_place
from hulls import ConcaveHull


def find_centroid(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert the image to grayscale
    blur = cv2.GaussianBlur(gray_image, (5, 5), cv2.BORDER_DEFAULT)

    ret, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)  # convert the grayscale image to binary image

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)  # find contours in the binary image

    cX, cY = 0, 0

    for c in contours:
        area = cv2.contourArea(c)
        M = cv2.moments(c)  # calculate moments for each contour
        if 1000 < area < 50000:
            # calculate x,y coordinate of center
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

            return cX, cY, c

    if cX == 0 and cY == 0:
        return 0, 0, []


def print_parameters(mode, img, cX, cY, c, left, right, all):
    if len(c) == 0:
        cv2.putText(img, "Poor contrast object", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(img, "Estimated Centroid:", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(img, str(cX), (180, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(img, str(cY), (220, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(img, (cX, cY), 5, (255, 0, 0), -1)

    else:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        perimeter2 = format(perimeter, '.3f')

        cv2.putText(img, "Area:", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(img, str(area), (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(img, "Perimeter:", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(img, str(perimeter2), (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.drawContours(img, [c], -1, (255, 0, 0), 2)
        cv2.putText(img, "Centroid:", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(img, str(cX), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(img, str(cY), (140, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(img, (cX, cY), 5, (255, 0, 0), -1)

    if len(all) != 0:
        print_bounding_area(img, all)

    print_gripper(mode, img, left, right)

    return img


def print_gripper(mode, img, left, right):
    # left gripper
    (aX, aY) = left
    (bX, bY) = right
    length = 20
    vX = bX-aX
    vY = bY-aY

    mag = math.sqrt(vX*vX + vY*vY)
    vX = vX / mag
    vY = vY / mag
    temp = vX
    vX = 0-vY
    vY = temp

    cX = int(bX + vX * length)
    cY = int(bY + vY * length)
    dX = int(bX - vX * length)
    dY = int(bY - vY * length)

    cv2.line(img, (cX, cY), (dX, dY), (0, 255, 0), 2)

    # right gripper
    (aX, aY) = right
    (bX, bY) = left
    vX = bX-aX
    vY = bY-aY

    mag = math.sqrt(vX*vX + vY*vY)
    vX = vX / mag
    vY = vY / mag
    temp = vX
    vX = 0-vY
    vY = temp

    cX = int(bX + vX * length)
    cY = int(bY + vY * length)
    dX = int(bX - vX * length)
    dY = int(bY - vY * length)

    cv2.line(img, left, right, (0, 255, 0), 2)
    cv2.line(img, (cX, cY), (dX, dY), (0, 255, 0), 2)

    # gripper parameters
    if left[0] < right[0]:
        (aX, aY) = left
        (bX, bY) = right
    else:
        (aX, aY) = right
        (bX, bY) = left
    midX = int((aX+bX)/2)
    midY = int((aY+bY)/2)
    distance = math.sqrt((aX - bX) ** 2 + (aY - bY) ** 2)
    distance = format(distance, '.2f')
    angle = math.degrees(math.atan2((bY - aY), (bX - aX)))
    angle = format(angle, '.2f')


    cv2.putText(img, "Gripper position:", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.circle(img, (midX, midY), 5, (0, 255, 0), -1)
    cv2.putText(img, str(midX), (170, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(img, str(midY), (210, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.putText(img, "Gripper distance:", (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(img, str(distance), (170, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.putText(img, "Gripper orientation:", (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(img, str(angle), (170, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if mode == 1:
        execute_grasp(midX, midY, angle)
    if mode == 2:
        execute_pick_and_place(midX, midY, angle)


def print_bounding_area(img, all):
    if len(all) >= 3:
        obj = ConcaveHull(all, 3)
        hull = obj.calculate()

        # hull = cv2.convexHull(all)
        cv2.drawContours(img, [hull], -1, (0, 255, 255), 2)

