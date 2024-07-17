from realsense_capture import capture
from grasp import initialize_open, execute_grasp

initialize_open()
capture()

from file_transfer import upload_and_receive


upload_and_receive()


with open('/home/kaisherng/Documents/yolov5_obb/parameters.txt') as f:
    for line in f:
        x, y, angle = line.split(' ')
x = float(x)
y = float(y)
angle = float(angle)

execute_grasp(x, y, angle)