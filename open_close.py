import math
import URBasic
import socket
import time

def open_close():
    # setup socket for gripper function
    ROBOT_IP = '10.149.230.20'
    HOST = "10.149.230.20"
    PORT = 30002
    robotModel = URBasic.robotModel.RobotModel()
    robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP,robotModel=robotModel)
    robot.reset_error()
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect((HOST,PORT))
    soc.send(("set_digital_out(2, True)" + "\n").encode("utf8"))
    script_open = "/home/kaisherng/Documents/yolov5_obb/opengrip.script"
    fopen = open(script_open)
    soc.send((fopen.read() + "\n").encode("utf8"))
    time.sleep(5)

    script_close = "/home/kaisherng/Documents/yolov5_obb/closegrip.script"
    fclose = open(script_close)
    soc.send((fclose.read() + "\n").encode("utf8"))
    robot.close()
    soc.close()

open_close()