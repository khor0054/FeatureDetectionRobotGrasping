import math
import URBasic
import socket
import time


def execute_grasp(midX, midY, angle):
    focal = 618
    depth = 0.46
    camera_tool_diff = 0.09
    angle = float(angle)
    #angle = 90 - angle

    Xoffset = (midX - 320) * (depth/focal)
    Yoffset = (240 - midY) * (depth/focal)

    ROBOT_IP = '10.149.230.20'
    ACCELERATION = 1  # Robot acceleration value
    VELOCITY = 1  # Robot speed value
    HOST = "10.149.230.20"
    PORT = 30002

    # setup socket for gripper function
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect((HOST,PORT))
    soc.send(("set_digital_out(2, True)" + "\n").encode("utf8"))

    # initialise robot with URBasic
    print("\n>> Initialising robot")
    robotModel = URBasic.robotModel.RobotModel()
    robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP,robotModel=robotModel)
    robot.reset_error()
    print("\n>> Robot initialised")
    time.sleep(0.5)

    # open gripper
    script_open = "/home/kaisherng/Documents/yolov5_obb/opengrip.script"
    fopen = open(script_open)
    soc.send((fopen.read() + "\n").encode("utf8"))
    data2 = soc.recv(1024)
    time.sleep(0.5)


    # move to start position
    robot_startposition = [math.radians(180),
                	       math.radians(-90),
                           math.radians(90),
                           math.radians(-90),
                           math.radians(-90),
                           math.radians(0)]
    #start_pose = [0.49, 0.13, 0.3, 2.2, 2.2, 0]
    robot.movej(q=robot_startposition, a= ACCELERATION, v= VELOCITY)
    time.sleep(0.5)

    # orientate gripper
    rotation = robot_startposition
    rotation[5] = math.radians(angle)
    robot.movej(q=rotation, a= ACCELERATION, v= VELOCITY)
    print('\n>> Angle: ', angle)
    time.sleep(0.5)

    # move to designated coordinates
    grip_pose = robot.get_actual_tcp_pose()
    grip_pose[0] = grip_pose[0] + Yoffset + 0.05
    grip_pose[1] = grip_pose[1] - Xoffset + 0.05
    grip_pose[2] = grip_pose[2] - (depth - camera_tool_diff)
    robot.movej(pose=grip_pose, a= ACCELERATION, v= VELOCITY)
    print('\n>> Grip pose: ', grip_pose)
    time.sleep(0.5)

    # close gripper
    script_close = "/home/kaisherng/Documents/yolov5_obb/closegrip.script"
    fclose = open(script_close)
    soc.send((fclose.read() + "\n").encode("utf8"))
    time.sleep(1)

    #place object
    """
    robot_dropposition = [math.radians(350),
                	       math.radians(-42),
                           math.radians(50),
                           math.radians(-105),
                           math.radians(-90),
                           math.radians(0)]
    robot.movej(q=robot_dropposition, a= ACCELERATION, v= VELOCITY)
    time.sleep(0.5)

    script_open = "/home/kaisherng/Documents/yolov5_obb/opengrip.script"
    fopen = open(script_open)
    soc.send((fopen.read() + "\n").encode("utf8"))
    time.sleep(1)
"""

    # move to end pose
    robot_endposition = [math.radians(180),
                	       math.radians(-90),
                           math.radians(90),
                           math.radians(-90),
                           math.radians(-90),
                           math.radians(0)]
    robot.movej(q=robot_endposition, a= ACCELERATION, v= VELOCITY)
    print('\n>> Recovery pose: ', robot_startposition)
    time.sleep(0.5)

    robot.close()
    soc.close()

def execute_pick_and_place(midX, midY, angle):
    focal = 618
    depth = 0.46
    camera_tool_diff = 0.09
    angle = float(angle)
    #angle = 90 - angle

    Xoffset = (midX - 320) * (depth/focal)
    Yoffset = (240 - midY) * (depth/focal)

    ROBOT_IP = '10.149.230.20'
    ACCELERATION = 1  # Robot acceleration value
    VELOCITY = 1  # Robot speed value
    HOST = "10.149.230.20"
    PORT = 30002

    # setup socket for gripper function
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect((HOST,PORT))
    soc.send(("set_digital_out(2, True)" + "\n").encode("utf8"))

    # initialise robot with URBasic
    print("\n>> Initialising robot")
    robotModel = URBasic.robotModel.RobotModel()
    robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP,robotModel=robotModel)
    robot.reset_error()
    print("\n>> Robot initialised")
    time.sleep(0.5)

    # move to start position
    robot_startposition = [math.radians(180),
                	       math.radians(-90),
                           math.radians(90),
                           math.radians(-90),
                           math.radians(-90),
                           math.radians(0)]
    #start_pose = [0.49, 0.13, 0.3, 2.2, 2.2, 0]
    robot.movej(q=robot_startposition, a= ACCELERATION, v= VELOCITY)
    time.sleep(0.5)

    # orientate gripper
    rotation = robot_startposition
    rotation[5] = math.radians(angle)
    robot.movej(q=rotation, a= ACCELERATION, v= VELOCITY)
    print('\n>> Angle: ', angle)
    time.sleep(0.5)

    # move to designated coordinates
    grip_pose = robot.get_actual_tcp_pose()
    grip_pose[0] = grip_pose[0] + Yoffset + 0.02
    grip_pose[1] = grip_pose[1] - Xoffset
    grip_pose[2] = grip_pose[2] - (depth - camera_tool_diff)
    robot.movej(pose=grip_pose, a= ACCELERATION, v= VELOCITY)
    print('\n>> Grip pose: ', grip_pose)
    time.sleep(0.5)

    # close gripper
    script_close = "/home/kaisherng/Documents/yolov5_obb/closegrip.script"
    fclose = open(script_close)
    soc.send((fclose.read() + "\n").encode("utf8"))
    time.sleep(1)

    #place object
    robot_dropposition = [math.radians(350),
                	       math.radians(-42),
                           math.radians(50),
                           math.radians(-105),
                           math.radians(-90),
                           math.radians(0)]
    robot.movej(q=robot_dropposition, a= ACCELERATION, v= VELOCITY)
    time.sleep(0.5)

    script_open = "/home/kaisherng/Documents/yolov5_obb/opengrip.script"
    fopen = open(script_open)
    soc.send((fopen.read() + "\n").encode("utf8"))
    time.sleep(1)

    # move to end pose
    robot_endposition = [math.radians(180),
                	       math.radians(-90),
                           math.radians(90),
                           math.radians(-90),
                           math.radians(-90),
                           math.radians(0)]
    robot.movej(q=robot_endposition, a= ACCELERATION, v= VELOCITY)
    print('\n>> Recovery pose: ', robot_startposition)
    time.sleep(0.5)

    robot.close()
    soc.close()

def initialize_open():
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
    robot.close()
    soc.close()