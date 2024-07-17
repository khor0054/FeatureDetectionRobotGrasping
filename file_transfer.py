import time
import paramiko
import time
from scp import SCPClient

def upload_and_receive():
    hostname = '10.97.24.188'
    username = 'khor0054'
    password = 'kaisherng'

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, username=username, password=password)
    print("\n>> Connected to ssh.")

    local_file = '/home/kaisherng/Documents/yolov5_obb/image_captures/capture.png'
    destination_file = '/home/khor0054/PycharmProjects/yolov5_obb/upload/capture.png'

    with SCPClient(ssh.get_transport()) as scp:
        scp.put(local_file, destination_file)

    print("\n>> Picture upload to ssh successful.")

    sftp = ssh.open_sftp()
    path = '/home/khor0054/PycharmProjects/yolov5_obb/upload/parameters.txt'

    while sftp_exists(sftp, path) == 0:
        print("Waiting...")
        time.sleep(1)

    remote_file = '/home/khor0054/PycharmProjects/yolov5_obb/upload/parameters.txt'
    local_destination = '/home/kaisherng/Documents/yolov5_obb/parameters.txt'

    with SCPClient(ssh.get_transport()) as scp:
        scp.get(remote_file, local_destination)


    print("\n>> Parameters received.")

    ssh.close()

def sftp_exists(sftp, path):
    try:
        sftp.stat(path)
        return 1
    except FileNotFoundError:
        return 0


upload_and_receive()