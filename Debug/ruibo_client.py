import socket
import numpy as np
import time
from robot_arm_control import*


if __name__ == "__main__":


    sep = 300
    epoch = 20
    
    reset_cmds = [3075,1636,1636,1249,1438,2050,2800]
    # step(reset_cmds)

    HOST = "192.168.0.186"  # The server's hostname or IP address
    PORT = 8880  # The port used by the server

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))

        reset_real = s.recv(1024)
        # print(reset_real)
        reset_real = np.frombuffer(reset_real, dtype=np.float32)
        gripper_pos = 2700
        initial_real = np.append(reset_real,gripper_pos)
        # print(initial_real)
        slow_move(initial_real[0:6],200)
        gripper_control(0)


        # after_pos = np.asarray(read_motor_pos(),dtype = np.float32)[:6]
        # print('sent')
        # s.sendall(after_pos.tobytes())

        # test = s.recv(1024)
        # print(test)
        tar_get = []
        re_al = []
        for i in range(40):
            tar_pos = s.recv(1024)
            tar_pos = np.frombuffer(tar_pos, dtype=np.float32)
            tar_get.append(tar_pos)
            print('tar',tar_pos)
            if len(tar_pos) == 6:
                # print('begin')
                slow_move(tar_pos,12)
                # print('over')
            # gripper_control()
                # print("READ: ",cur_pos)
                # print("CMDS: ",mv_cmds)
            elif len(tar_pos) == 2:

                # print('gripper', tar_pos)
                gripper_control(tar_pos[0])
                    # gripper_act(2800)
            # note the last cmd pos

            after_pos = np.asarray(read_motor_pos()[:6],dtype = np.float32)
            # print('sent')
            # time.sleep(0.1)

            #! have a try?
            # print('WTF?')
            print('a',after_pos)
            re_al.append(after_pos)
            s.sendall(after_pos.tobytes())

    # step(reset_cmds)
    slow_move(reset_cmds[0:6],80)
    gripper_act(2800)
    # time.sleep(0.2)
    clear_process()
