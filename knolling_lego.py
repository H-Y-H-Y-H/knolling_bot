from tracemalloc import start
from turtle import position
from items import sort
import pybullet as p
import pybullet_data as pd
import os
import numpy as np
import random
random.seed(223)
np.random.seed(2022)
import math
import cv2
import logging
from easy_logx.easy_logx import EasyLog
import os
import numpy as np
import time
from turdf import *
import socket

logger = EasyLog(log_level=logging.INFO)

class Arm_env():
    
    def __init__(self, is_render=True, num_2x2=0, num_2x3=0, num_2x4=0, num_pencil=0, order_flag='center', kinds=4, grasp_order=None):

        self.kImageSize = {'width': 480, 'height': 480}
        self.urdf_path = 'urdf'
        self.pybullet_path = pd.getDataPath()
        self.is_render = is_render

        self.ik_low = [-1.57, -1.57, -1.57, -1.57, -1.57, -10, -10]
        self.ik_high = [1.57, 1.57, 1.57, 1.57, 1.57, 10, 10]
        self.num_motor = 5

        self.low_scale = np.array([0.05, -0.15, 0.006, - np.pi / 2, 0])
        self.high_scale = np.array([0.3, 0.15, 0.05, np.pi / 2, 0.4])
        self.low_act = -np.ones(5)
        self.high_act = np.ones(5)
        self.x_low_obs = self.low_scale[0]
        self.x_high_obs = self.high_scale[0]
        self.y_low_obs = self.low_scale[1]
        self.y_high_obs = self.high_scale[1]
        self.z_low_obs = self.low_scale[2]
        self.z_high_obs = self.high_scale[2]

        self.table_boundary = 0.05

        self.lateral_friction = 1
        self.spinning_friction = 1
        self.rolling_friction = 0
        self.num_2x2 = num_2x2
        self.num_2x3 = num_2x3
        self.num_2x4 = num_2x4
        self.num_pencil = num_pencil
        self.order_flag = order_flag
        self.kinds = kinds
        self.grasp_order = grasp_order
        # 5 6 9不用管，固定的！

        #! the parameters of reorder
        self.gap_item = 0.03
        self.gap_block = 0.02

        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.camera_parameters = {
            'width': 640.,
            'height': 480,
            'fov': 42,
            'near': 0.1,
            'far': 100.,
            'eye_position': [0.59, 0, 0.8],
            'target_position': [0.55, 0, 0.05],
            'camera_up_vector':
                [1, 0, 0],  # I really do not know the parameter's effect.
            'light_direction': [
                0.5, 0, 1
            ],  # the direction is from the light source position to the origin of the world frame.
        }
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.15, 0, 0],
            distance=0.4,
            yaw=90,
            pitch=-90,
            roll=0,
            upAxisIndex=2)
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_parameters['fov'],
            aspect=self.camera_parameters['width'] /
                   self.camera_parameters['height'],
            nearVal=self.camera_parameters['near'],
            farVal=self.camera_parameters['far'])

        p.configureDebugVisualizer(lightPosition=[5, 0, 5])
        p.resetDebugVisualizerCamera(cameraDistance=0.7,
                                     cameraYaw=45,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[0.1, 0, 0.4])
        p.setAdditionalSearchPath(pd.getDataPath())

    def get_image(self, gray_flag=False, resize_flag=False):
        # reset camera
        (width, length, image, _, _) = p.getCameraImage(width=640,
                                                        height=480,
                                                        viewMatrix=self.view_matrix,
                                                        projectionMatrix=self.projection_matrix,
                                                        renderer=p.ER_BULLET_HARDWARE_OPENGL)
        if gray_flag:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if resize_flag:
            image = cv2.resize(image, (self.kImageSize['width'], self.kImageSize['height']))[None, :, :] / 255.
            return image[0]

        return image

    def reset(self, random_flag):
        

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # Draw workspace lines
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs])

        baseid = p.loadURDF(os.path.join(self.urdf_path, "base.urdf"), basePosition=[0, 0, -0.05], useFixedBase=1,
                            flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        self.arm_id = p.loadURDF(os.path.join(self.urdf_path, "robot_arm928/robot_arm1.urdf"),
                                 basePosition=[-0.08, 0, 0.02], useFixedBase=True,
                                 flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        textureId = p.loadTexture(os.path.join(self.urdf_path, "table/table.png"))
        p.changeDynamics(baseid, -1)
        p.changeDynamics(self.arm_id, 7, 
                        lateralFriction=self.lateral_friction, 
                        spinningFriction=self.spinning_friction, 
                        rollingFriction=self.rolling_friction, 
                        # restitution=0.9,
                        # linearDamping=0, 
                        # angularDamping=0, 
                        frictionAnchor=True)
        p.changeDynamics(self.arm_id, 8, 
                        lateralFriction=self.lateral_friction, 
                        spinningFriction=self.spinning_friction, 
                        rollingFriction=self.rolling_friction, 
                        # restitution=0.9,
                        linearDamping=0, 
                        angularDamping=0, 
                        frictionAnchor=True)
        p.changeVisualShape(baseid, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=textureId)

        #! there are pos and ori before processing.
        items_sort = sort(self.num_2x2, self.num_2x3, self.num_2x4, self.num_pencil)

        if random_flag == False: #! use the pos and ori from real world

            self.xyz_list, self.pos_list, self.ori_list, self.all_index = items_sort.get_data(self.kinds)

        elif random_flag == True: #! use the pos and ori from the simulator

            self.xyz_list, _, _, self.all_index = items_sort.get_data(self.kinds)
            self.pos_list = []
            self.ori_list = []

            restrict = np.max(self.xyz_list)
            num_list = np.array([self.num_2x2, self.num_2x3, self.num_2x4, self.num_pencil])
            # self.all_index = np.asarray(self.all_index)

            self.obj_idx = []
            last_pos = np.array([[0, 0, 1]])
            for i in range(len(num_list)):

                for j in range(num_list[i]):

                    rdm_pos = np.array([random.uniform(self.x_low_obs, self.x_high_obs), random.uniform(self.y_low_obs, self.y_high_obs), 0.006])
                    ori = [0, 0, random.uniform(-math.pi / 2, math.pi / 2)]
                    
                    check_list = np.zeros(last_pos.shape[0])

                    while 0 in check_list:
                        rdm_pos = [random.uniform(self.x_low_obs, self.x_high_obs), random.uniform(self.y_low_obs, self.y_high_obs), 0.006]
                        for z in range(last_pos.shape[0]):
                            if np.linalg.norm(last_pos[z] - rdm_pos) < restrict:
                                check_list[z] = 0
                            else:
                                check_list[z] = 1

                    last_pos = np.append(last_pos, [rdm_pos], axis= 0)

                    self.obj_idx.append(p.loadURDF(os.path.join(self.urdf_path, f"item_{i}/{j}.urdf"), basePosition=rdm_pos,
                                            baseOrientation=p.getQuaternionFromEuler(ori), useFixedBase=False,
                                            flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
                    
                    self.pos_list.append(rdm_pos)
                    self.ori_list.append(ori)
                    
            self.pos_list = np.asarray(self.pos_list)
            self.ori_list = np.asarray(self.ori_list)
            print(self.obj_idx)
            for i in range(len(self.obj_idx)):
                p.changeDynamics(self.obj_idx[i], -1, 
                                lateralFriction=self.lateral_friction, 
                                spinningFriction=self.spinning_friction,
                                rollingFriction=self.rolling_friction, 
                                # restitution=0.5,
                                # linearDamping=0, 
                                # angularDamping=0, 
                                frictionAnchor=True)

        #! set the initial pos of the arm
        ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=[0.05, 0, 0.066],
                                                maxNumIterations=200,
                                                targetOrientation=p.getQuaternionFromEuler([0, math.pi/2, 0]))
        for motor_index in range(self.num_motor):
            p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL, targetPosition = ik_angles0[motor_index], maxVelocity=7)
        for i in range(60):
            p.stepSimulation()
        # self.images = self.get_image()

        return self.images

    def change_config(self): #! this is main function!!!!!!!!!

        p.resetSimulation()
        p.setGravity(0, 0, -10)

        # Draw workspace lines
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs])

        baseid = p.loadURDF(os.path.join(self.urdf_path, "base.urdf"), basePosition=[0, 0, -0.05], useFixedBase=1,
                            flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        self.arm_id = p.loadURDF(os.path.join(self.urdf_path, "robot_arm928/robot_arm1.urdf"),
                                 basePosition=[-0.08, 0, 0.02], useFixedBase=True,
                                 flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        textureId = p.loadTexture(os.path.join(self.urdf_path, "table/table.png"))
        p.changeDynamics(baseid, -1, lateralFriction=self.lateral_friction, frictionAnchor=True)
        p.changeDynamics(self.arm_id, 7, lateralFriction=self.lateral_friction, frictionAnchor=True)
        p.changeDynamics(self.arm_id, 8, lateralFriction=self.lateral_friction, frictionAnchor=True)
        p.changeVisualShape(baseid, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=textureId)

        items_sort = sort(self.num_2x2, self.num_2x3, self.num_2x4, self.num_pencil)
        self.xyz_list, _, _, self.all_index = items_sort.get_data(self.kinds)
        print(f'this is trim xyz list\n {self.xyz_list}')
        print(f'this is trim index list\n {self.all_index}')

        #! the tidy configuration
        self.items_pos_list, self.items_ori_list = self.reorder_items()
        # print(self.items_pos_list) 
        
        print(self.items_ori_list)

        if self.order_flag == 'center':

            bias_pos = [0.1, 0, 0.006]
            self.items_pos_list = self.items_pos_list + bias_pos
            # none_list = np.array([[0]])
            num_list = np.array([self.num_2x2, self.num_2x3, self.num_2x4, self.num_pencil])
            # self.all_index = np.asarray(self.all_index)

            items_names = globals()
            self.obj_idx = []
            for i in range(len(num_list)):
                if num_list[i] == 0:
                    # self.all_index = np.insert(self.all_index, i, values=none_list, axis=0)
                    continue
                items_names[f'index_{i}'] = self.all_index[i]
                items_names[f'num_{i}'] = len(items_names[f'index_{i}'])
                items_names[f'pos_{i}'] = self.items_pos_list[items_names[f'index_{i}'], :]
                items_names[f'ori_{i}'] = self.items_ori_list[items_names[f'index_{i}'], :]
                # print(items_names[f'pos_{i}'])
                for j in range(num_list[i]):
                    self.obj_idx.append(p.loadURDF(os.path.join(self.urdf_path, f"item_{i}/{j}.urdf"), basePosition=items_names[f'pos_{i}'][j],
                                            baseOrientation=p.getQuaternionFromEuler(items_names[f'ori_{i}'][j]), useFixedBase=True,
                                            flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))

        self.images = self.get_image()
        # while 1:
        #     p.stepSimulation()

        return self.images

    def manipulator_operation_actual(self):
    
        print(f'*********************************')
        print(f'this is the input pos before calculation\n {self.pos_list}')
        print(f'this is the input ori before calculation\n {self.ori_list}')
        print(f'this is the input pos after calculation\n {self.items_pos_list}')
        print(f'this is the input ori after calculation\n {self.items_ori_list}')
        print(f'this is the index of all items based on the order defined above\n {self.all_index}')
        print(f'*********************************')

        manipulator_pos_before = []
        manipulator_ori_before = []
        manipulator_pos_after = []
        manipulator_ori_after = []

        for i in range(len(self.all_index)):
            for j in range(len(self.all_index[i])):
                
                # manipulator_pos_before.append(self.pos_list[self.all_index[i][j]])
                # manipulator_ori_before.append(self.ori_list[self.all_index[i][j]])
                manipulator_pos_after.append(self.items_pos_list[self.all_index[i][j]])
                manipulator_ori_after.append(self.items_ori_list[self.all_index[i][j]])

        # manipulator_pos_before = np.asarray(manipulator_pos_before)
        # manipulator_ori_before = np.asarray(manipulator_ori_before)
        manipulator_pos_after = np.asarray(manipulator_pos_after)
        manipulator_ori_after = np.asarray(manipulator_ori_after)

        print(f'*********************************')
        print(f'this is the pos before operation\n {manipulator_pos_before}')
        print(f'this is the ori before operation\n {manipulator_ori_before}')
        print(f'this is the pos after operation\n {manipulator_pos_after}')
        print(f'this is the ori after operation\n {manipulator_ori_after}')
        print(f'this is the index of all items based on the order defined above\n {self.all_index}')
        print(f'*********************************')

        HOST = "192.168.0.186"  # Standard loopback interface address (localhost)
        PORT = 8880  # Port to listen on (non-privileged ports are > 1023)
        real_pos_list = []
        Save_real_pos = False

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            # It should be an integer from 1 to 65535, as 0 is reserved. Some systems may require superuser privileges if the port number is less than 1024.
            # associate the socket with a specific network interface
            s.listen()
            print(f"Waiting for connection...\n")
            conn, addr = s.accept()
            print(conn)

            # The with statement is used with conn to automatically close the socket at the end of the block.
            with conn:
                print(f"Connected by {addr}")

                for i in range(len(self.pos_list)):

                    cmds_list = []
                    rest_pos = np.array([0, 0, 0.05])
                    rest_ori = np.array([0, 1.57, 0])
                    offset_low = np.array([0, 0, 0.007])
                    offset_high = np.array([0, 0, 0.04])

                    ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=rest_pos,
                                                            maxNumIterations=200,
                                                            targetOrientation=p.getQuaternionFromEuler(rest_ori))
                    cmds_list.append(rad2cmd(ik_angles0[0:5]))

                    ik_angles1 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=offset_high+manipulator_pos_before,
                                                            maxNumIterations=200,
                                                            targetOrientation=p.getQuaternionFromEuler(rest_ori+manipulator_ori_before))
                    cmds_list.append(rad2cmd(ik_angles1[0:5]))

                    ik_angles2 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=offset_low+manipulator_pos_before,
                                                            maxNumIterations=200,
                                                            targetOrientation=p.getQuaternionFromEuler(rest_ori+manipulator_ori_before))
                    cmds_list.append(rad2cmd(ik_angles2[0:5]))

                    cmds_list.append([1, 1])

                    ik_angles3 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=offset_high+manipulator_pos_before,
                                                            maxNumIterations=200,
                                                            targetOrientation=p.getQuaternionFromEuler(rest_ori+manipulator_ori_before))
                    cmds_list.append(rad2cmd(ik_angles3[0:5]))

                    ik_angles4 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=offset_high+manipulator_pos_after,
                                                            maxNumIterations=200,
                                                            targetOrientation=p.getQuaternionFromEuler(rest_ori+manipulator_ori_after))

                    cmds_list.append(rad2cmd(ik_angles4[0:5]))

                    ik_angles5 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=offset_low+manipulator_pos_after,
                                                            maxNumIterations=200,
                                                            targetOrientation=p.getQuaternionFromEuler(rest_ori+manipulator_ori_after))
                    cmds_list.append(rad2cmd(ik_angles5[0:5]))

                    cmds_list.append([0, 0])

                    ik_angles6 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=offset_high+manipulator_pos_after,
                                                            maxNumIterations=200,
                                                            targetOrientation=p.getQuaternionFromEuler(rest_ori+manipulator_ori_after))
                    cmds_list.append(rad2cmd(ik_angles6[0:5]))

                    ik_angles7 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=rest_pos,
                                                            maxNumIterations=200,
                                                            targetOrientation=p.getQuaternionFromEuler(rest_ori))
                    cmds_list.append(rad2cmd(ik_angles7[0:5]))

                    for j in range(len(cmds_list)):
                        if len(cmds_list[j]) == 5:

                            print(f'this is the assigned cmds {cmds_list[j]}')
                            pos_sim = sim_cmd2tarpos(cmds_list[j])
                            pos_real = real_cmd2tarpos(cmds_list[j])
                            pos_real = np.asarray(pos_real, dtype=np.float32)
                            print(f'this is the cmd pos {pos_real}')
                            conn.sendall(pos_real.tobytes())
                            # time.sleep(5)

                            p.setJointMotorControl2(self.arm_id, 0, p.POSITION_CONTROL, targetPosition = pos_sim[0], maxVelocity= 3)
                            p.setJointMotorControl2(self.arm_id, 1, p.POSITION_CONTROL, targetPosition=pos_sim[1], maxVelocity=3)
                            p.setJointMotorControl2(self.arm_id, 2, p.POSITION_CONTROL, targetPosition=pos_sim[2], maxVelocity=3)
                            p.setJointMotorControl2(self.arm_id, 3, p.POSITION_CONTROL, targetPosition=pos_sim[3], maxVelocity=3)
                            p.setJointMotorControl2(self.arm_id, 4, p.POSITION_CONTROL, targetPosition=pos_sim[4], maxVelocity=3)
                        
                        elif len(cmds_list[j]) == 2:
                            pos_real = np.asarray(cmds_list[j], dtype=np.float32)
                            conn.sendall(pos_real.tobytes())
                            # print(f'this is the cmd pos {pos_real}')
                            #gripper
                            p.setJointMotorControlArray(self.arm_id, [7,8], p.POSITION_CONTROL, targetPositions=[cmds_list[j][0], cmds_list[j][1]])

                        for _ in range(200):
                            p.stepSimulation()
                            # time.sleep(1/48)

                        real_pos = conn.recv(1024)
                        real_pos = np.frombuffer(real_pos, dtype=np.float32)
                        print(f'this is real pos {real_pos}')

                        if Save_real_pos == True:
                            real_pos_list.append(real_pos)

    def check_desk(self, manipulator_pos_before, manipulator_ori_before, manipulator_pos_after, manipulator_ori_after):

        clean_flag = False
        restrict = np.max(self.xyz_list)
        gripper_width = 0.03
        barricade_pos = []
        barricade_index = []

        for i in range(len(manipulator_pos_before)):
            for j in range(len(manipulator_pos_after)):
                if np.linalg.norm(manipulator_pos_after[j] - manipulator_pos_before[i]) < restrict:
                    if i not in barricade_index:
                        print('We need to clean the desktop to provide an enough space')
                        barricade_pos.append(manipulator_pos_before[i])
                        barricade_index.append(i)
                    clean_flag = True
        barricade_pos = np.asarray(barricade_pos)
        print('this is the barricade pos', barricade_pos)

        x_high = np.max(manipulator_pos_after[:, 0])
        x_low = np.min(manipulator_pos_after[:, 0])
        y_high = np.max(manipulator_pos_after[:, 1])
        y_low = np.min(manipulator_pos_after[:, 1])
        p.addUserDebugLine(lineFromXYZ=[x_low, y_low, 0.006], lineToXYZ=[x_high, y_low, 0.006])
        p.addUserDebugLine(lineFromXYZ=[x_low, y_low, 0.006], lineToXYZ=[x_low, y_high, 0.006])
        p.addUserDebugLine(lineFromXYZ=[x_high, y_high, 0.006], lineToXYZ=[x_high, y_low, 0.006])
        p.addUserDebugLine(lineFromXYZ=[x_high, y_high, 0.006], lineToXYZ=[x_low, y_high, 0.006])

        if clean_flag == True:
            
            #! pos
            offset_low = np.array([0, 0, -0.004])
            offset_high = np.array([0, 0, 0.06])
            #! ori
            rest_ori = np.array([0, 1.57, 0])
            offset_right = np.array([0, 0, math.pi/4])
            offset_left = np.array([0, 0, -math.pi/4])
            #! direction
            if y_high - y_low > x_high - x_low:
                offset_horizontal = np.array([np.max(self.xyz_list) - 0.01, 0, 0])
                offset_rectangle = np.array([0, 0, math.pi/2])
                direction = 'x'
            else:
                offset_horizontal = np.array([0, np.max(self.xyz_list) - 0.01, 0])
                offset_rectangle = np.array([0, 0, 0])
                direction = 'y'

            trajectory_pos_list = []
            trajectory_ori_list = []
            for i in range(len(barricade_index)):
                if direction == 'x':
                    terminate = np.array([x_high, barricade_pos[i][1], barricade_pos[i][2]])
                elif direction == 'y':
                    terminate = np.array([barricade_pos[i][0], y_high, barricade_pos[i][2]])
                trajectory_pos_list.append(np.array([0.032, 0.032]))
                trajectory_ori_list.append(rest_ori + offset_rectangle)
                trajectory_pos_list.append(barricade_pos[i] + offset_high - offset_horizontal)
                trajectory_ori_list.append(rest_ori + offset_rectangle)
                trajectory_pos_list.append(barricade_pos[i] + offset_low - offset_horizontal)
                trajectory_ori_list.append(rest_ori + offset_rectangle)
                trajectory_pos_list.append(offset_low + offset_horizontal + terminate)
                trajectory_ori_list.append(rest_ori + offset_rectangle)
                trajectory_pos_list.append(offset_high + offset_horizontal + terminate)
                trajectory_ori_list.append(rest_ori + offset_rectangle)
                trajectory_pos_list.append(np.array([0.025, 0.025]))
                trajectory_ori_list.append(rest_ori + offset_rectangle)

            last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
            last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))
            new_pos_before = []
            new_ori_before = []
            for j in range(len(trajectory_pos_list)):

                if len(trajectory_pos_list[j]) == 3:
                    if trajectory_pos_list[j][0] == last_pos[0] and trajectory_pos_list[j][1] == last_pos[1]:
                        self.trajectory_vertical(last_pos, trajectory_pos_list[j], trajectory_ori_list[j])
                    else:
                        # print('test', trajectory_pos_list[j])
                        self.trajectory_horizontal(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j])
                    last_pos = np.copy(trajectory_pos_list[j])
                    last_ori = np.copy(trajectory_ori_list[j])

                elif len(trajectory_pos_list[j]) == 2:
                    self.gripper_control(trajectory_pos_list[j])
            
            for i in range(len(manipulator_pos_before)):
                new_pos_before.append(p.getBasePositionAndOrientation(self.obj_idx[i])[0])
                new_ori_before.append(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.obj_idx[i])[1]))
            
            manipulator_pos_before = []
            manipulator_ori_before = []
            for i in self.grasp_order:
                for j in range(len(self.all_index[i])):
                    manipulator_pos_before.append(new_pos_before[self.all_index[i][j]])
                    manipulator_ori_before.append(new_ori_before[self.all_index[i][j]])
            manipulator_pos_before = np.asarray(manipulator_pos_before)
            manipulator_ori_before = np.asarray(manipulator_ori_before)
        else:
            pass

        return manipulator_pos_before, manipulator_ori_before

    def check_crowded(self, manipulator_pos_before, manipulator_ori_before, manipulator_pos_after, manipulator_ori_after, new_xyz_list):

        clean_flag = False
        restrict = np.max(self.xyz_list) - 0.01
        gripper_width = 0.033
        gripper_height = 0.023
        crowded_pos = []
        crowded_ori = []
        crowded_index = []

        for i in range(len(manipulator_pos_before)):
            for j in range(i+1, len(manipulator_pos_before)):
                if np.linalg.norm(manipulator_pos_before[j] - manipulator_pos_before[i]) < restrict:
                    if i not in crowded_index:
                        print('We need to clean the items surrounding it to provide an enough space')
                        crowded_pos.append(manipulator_pos_before[i])
                        crowded_ori.append(manipulator_ori_before[i])
                        crowded_index.append(i)
                    clean_flag = True
        crowded_pos = np.asarray(crowded_pos)
        print(crowded_pos)

        if clean_flag == True:
            
            #! pos
            offset_low = np.array([0, 0, -0.004])
            offset_high = np.array([0, 0, 0.06])
            #! ori
            rest_ori = np.array([0, 1.57, 0])
            # offset_right = np.array([0, 0, math.pi/4])
            # offset_left = np.array([0, 0, -math.pi/4])

            trajectory_pos_list = []
            trajectory_ori_list = []
            for i in range(len(crowded_pos)):

                print(new_xyz_list[crowded_index[i]])
                x = new_xyz_list[crowded_index[i]][0] / 2
                y = new_xyz_list[crowded_index[i]][1] / 2
                print(manipulator_ori_before[crowded_index[i]])
                theta = manipulator_ori_before[crowded_index[i]]
                vertex_1 = np.array([(x*math.cos(theta)) - (y*math.sin(theta)), (x*math.sin(theta)) + (y*math.cos(theta)), 0])
                vertex_2 = np.array([(-x*math.cos(theta)) - (y*math.sin(theta)), (-x*math.sin(theta)) + (y*math.cos(theta)), 0])
                vertex_3 = np.array([(-x*math.cos(theta)) - (-y*math.sin(theta)), (-x*math.sin(theta)) + (-y*math.cos(theta)), 0])
                vertex_4 = np.array([(x*math.cos(theta)) - (-y*math.sin(theta)), (x*math.sin(theta)) + (-y*math.cos(theta)), 0])
                point_1 = vertex_1 + np.array([(gripper_width*math.cos(theta)) - (gripper_height*math.sin(theta)), (gripper_width*math.sin(theta)) + (gripper_height*math.cos(theta)), 0])
                point_2 = vertex_2 + np.array([(-gripper_width*math.cos(theta)) - (gripper_height*math.sin(theta)), (-gripper_width*math.sin(theta)) + (gripper_height*math.cos(theta)), 0])
                point_3 = vertex_3 + np.array([(-gripper_width*math.cos(theta)) - (-gripper_height*math.sin(theta)), (-gripper_width*math.sin(theta)) + (-gripper_height*math.cos(theta)), 0])
                point_4 = vertex_4 + np.array([(gripper_width*math.cos(theta)) - (-gripper_height*math.sin(theta)), (gripper_width*math.sin(theta)) + (-gripper_height*math.cos(theta)), 0])

                trajectory_pos_list.append(np.array([0.032, 0.032]))
                trajectory_ori_list.append(rest_ori)
                trajectory_pos_list.append(crowded_pos[i] + offset_high + point_1)
                trajectory_ori_list.append(rest_ori + crowded_ori[i])
                trajectory_pos_list.append(crowded_pos[i] + offset_low + point_1)
                trajectory_ori_list.append(rest_ori + crowded_ori[i])
                trajectory_pos_list.append(crowded_pos[i] + offset_low + point_2)
                trajectory_ori_list.append(rest_ori + crowded_ori[i])
                trajectory_pos_list.append(crowded_pos[i] + offset_low + point_3)
                trajectory_ori_list.append(rest_ori + crowded_ori[i])
                trajectory_pos_list.append(crowded_pos[i] + offset_low + point_4)
                trajectory_ori_list.append(rest_ori + crowded_ori[i])
                trajectory_pos_list.append(crowded_pos[i] + offset_high + point_4)
                trajectory_ori_list.append(rest_ori + crowded_ori[i])
                trajectory_pos_list.append(np.array([0.025, 0.025]))
                trajectory_ori_list.append(rest_ori)

            last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
            last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))
            new_pos_before = []
            new_ori_before = []
            for j in range(len(trajectory_pos_list)):

                if len(trajectory_pos_list[j]) == 3:
                    if trajectory_pos_list[j][0] == last_pos[0] and trajectory_pos_list[j][1] == last_pos[1]:
                        self.trajectory_vertical(last_pos, trajectory_pos_list[j], trajectory_ori_list[j])
                    else:
                        # print('test', trajectory_pos_list[j])
                        self.trajectory_horizontal(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j])
                    last_pos = np.copy(trajectory_pos_list[j])
                    last_ori = np.copy(trajectory_ori_list[j])

                elif len(trajectory_pos_list[j]) == 2:
                    if trajectory_pos_list[j][0] == 1:
                        self.gripper_control()
            
            for i in range(len(manipulator_pos_before)):
                new_pos_before.append(p.getBasePositionAndOrientation(self.obj_idx[i])[0])
                new_ori_before.append(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.obj_idx[i])[1]))
            
            manipulator_pos_before = []
            manipulator_ori_before = []
            for i in self.grasp_order:
                for j in range(len(self.all_index[i])):
                    manipulator_pos_before.append(new_pos_before[self.all_index[i][j]])
                    manipulator_ori_before.append(new_ori_before[self.all_index[i][j]])
            manipulator_pos_before = np.asarray(manipulator_pos_before)
            manipulator_ori_before = np.asarray(manipulator_ori_before)
        else:
            pass

        return manipulator_pos_before, manipulator_ori_before

    def gripper_control(self, tar):

        tar = np.array(tar)
        offset = tar - np.asarray(p.getJointStates(self.arm_id, [7, 8]), dtype=object)[:, 0]
        print('this is offset', offset)
        if offset[0] > 0:
            velocity = [0.01, 0.01]
            step = [0.032 / 60, 0.032 / 60]
        elif offset[0] < 0:
            velocity = [-0.0005, -0.0005]
            step = [-0.032 / 50, -0.032 / 50]

        while True:
            cur_pos = np.asarray(p.getJointStates(self.arm_id, [7, 8]), dtype=object)[:, 0]
            # print(cur_pos)
            tar_pos = np.add(cur_pos, step)

            for i in [7, 8]:
                p.setJointMotorControl2(self.arm_id, i, p.POSITION_CONTROL, targetPosition=tar_pos[0], force=1000)
            # p.setJointMotorControlArray(self.arm_id, [7, 8], p.POSITION_CONTROL, targetPositions=tar_pos, forces=[1000, 1000])
            # p.setJointMotorControlArray(self.arm_id, [7, 8], p.VELOCITY_CONTROL, targetVelocities=velocity, forces=[1, 1])

            for i in range(20):
                p.stepSimulation()
                # time.sleep(1/960)

            obs_pos = np.asarray(p.getJointStates(self.arm_id, [7, 8]), dtype=object)[:, 0]
            print('this is obs', obs_pos)
            if abs(obs_pos[0] - obs_pos[1]) > 0.00008:
                print('there is a gap')
                for i in [7, 8]:
                    p.setJointMotorControl2(self.arm_id, i, p.POSITION_CONTROL, targetPosition=obs_pos[0], force=10)
                for i in range(20):
                    p.stepSimulation()
                obs_pos = np.asarray(p.getJointStates(self.arm_id, [7, 8]), dtype=object)[:, 0]
                print('this is the fixed obs', obs_pos)
                # time.sleep(1/480)

            if abs(tar[0] - obs_pos[0]) <= 0.001 or abs(tar[1] - obs_pos[1]) <= 0.001:
                print('this is end obs', obs_pos)
                print('over1, reach')
                break
            # elif obs_pos[0] > 0.02915 and obs_pos[1] > 0.02915:
            #     print('this is end obs', obs_pos)
            elif abs(obs_pos[1] - tar_pos[1]) > 0.0004 or abs(obs_pos[0] - tar_pos[0]) > 0.0004:
                print('this is end obs', obs_pos)
                print('over2, get')
                break
        
        
    def trajectory_vertical(self, cur_pos, tar_pos, tar_ori):

        target_pos = np.copy(tar_pos)
        step = (tar_pos - cur_pos) / 25

        while True:
            tar_pos = cur_pos + step
            ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=tar_pos,
                                                    maxNumIterations=200,
                                                    targetOrientation=p.getQuaternionFromEuler(tar_ori))
            for motor_index in range(self.num_motor):
                p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL, targetPosition = ik_angles0[motor_index], maxVelocity=3)

            for i in range(20):
                p.stepSimulation()
                # time.sleep(1/480)

            if abs(target_pos[2] - tar_pos[2]) < 0.0001:
                break
            cur_pos = tar_pos

    def trajectory_horizontal(self, cur_pos, cur_ori, tar_pos, tar_ori):

        target_pos = np.copy(tar_pos)
        target_ori = np.copy(tar_ori)
        step_pos = (tar_pos - cur_pos) / 50
        step_ori = (tar_ori - cur_ori) / 50

        while True:
            tar_pos = cur_pos + step_pos
            tar_ori = cur_ori + step_ori
            ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=tar_pos,
                                                    maxNumIterations=200,
                                                    targetOrientation=p.getQuaternionFromEuler(tar_ori))
            for motor_index in range(self.num_motor):
                p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL, targetPosition = ik_angles0[motor_index], maxVelocity=3)

            for i in range(20):
                p.stepSimulation()
                time.sleep(1/480)
            if abs(target_pos[0] - tar_pos[0]) < 0.001 and abs(target_pos[1] - tar_pos[1]) < 0.001 and abs(target_ori[2] - tar_ori[2]) < 0.001:
                break
            cur_pos = tar_pos
            cur_ori = tar_ori

    def manipulator_operation_virtual(self):

        manipulator_pos_before = []
        manipulator_ori_before = []
        manipulator_pos_after = []
        manipulator_ori_after = []
        new_pos_before = []
        new_ori_before = []
        new_xyz_list = []

        # print(f'*********************************')
        # print(f'this is the input pos before calculation\n {self.pos_list}')
        # print(f'this is the input ori before calculation\n {self.ori_list}')
        # print(f'this is the input pos after calculation\n {self.items_pos_list}')
        # print(f'this is the input ori after calculation\n {self.items_ori_list}')
        # print(f'this is the index of all items based on the order defined above\n {self.all_index}')
        # print(f'*********************************')

        # for i in range(len(self.all_index)):
        #     for j in range(len(self.all_index[i])):
        #         manipulator_pos_after.append(self.items_pos_list[self.all_index[i][j]])
        #         manipulator_ori_after.append(self.items_ori_list[self.all_index[i][j]])

        # manipulator_pos_after = self.items_pos_list
        # manipulator_ori_after = self.items_ori_list

        # for i in range(len(self.all_index)):
        #     for j in range(len(self.all_index[i])):
        #         manipulator_pos_before.append(self.pos_list[self.all_index[i][j]])
        #         manipulator_ori_before.append(self.ori_list[self.all_index[i][j]])
        # print(np.asarray(manipulator_pos_before1))

        for i in range(len(self.obj_idx)):
            new_pos_before.append(p.getBasePositionAndOrientation(self.obj_idx[i])[0])
            new_ori_before.append(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.obj_idx[i])[1]))
        print(np.asarray(new_pos_before))
        # for i in range(len(self.all_index)):

        #! change the order of grasping (2x3,2x2,2x4)
        for i in self.grasp_order:
            for j in range(len(self.all_index[i])):
                manipulator_pos_before.append(new_pos_before[self.all_index[i][j]])
                manipulator_ori_before.append(new_ori_before[self.all_index[i][j]])
                manipulator_pos_after.append(self.items_pos_list[self.all_index[i][j]])
                manipulator_ori_after.append(self.items_ori_list[self.all_index[i][j]])
                new_xyz_list.append(self.xyz_list[self.all_index[i][j]])
        print(np.asarray(manipulator_pos_before))
        manipulator_pos_before = np.asarray(manipulator_pos_before)
        manipulator_ori_before = np.asarray(manipulator_ori_before)
        manipulator_pos_after = np.asarray(manipulator_pos_after)
        manipulator_ori_after = np.asarray(manipulator_ori_after)
        new_xyz_list = np.asarray(new_xyz_list)
        print(new_xyz_list)

        manipulator_pos_before, manipulator_ori_before = self.check_desk(manipulator_pos_before, manipulator_ori_before, manipulator_pos_after, manipulator_ori_after)
        # manipulator_pos_before, manipulator_ori_before = self.check_crowded(manipulator_pos_before, manipulator_ori_before, manipulator_pos_after, manipulator_ori_after, new_xyz_list)

        print(f'*********************************')
        print(f'this is the pos before operation\n {manipulator_pos_before}')
        print(f'this is the ori before operation\n {manipulator_ori_before}')
        print(f'this is the pos after operation\n {manipulator_pos_after}')
        print(f'this is the ori after operation\n {manipulator_ori_after}')
        print(f'this is the index of all items based on the order defined above\n {self.all_index}')
        print(f'*********************************')

        rest_pos = np.array([0, 0, 0.05])
        rest_ori = np.array([0, 1.57, 0])
        offset_low = np.array([0, 0, -0.008])
        offset_high = np.array([0, 0, 0.06])

        for i in range(len(self.obj_idx)):

            trajectory_pos_list = [offset_high+manipulator_pos_before[i],
                                    offset_low+manipulator_pos_before[i],
                                    [0.032, 0.032],
                                    offset_high+manipulator_pos_before[i],
                                    offset_high+manipulator_pos_after[i],
                                    offset_high+manipulator_pos_after[i],
                                    offset_low+manipulator_pos_after[i],
                                    [0.024, 0.024],
                                    offset_high+manipulator_pos_after[i]]

            trajectory_ori_list = [rest_ori+manipulator_ori_before[i],
                                    rest_ori+manipulator_ori_before[i],
                                    [0.032, 0.032],
                                    rest_ori+manipulator_ori_before[i],
                                    rest_ori+manipulator_ori_after[i],
                                    rest_ori+manipulator_ori_after[i],
                                    rest_ori+manipulator_ori_after[i],
                                    [0.024, 0.024],
                                    rest_ori+manipulator_ori_after[i]]
            last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
            last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))

            for j in range(len(trajectory_pos_list)):

                if len(trajectory_pos_list[j]) == 3:
                    if trajectory_pos_list[j][0] == last_pos[0] and trajectory_pos_list[j][1] == last_pos[1]:
                        # print('this is the vertical operation')
                        self.trajectory_vertical(last_pos, trajectory_pos_list[j], trajectory_ori_list[j])
                    else:
                        # print('test', trajectory_pos_list[j])
                        self.trajectory_horizontal(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j])
                    last_pos = np.copy(trajectory_pos_list[j])
                    last_ori = np.copy(trajectory_ori_list[j])

                elif len(trajectory_pos_list[j]) == 2:
                    self.gripper_control(trajectory_pos_list[j])

            for _ in range(20):
                p.stepSimulation()
                # self.images = self.get_image()
                time.sleep(1/480)
        while 1:
            p.stepSimulation()

    def calculate_items(self, item_num, item_xyz):

        min_xy = np.ones(2) * 100
        best_item_config = []

        if item_num % 2 == 0:
            fac = [] # 定义一个列表存放因子
            for i in range(1, item_num + 1):
                if item_num % i == 0:
                    fac.append(i)
                    continue
            for i in range(len(fac)):
                num_row = int(fac[i])
                # print(num_row)
                num_column = int(item_num / num_row)
                # print(num_column)
                len_x = num_row * item_xyz[0][0] + (num_row - 1) * self.gap_item
                len_y = num_column * item_xyz[0][1] + (num_column - 1) * self.gap_item

                if np.sum(np.array([len_x, len_y])) < np.sum(min_xy):
                    # print('for 2x2, this is the shorter distance')
                    min_xy = np.array([len_x, len_y])
                    best_item_config = [num_row, num_column]
        else:
            len_x1 = item_xyz[0][0] * item_num + (item_num - 1) * self.gap_item
            len_y1 = item_xyz[0][1]
            len_x2 = item_xyz[0][0]
            len_y2 = item_xyz[0][1] * item_num + (item_num - 1) * self.gap_item
            if np.sum(np.array([len_x1, len_y1])) < np.sum(np.array([len_x2, len_y2])):
                min_xy = np.array([len_x1, len_y1])
                best_item_config = [item_num, 1]
            else:
                min_xy = np.array([len_x2, len_y2])
                best_item_config = [1, item_num]
        best_item_config = np.asarray(best_item_config)

        return min_xy, best_item_config

    def reorder_items(self):

        min_result = []
        best_config = []
        for i in range(self.kinds):
            item_index = self.all_index[i] #! it can replace the self.cube_2x2 ...
            item_xyz = self.xyz_list[item_index, :]
            item_num = len(item_index)
            xy, config = self.calculate_items(item_num, item_xyz)
            # print(f'this is min xy {xy}')
            min_result.append(list(xy))
            # print(f'this is the best item config {config}')
            best_config.append(list(config))
        min_result = np.asarray(min_result).reshape(-1, 2)
        min_xy = np.copy(min_result)
        best_config = np.asarray(best_config).reshape(-1, 2)
        # print(min_xy)
        # print(best_config)

        #! 安排总的摆放
        iteration = 500
        all_num = best_config.shape[0]
        all_x = 100
        all_y = 100

        if all_num % 2 == 0:
            fac = [] # 定义一个列表存放因子
            for i in range(1, all_num + 1):
                if all_num % i == 0:
                    fac.append(i)
                    continue
            # print(fac)
        else:
            fac = [1, all_num]

        for i in range(iteration):
            
            sequence = np.random.choice(best_config.shape[0], size=self.kinds, replace=False)
            zero_or_90 = np.random.choice(np.array([0, 90]))

            for j in range(len(fac)):

                min_xy = np.copy(min_result)
                # print(f'this is the min_xy before rotation\n {min_xy}')

                num_row = int(fac[j])
                num_column = int(all_num / num_row)
                sequence = sequence.reshape(num_row, num_column)
                min_x = 0
                min_y = 0
                rotate_flag = np.full((num_row, num_column), False, dtype=bool)
                # print(f'this is {sequence}')

                for m in range(num_row):
                    
                    for n in range(num_column):
                        new_row = min_xy[sequence[m][n]]
                        zero_or_90 = np.random.choice(np.array([0, 90]))
                        if zero_or_90 == 90:
                            rotate_flag[m][n] = True
                            temp = new_row[0]
                            new_row[0] = new_row[1]
                            new_row[1] = temp

                    #! insert 'whether to rotate' here
                for a in range(num_row):
                    new_row = min_xy[sequence[a, :]]
                    min_x = min_x + np.max(new_row, axis=0)[0]

                for b in range(num_column):
                    new_column = min_xy[sequence[:, b]]
                    min_y = min_y + np.max(new_column, axis=0)[1]
                
                if min_x + min_y < all_x + all_y:
                    best_all_config = sequence
                    all_x = min_x
                    all_y = min_y
                    best_rotate_flag = rotate_flag
                    best_min_xy = np.copy(min_xy)
        print(f'in iteration{i}, the min all_x and all_y are {all_x} {all_y}')
        print(all_x + all_y)

        return self.reorder_block(best_config, best_all_config, best_rotate_flag, best_min_xy)

    def calculate_block(self, best_config, start_pos, index_block, item_index, item_xyz, index_flag):
        
        #! initiate the pos and ori
        #! we don't analysis these imported oris
        #! we directly define the ori is 0 or 90 degree, depending on the algorithm.
        item_pos = np.zeros([len(item_index), 3])
        item_ori = np.zeros([len(item_index), 3])
        # print(item_pos)
        num_2x2_row = best_config[index_block][0]
        num_2x2_column = best_config[index_block][1]
        index_2x2 = np.arange(item_pos.shape[0]).reshape(num_2x2_row, num_2x2_column)

        #! the initial position of the first items

        if index_flag == True:

            temp = np.copy(item_xyz[:, 0])
            item_xyz[:, 0] = item_xyz[:, 1]
            item_xyz[:, 1] = temp
            item_ori[:, 2] = math.pi/2
            # print(item_ori)
            temp = num_2x2_row
            num_2x2_row = num_2x2_column
            num_2x2_column = temp
            index_2x2 = index_2x2.transpose()
        else:

            item_ori[:, 2] = 0
            # print(item_ori)

        start_pos[0] = start_pos[0] + item_xyz[0][0]/2
        start_pos[1] = start_pos[1] + item_xyz[0][1]/2
        # print(f'this is try start {start_pos}')

        for j in range(num_2x2_row):
            for k in range(num_2x2_column):
                x_2x2 = start_pos[0] + (item_xyz[index_2x2[j][k]][0]) * j + self.gap_item * j
                y_2x2 = start_pos[1] + (item_xyz[index_2x2[j][k]][1]) * k + self.gap_item * k
                item_pos[index_2x2[j][k]][0] = x_2x2
                item_pos[index_2x2[j][k]][1] = y_2x2
        # print(item_pos)

        return item_ori, item_pos

    def reorder_block(self, best_config, best_all_config, best_rotate_flag, min_xy):

        print(f'the best configuration of all items is\n {best_all_config}')
        print(f'the best configuration of each kind of items is\n {best_config}')
        print(f'the rotate of each block of items is\n {best_rotate_flag}')
        print(f'this is the min_xy of each kind of items after rotation\n {min_xy}')

        num_all_row = best_all_config.shape[0]
        num_all_column = best_all_config.shape[1]

        start_x = [0]
        start_y = [0]
        previous_start_x = 0
        previous_start_y = 0

        for m in range(num_all_row):
            new_row = min_xy[best_all_config[m, :]]
            # print(new_row)
            # print(np.max(new_row, axis=0)[0])
            start_x.append((previous_start_x + np.max(new_row, axis=0)[0] + self.gap_block))
            previous_start_x = (previous_start_x + np.max(new_row, axis=0)[0] + self.gap_block)
        start_x = np.delete(start_x, -1)
        # print(f'this is start_x {start_x}')

        for n in range(num_all_column):
            new_column = min_xy[best_all_config[:, n]]
            # print(new_column)
            # print(np.max(new_column, axis=0)[1])
            start_y.append((previous_start_y + np.max(new_column, axis=0)[1] + self.gap_block))
            previous_start_y = (previous_start_y + np.max(new_column, axis=0)[1] + self.gap_block)
        start_y = np.delete(start_y, -1)
        # print(f'this is start_y {start_y}')

        #! determine the start position per item
        item_pos = np.zeros([len(self.xyz_list), 3])
        item_ori = np.zeros([len(self.xyz_list), 3])
        # print(self.xyz_list[self.all_index[0]])
        # print(self.all_index)
        for m in range(num_all_row):
            for n in range(num_all_column):

                item_index = self.all_index[best_all_config[m][n]]#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!determine the index of blocks
                # print('try', item_index)
                item_xyz = self.xyz_list[item_index, :]
                # print('try', item_xyz)
                start_pos = np.asarray([start_x[m], start_y[n]])
                index_block = best_all_config[m][n]
                index_flag = best_rotate_flag[m][n]

                ori, pos = self.calculate_block(best_config, start_pos, index_block, item_index, item_xyz, index_flag)
                # print('tryori', ori)
                # print('trypos', pos)
                item_pos[item_index] = pos
                item_ori[item_index] = ori

        # print('this is try', item_pos)
        # print('this is try', item_ori)

        return item_pos, item_ori # pos_list, ori_list

if __name__ == '__main__':

    random_flag = True

    command = 'virtual' #! image virtual real

    if command == 'image':

        env = Arm_env(is_render=True, num_2x2=2, num_2x3=3, num_2x4=4, num_pencil=0, order_flag = 'center', kinds = 3)
        
        image_chaotic = env.reset(True)
        temp = np.copy(image_chaotic[:, :, 0])
        image_chaotic[:, :, 0] = np.copy(image_chaotic[:, :, 2])
        image_chaotic[:, :, 2] = temp
        # print(f'this is {image_chaotic}')
        image_trim = env.change_config()
        temp = np.copy(image_trim[:, :, 0])
        image_trim[:, :, 0] = np.copy(image_trim[:, :, 2])
        image_trim[:, :, 2] = temp
        # print(image_trim.shape)

        new_img = np.concatenate((image_chaotic, image_trim), axis=1)
        # print(new_img)

        cv2.line(new_img,(int(new_img.shape[1]/2),0),(int(new_img.shape[1]/2),new_img.shape[0]),(0,0,0),20)
        cv2.imshow("Comparison between Chaotic Configuration and Trim Configuration", new_img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    if command == 'real':

        env = Arm_env(is_render=True, num_2x2=2, num_2x3=3, num_2x4=4, num_pencil=0, order_flag = 'center', kinds = 3)
        
        _ = env.reset(random_flag)
        _ = env.change_config()

        env.manipulator_operation_actual()

    if command == 'virtual':

        grasp_order = [1, 0, 2]
        env = Arm_env(is_render=True, num_2x2=2, num_2x3=1, num_2x4=1, num_pencil=0, order_flag = 'center', kinds = 3, grasp_order=grasp_order)

        _ = env.change_config()
        _ = env.reset(random_flag)
        
        env.manipulator_operation_virtual()
