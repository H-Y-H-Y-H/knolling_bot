import time
import logging
from xml.etree.ElementTree import TreeBuilder

import cv2
# from easy_logx.easy_logx import EasyLog
from gym import spaces
from gym.utils import seeding
import pybullet as p
import pybullet_data as pd
import os
import gym
# import cv2
import numpy as np
import random
import math
import cv2
from PIL import Image
from urdfpy import URDF

# logger = EasyLog(log_level=logging.INFO)

def find_corner(x,y,type,yaw):

    gamma = yaw

    rot_z = [[np.cos(gamma), -np.sin(gamma)],
             [np.sin(gamma), np.cos(gamma)]]

    pos = [x, y]

    rot_z = np.asarray(rot_z)

    if type == 0:
        c1 = [16/2,16/2]
        c2 = [16/2,-16/2]
        c3 = [-16/2,16/2]
        c4 = [-16/2,-16/2]

    elif type == 1:
        c1 = [24/2,16/2]
        c2 = [24/2,-16/2]
        c3 = [-24/2,16/2]
        c4 = [-24/2,-16/2]

    elif type == 2:
        c1 = [32/2,16/2]
        c2 = [32/2,-16/2]
        c3 = [-32/2,16/2]
        c4 = [-32/2,-16/2]

    c1,c2,c3,c4 = np.asarray(c1),np.asarray(c2),np.asarray(c3),np.asarray(c4)
    c1 = c1/1000
    c2 = c2/1000
    c3 = c3/1000
    c4 = c4/1000

    corn1 = np.dot(rot_z,c1)
    corn2 = np.dot(rot_z,c2)
    corn3 = np.dot(rot_z,c3)
    corn4 = np.dot(rot_z,c4)

    corn11 = [corn1[0] + x, corn1[1] + y]
    corn22 = [corn2[0] + x, corn2[1] + y]
    corn33 = [corn3[0] + x, corn3[1] + y]
    corn44 = [corn4[0] + x, corn4[1] + y]

    return corn11, corn22, corn33, corn44


def resolve_img(corn1,corn2,corn3,corn4):

    along_axis = [abs(np.arctan(corn1[1] / (corn1[0]-0.15))), abs(np.arctan(corn2[1] / (corn2[0]-0.15))), abs(np.arctan(corn3[1] / (corn3[0]-0.15))),
                  abs(np.arctan(corn4[1] / (corn4[0]-0.15)))]

    # print(along_axis)


    cube_h = 12/1000 # cube height (m)

    dist1 = math.dist([0.15, 0], [corn1[0], corn1[1]])
    dist2 = math.dist([0.15, 0], [corn2[0], corn2[1]])
    dist3 = math.dist([0.15, 0], [corn3[0], corn3[1]])
    dist4 = math.dist([0.15, 0], [corn4[0], corn4[1]])

    # print(dist1,dist2,dist3,dist4)
    add_value1 = (cube_h * dist1) / (0.387 - cube_h)
    add_value2 = (cube_h * dist2) / (0.387 - cube_h)
    add_value3 = (cube_h * dist3) / (0.387 - cube_h)
    add_value4 = (cube_h * dist4) / (0.387 - cube_h)

    # new_dist1 = dist1 + add_value1
    # new_dist2 = dist2 + add_value2
    # new_dist3 = dist3 + add_value3
    # new_dist4 = dist4 + add_value4


    # sign = lambda x: math.copysign(1, x)

    corn1[0] = corn1[0] + np.sign(corn1[0]-0.15) * add_value1 * math.cos(along_axis[0])
    corn1[1] = corn1[1] + np.sign(corn1[1]) * add_value1 * math.sin(along_axis[0])

    corn2[0] = corn2[0] + np.sign(corn2[0]-0.15) * add_value2 * math.cos(along_axis[1])
    corn2[1] = corn2[1] + np.sign(corn2[1]) * add_value2 * math.sin(along_axis[1])

    corn3[0] = corn3[0] + np.sign(corn3[0]-0.15) * add_value3 * math.cos(along_axis[2])
    corn3[1] = corn3[1] + np.sign(corn3[1]) * add_value3 * math.sin(along_axis[2])

    corn4[0] = corn4[0] + np.sign(corn4[0]-0.15) * add_value4 * math.cos(along_axis[3])
    corn4[1] = corn4[1] + np.sign(corn4[1]) * add_value4 * math.sin(along_axis[3])

    return corn1, corn2, corn3, corn4

def xyz_resolve(x,y):

    dist = math.dist([0.15, 0], [x, y])

    cube_h = 12/1000

    add_value = (cube_h * dist) / (0.387 - cube_h)

    along_axis = abs(np.arctan(y/(x-0.15)))

    # sign = lambda x: math.copysign(1, x)

    x_new = x + np.sign(x-0.15) * add_value * math.cos(along_axis)
    y_new = y + np.sign(y) * add_value * math.sin(along_axis)

    return x_new, y_new



class Arm_env(gym.Env):

    def __init__(self,max_step, is_render=True, boxes_index=None, x_grasp_accuracy=0.2, y_grasp_accuracy=0.2,
                 z_grasp_accuracy=0.2):

        self.kImageSize = {'width': 480, 'height': 480}

        self.step_counter = 0

        self.urdf_path = '../urdf'
        self.pybullet_path = pd.getDataPath()
        self.is_render = is_render

        self.x_low_obs = 0.05
        self.x_high_obs = 0.3
        self.y_low_obs = -0.15
        self.y_high_obs = 0.15
        self.z_low_obs = 0.005
        self.z_high_obs = 0.05
        self.x_grasp_interval = (self.x_high_obs - self.x_low_obs) * x_grasp_accuracy
        self.y_grasp_interval = (self.y_high_obs - self.y_low_obs) * y_grasp_accuracy
        self.z_grasp_interval = (self.z_high_obs - self.z_low_obs) * z_grasp_accuracy

        self.yaw_low_obs = - np.pi / 2
        self.yaw_high_obs = np.pi / 2
        self.gripper_low_obs = 0
        self.gripper_high_obs = 0.4
        self.obs = np.zeros(19)
        self.table_boundary = 0.05
        self.max_step = max_step

        self.friction = 0.99
        self.boxes_index = boxes_index
        # self.action_space = np.asarray([np.pi/3, np.pi / 6, np.pi / 4, np.pi / 2, np.pi])
        # self.shift = np.asarray([-np.pi/6, -np.pi/12, 0, 0, 0])
        self.ik_space = np.asarray([0.3, 0.4, 0.06, np.pi])  # x, y, z, yaw
        self.ik_space_shift = np.asarray([0, -0.2, 0, -np.pi / 2])

        self.slep_t = 1 / 120
        self.joints_index = [0, 1, 2, 3, 4, 7, 8]
        # 5 6 9不用管，固定的！
        self.init_joint_positions = [0, 0, -1.57, 0, 0, 0, 0, 0, 0, 0]

        # if self.is_render:
        #     p.connect(p.GUI)
        # else:
        #     p.connect(p.DIRECT)

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
            cameraTargetPosition=[0.150, 0, 0], #0.175
            distance=0.4,
            yaw=90,
            pitch = -90,
            roll=0,
            upAxisIndex=2)
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_parameters['fov'],
            aspect=self.camera_parameters['width'] /
                   self.camera_parameters['height'],
            nearVal=self.camera_parameters['near'],
            farVal=self.camera_parameters['far'])

        if random.uniform(0, 1) > 0.5:
            p.configureDebugVisualizer(lightPosition=[random.randint(1, 1), random.uniform(0, 1), 2],
                                       shadowMapResolution=8192, shadowMapIntensity=np.random.randint(5, 7) / 10)
        else:
            p.configureDebugVisualizer(lightPosition=[random.randint(1, 1), random.uniform(-1, 0), 2],
                                       shadowMapResolution=8192, shadowMapIntensity=np.random.randint(5, 7) / 10)
        p.resetDebugVisualizerCamera(cameraDistance=0.7,
                                     cameraYaw=45,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[0.1, 0, 0.4])
        p.setAdditionalSearchPath(pd.getDataPath())

    def reset_table(self, close_flag = False, texture_flag = False, use_lego_urdf = None, lego_list = None):
        self.r = 0
        self.step_counter = 0

        p.resetSimulation()

        baseid = p.loadURDF(os.path.join(self.urdf_path, "plane_1.urdf"), basePosition=[0, -0.2, 0],
                            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),useFixedBase=1,
                            flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)


        if not close_flag:
            p.setGravity(0, 0, -10)
        else:
            if random.random() < 0.5:  # x_mode
                wall_flag = 0

                wall_pos = np.random.uniform(0, 0.23)
                p.setGravity(-10, 0, -15)
                wallid = p.loadURDF(os.path.join(self.urdf_path, "plane_2.urdf"), basePosition=[wall_pos, 0, 0],
                                    baseOrientation=p.getQuaternionFromEuler([0, 1.57, 0]), useFixedBase=1,
                                    flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
            else:  # y_mode
                wall_flag = 1

                wall_pos = np.random.uniform(-0.18, 0.12)
                p.setGravity(0, -10, -15)
                wallid = p.loadURDF(os.path.join(self.urdf_path, "plane_2.urdf"), basePosition=[0, wall_pos, 0],
                                    baseOrientation=p.getQuaternionFromEuler([-1.57, 0, 0]), useFixedBase=1,
                                    flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

            p.changeVisualShape(wallid, -1, rgbaColor=[1, 1, 1, 0])

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

        # Texture change
        background = np.random.randint(5, 6)
        textureId = p.loadTexture(f"../urdf/img_{background}.png")
        # textureId = p.loadTexture(f"../urdf/textures/red2.png")
        p.changeVisualShape(baseid, -1, textureUniqueId=textureId, )

        p.changeDynamics(baseid, -1, lateralFriction=self.friction)


        # p.changeVisualShape(baseid, -1, rgbaColor=[1, 1, 1, 1])

        # Generate the pos and orin of objects randomly.
        self.obj_idx = []

        while True:
            # for i in range(1):
            dis_flag = True

            rdm_pos_z = 0.006
            rdm_ori_yaw = np.random.uniform(0, np.pi, size=(len(self.boxes_index) - 1))
            # rdm_ori_yaw = np.zeros(self.num_objects -1)
            rdm_ori_yaw = np.append(rdm_ori_yaw, rdm_ori_yaw[0])

            if close_flag:
                # print('this is wall pos', wall_pos)
                # print(wall_flag)
                if wall_flag == 0: # x
                    rdm_pos_x = np.random.uniform(wall_pos+0.02, wall_pos+0.25, size=(len(self.boxes_index) - 1))
                    rdm_pos_y = np.random.uniform(-0.14, 0.14, size=(len(self.boxes_index) - 1))
                else: # y
                    rdm_pos_x = np.random.uniform(0.06, 0.25, size=(len(self.boxes_index) - 1))
                    rdm_pos_y = np.random.uniform(wall_pos + 0.02, wall_pos+0.25, size=(len(self.boxes_index) - 1))
            else:
                rdm_pos_x = np.random.uniform(0.06, 0.25, size=(len(self.boxes_index) - 1))
                rdm_pos_y = np.random.uniform(-0.14, 0.14, size=(len(self.boxes_index) - 1))

            rot_parallel = [[np.cos(rdm_ori_yaw[0]), -np.sin(rdm_ori_yaw[0])],
                            [np.sin(rdm_ori_yaw[0]), np.cos(rdm_ori_yaw[0])]]
            rot_parallel = np.asarray(rot_parallel)

            xy_parallel = np.dot(rot_parallel, np.asarray([np.random.uniform(-0.016, 0.016), 0.050]))

            xy_parallel = np.add(xy_parallel, np.asarray([rdm_pos_x[0], rdm_pos_y[0]]))
            # print(xy_parallel)

            rdm_pos_x = np.append(rdm_pos_x, xy_parallel[0])
            rdm_pos_y = np.append(rdm_pos_y, xy_parallel[1])

            # rdm_pos_x = [0.15, 0.15, 0.15]
            # rdm_pos_y = [0.18, -0.18, 0]
            for i in range(len(self.boxes_index)):
                if dis_flag == False:
                    break

                if i == 0:
                    num2 = len(self.boxes_index) - 1

                else:
                    num2 = len(self.boxes_index)

                # print(num2)
                for j in range(i + 1, num2):

                    dis_check = math.dist([rdm_pos_x[i], rdm_pos_y[i]], [rdm_pos_x[j], rdm_pos_y[j]])

                    if dis_check < 0.048:
                        # print(i,"and",j,"gg")
                        dis_flag = False

            #
            if dis_flag == True:
                break

        r1 = np.random.uniform(0, 0.9)
        g1 = np.random.uniform(0, 0.9)
        b1 = np.random.uniform(0, 0.9)
        lw_list = []
        lw_data = np.array([[0.016, 0.016, 0.012],
                            [0.024, 0.016, 0.012],
                            [0.032, 0.016, 0.012]])

        if use_lego_urdf == True:
            lego_path = "../urdf/"
            for i in range(len(lego_list)):
                for j in range(lego_list[i]):
                    lw_list.append(lw_data[i])
                    self.obj_idx.append(p.loadURDF((lego_path + "item_%d/%d.urdf" % (i, j)),
                                   basePosition=[rdm_pos_x[i], rdm_pos_y[i], rdm_pos_z],
                                   baseOrientation=p.getQuaternionFromEuler([0, 0, rdm_ori_yaw[i]]), useFixedBase=0,
                                   flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
                    #
                    r = np.random.uniform(0, 0.9)
                    g = np.random.uniform(0, 0.9)
                    b = np.random.uniform(0, 0.9)

                    if random.random() < 0.05:
                        p.changeVisualShape(self.obj_idx[i], -1, rgbaColor=(0.1, 0.1, 0.1, 1))
                    else:
                        p.changeVisualShape(self.obj_idx[i], -1, rgbaColor=(r, g, b, 1))

                    if len(self.boxes_index) >= 3:
                        if i == 0 or i == (len(self.boxes_index) - 1):
                            p.changeVisualShape(self.obj_idx[i], -1, rgbaColor=(r1, g1, b1, 1))
                    else:
                        pass
        else:
            lego_path = "../urdf/box_generator/"
            for i in range(len(self.boxes_index)):
                boxes = URDF.load('../urdf/box_generator/box_%d.urdf' % self.boxes_index[i])
                lw_list.append(boxes.links[0].visuals[0].geometry.box.size)

                self.obj_idx.append(p.loadURDF((lego_path + "box_%d.urdf" % self.boxes_index[i]), basePosition=[rdm_pos_x[i], rdm_pos_y[i], rdm_pos_z],
                                   baseOrientation=p.getQuaternionFromEuler([0, 0, rdm_ori_yaw[i]]), useFixedBase=0,
                                   flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
                #
                r = np.random.uniform(0, 0.9)
                g = np.random.uniform(0, 0.9)
                b = np.random.uniform(0, 0.9)

                if random.random() < 0.05:
                    p.changeVisualShape(self.obj_idx[i], -1, rgbaColor=(0.1, 0.1, 0.1, 1))
                else:
                    p.changeVisualShape(self.obj_idx[i], -1, rgbaColor=(r, g, b, 1))

                if len(self.boxes_index) >= 3:
                    if i == 0 or i == (len(self.boxes_index) - 1):
                        p.changeVisualShape(self.obj_idx[i], -1, rgbaColor=(r1, g1, b1, 1))
                else:
                    pass
        lw_list = np.asarray(lw_list)

        for _ in range(int(40+close_flag*260)):
            # time.sleep(1/480)
            p.stepSimulation()


        return self.get_obs(), lw_list[:, :2]

    def get_obs(self):
        # Get end-effector obs
        self.ee_pos = np.zeros(3)
        self.ee_ori = np.zeros(3)

        # Get box obs!
        self.box_pos, self.box_ori = [], []
        for i in range(len(self.obj_idx)):
            box_pos = np.asarray(p.getBasePositionAndOrientation(self.obj_idx[i])[0])
            box_ori = np.asarray(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.obj_idx[i])[1]))
            self.box_pos = np.append(self.box_pos, box_pos).astype(np.float32)
            self.box_ori = np.append(self.box_ori, box_ori).astype(np.float32)

        # ee_pos = 3, ee_ori = 3, box_pos = 3 * num_objects, box_ori = 3 * num_objects, joints_angle = 7
        self.obs = np.concatenate([self.ee_pos, self.ee_ori, self.box_pos, self.box_ori])
        self.obs = self.obs.astype(np.float32)
        # logger.debug(f'the shape of obs is {self.obs.size}')

        return self.obs

    def get_image(self):
        # reset camera
        light_x = 1000
        light_y = 0
        light_z = 5
        (width, length, image, _, _) = p.getCameraImage(width=640,
                                                        height=480,
                                                        viewMatrix=self.view_matrix,
                                                        projectionMatrix=self.projection_matrix,
                                                        renderer=p.ER_BULLET_HARDWARE_OPENGL)
                                                        # lightDirection = [light_x, light_y, light_z])
        # image = image[:,80:560]
        # image = image[112:368,192:448]
        # image = image
        # img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # # rgb_opengl = cv2.resize(image, (256, 256))
        # rgb_opengl = image
        # rgbim = Image.fromarray(rgb_opengl)
        # rgbim_no_alpha = rgbim.convert('RGB')
        rgbim_no_alpha = image

        return rgbim_no_alpha


