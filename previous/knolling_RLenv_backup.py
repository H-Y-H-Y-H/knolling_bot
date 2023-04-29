from pickle import FALSE
import time
import logging
from xml.etree.ElementTree import TreeBuilder

import cv2
from easy_logx.easy_logx import EasyLog
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

logger = EasyLog(log_level=logging.INFO)


class Arm_env(gym.Env):

    def __init__(self, max_step, is_render=True, num_objects=1, x_grasp_accuracy=0.2, y_grasp_accuracy=0.2,
                 z_grasp_accuracy=0.2):

        self.kImageSize = {'width': 480, 'height': 480}

        self.step_counter = 0

        self.urdf_path = '../urdf'
        self.pybullet_path = pd.getDataPath()
        self.is_render = is_render

        self.ik_low = [-1.57, -1.57, -1.57, -1.57, -1.57, -10, -10]
        self.ik_high = [1.57, 1.57, 1.57, 1.57, 1.57, 10, 10]

        self.low_scale = np.array([0.05, -0.15, 0.005, - np.pi / 2, 0])
        self.high_scale = np.array([0.3, 0.15, 0.05, np.pi / 2, 0.4])
        self.low_act = -np.ones(5)
        self.high_act = np.ones(5)
        self.x_low_obs = self.low_scale[0]
        self.x_high_obs = self.high_scale[0]
        self.y_low_obs = self.low_scale[1]
        self.y_high_obs = self.high_scale[1]
        self.z_low_obs = self.low_scale[2]
        self.z_high_obs = self.high_scale[2]


        self.x_grasp_interval = (self.x_high_obs - self.x_low_obs) * x_grasp_accuracy
        self.y_grasp_interval = (self.y_high_obs - self.y_low_obs) * y_grasp_accuracy
        self.z_grasp_interval = (self.z_high_obs - self.z_low_obs) * z_grasp_accuracy

        self.obs = np.zeros(19)
        self.table_boundary = 0.05
        self.max_step = max_step

        self.friction = 0.99
        self.num_objects = num_objects
        # self.action_space = np.asarray([np.pi/3, np.pi / 6, np.pi / 4, np.pi / 2, np.pi])
        # self.shift = np.asarray([-np.pi/6, -np.pi/12, 0, 0, 0])
        self.ik_space = np.asarray([0.3, 0.4, 0.06, np.pi])  # x, y, z, yaw
        self.ik_space_shift = np.asarray([0, -0.2, 0, -np.pi / 2])

        self.slep_t = 1 / 50
        self.joints_index = [0, 1, 2, 3, 4, 7, 8]
        # 5 6 9不用管，固定的！
        self.init_joint_positions = [0, 0, -1.57, 0, 0, 0, 0, 0, 0, 0]

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
            cameraTargetPosition=[0.25, 0, 0.05],
            distance=0.4,
            yaw=90,
            pitch=-50.5,
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

        # self.action_space = spaces.Box(
        #     low=np.array([self.x_low_obs, self.y_low_obs, self.z_low_obs, self.yaw_low_obs, self.gripper_low_obs]),
        #     high=np.array(
        #         [self.x_high_obs, self.y_high_obs, self.z_high_obs, self.yaw_high_obs, self.gripper_high_obs]),
        #     dtype=np.float32)
        self.action_space = spaces.Box(low=self.low_act, high=self.high_act, dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.ones(19) * np.inf, high=np.ones(19) * np.inf, dtype=np.float32)
        self.seed()

    def get_parameters(self, num_2x2=0, num_2x3=0, num_2x4=0, num_pencil=0,
                       total_offset=[0.1, 0, 0.006], grasp_order=[1, 0, 2],
                       gap_item=0.03, gap_block=0.02,
                       from_virtual=True, real_operate=False, obs_order='1', random_offset = False):

        self.num_2x2 = num_2x2
        self.num_2x3 = num_2x3
        self.num_2x4 = num_2x4
        self.num_pencil = num_pencil
        self.total_offset = total_offset
        self.grasp_order = grasp_order
        self.gap_item = gap_item
        self.gap_block = gap_block
        self.from_virtual = from_virtual
        self.real_operate = real_operate
        self.obs_order = obs_order
        self.random_offset = random_offset
        self.num_list = np.array([self.num_2x2, self.num_2x3, self.num_2x4, self.num_pencil])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        self.step_counter = 0

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
        p.changeDynamics(baseid, -1, lateralFriction=self.friction)
        p.changeDynamics(self.arm_id, 7, lateralFriction=self.friction)
        p.changeDynamics(self.arm_id, 8, lateralFriction=self.friction)
        p.changeVisualShape(baseid, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=textureId)

        # Generate the pos and orin of objects randomly.
        self.obj_idx = []
        for i in range(self.num_objects):
            rdm_pos = [random.uniform(self.x_low_obs, self.x_high_obs), random.uniform(self.y_low_obs, self.y_high_obs),
                       0.01]
            rdm_ori = [0, 0, random.uniform(-math.pi / 2, math.pi / 2)]
            self.obj_idx.append(p.loadURDF(os.path.join(self.urdf_path, "item_1/%d.urdf" % i), basePosition=rdm_pos,
                                           baseOrientation=p.getQuaternionFromEuler(rdm_ori),
                                           flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
            p.changeDynamics(self.obj_idx[i], -1, lateralFriction=self.friction, spinningFriction=0.02,
                             rollingFriction=0.002)
            logger.debug(f'this is the urdf id: {self.obj_idx}')

        # ! initiate the position
        # TBD use xyz pos to initialize robot arm (IK)
        p.setJointMotorControlArray(self.arm_id, [0, 1, 2, 3, 4, 7, 8], p.POSITION_CONTROL,
                                    targetPositions=[0, -0.48627556248779596, 1.1546790099090924, 0.7016159753143177, 0,
                                                     0, 0],
                                    forces=[10] * 7)
        
        for _ in range(40):
            p.stepSimulation()
            self.images = self.get_image()
        return self.get_obs()

    def act(self, action_):
        # print(self.step_counter, action_)
        action_ = (action_ - self.low_act) / (self.high_act - self.low_act) * (self.high_scale - self.low_scale) + self.low_scale
        self.ik_angle = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=action_[:3], maxNumIterations=3000,
                                                     lowerLimits=self.ik_low, upperLimits=self.ik_high,
                                                     jointRanges=[3.14, 3.14, 3.14, 3.14, 3.14, 20, 20],
                                                     restPoses=[0, -0.48627556248779596, 1.1546790099090924,
                                                                0.7016159753143177, 0, 0, 0],
                                                     targetOrientation=p.getQuaternionFromEuler([0, 1.57, action_[3]]))
        # self.ik_angle_2 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=action_[:3], maxNumIterations=3000,
        #                                         lowerLimits=self.ik_low, upperLimits=self.ik_high, 
        #                                         targetOrientation=p.getQuaternionFromEuler([0, 1.57, action_[3]]))
        # self.ik_angle = np.asarray(self.ik_angle)
        # self.ik_angle_2 = np.asarray(self.ik_angle_2)
        # if np.sum(self.ik_angle - self.ik_angle) > 0.01:
        #     print('the calculation of ik is wrong!')

        p.setJointMotorControl2(self.arm_id, 4, p.POSITION_CONTROL, targetPosition=self.ik_angle[4], force=15,
                                maxVelocity=32)
        for i in [0, 2, 3]:
            p.setJointMotorControl2(self.arm_id, i, p.POSITION_CONTROL, targetPosition=self.ik_angle[i], force=41,
                                    maxVelocity=24)
        p.setJointMotorControl2(self.arm_id, 1, p.POSITION_CONTROL, targetPosition=self.ik_angle[1], force=82,
                                maxVelocity=24)

        for i in range(40):
            # self.images = self.get_image()
            p.stepSimulation()

            if self.is_render:
                time.sleep(self.slep_t)

    def slider_act(self, a_pos):

        a_pos[:4] = a_pos[:4] * self.ik_space + self.ik_space_shift
        # a_joint = a_pos[:5]  # * self.action_space + self.shift
        ik_angle = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=a_pos[:3], maxNumIterations=2000,
                                                targetOrientation=p.getQuaternionFromEuler([0, 1.57, a_pos[3]]))
        # Joint execution
        for i in range(5):
            p.setJointMotorControl2(self.arm_id, i, p.POSITION_CONTROL, targetPosition=ik_angle[i], force=10,
                                    maxVelocity=4)

        # Gripper execution
        a_gripper = a_pos[4] // 0.5001
        p.setJointMotorControlArray(self.arm_id, [7, 8],
                                    p.POSITION_CONTROL,
                                    targetPositions=[a_gripper, a_gripper])

        gripper_a = a_pos[4]
        gripper_a //= 0.5001

        if gripper_a == 1:
            self.gripper_control()

        for i in range(40):
            self.images = self.get_image()
            p.stepSimulation()
            time.sleep(self.slep_t)

    def gripper_control(self):
        flag = False
        while True:
            cur_pos = np.asarray(p.getJointStates(self.arm_id, [7, 8]))[:, 0]
            tar_pos = np.add(cur_pos, [0.036 / 20, 0.036 / 20])
            # logger.info(f'tar is {tar_pos}')
            p.setJointMotorControlArray(self.arm_id, [7, 8], p.POSITION_CONTROL, targetPositions=tar_pos)

            for i in range(20):
                p.stepSimulation()
                time.sleep(self.slep_t)

            obs_pos = np.asarray(p.getJointStates(self.arm_id, [7, 8]))[:, 0]
            # logger.info(f'obs is {obs_pos}')
            if obs_pos[1] >= 0.03186:
                logger.info(f'max!')
                break
            elif abs(obs_pos[1] - tar_pos[1]) > 0.0005:
                logger.info(f'get it, the flag is {flag}')
                flag = True
                break
            # if abs(obs_pos[1] - tar_pos[1]) > 0.0005:
            #     if obs_pos[1] >= 0.03100:
            #         logger.info(f'max!')
            #         break
            #     flag = True
            #     if abs(obs_pos[0] - obs_pos[1]) < 0.0005:
            #         logger.info(f'get it, the flag is {flag}')
            #         break
            flag = False
            # if tar_pos[0] >= 0.032:
            #     flag = False
            #     logger.info(
            #         f'decided to grasp and the distance is appropriate, but not catch the box, the flag is {flag}')
            #     break

        return flag

    def step(self, action_):
        # action_: 4 xyzyaw + gripper [0,1]

        # current_pos = obs[:3]# from obs
        # current_yaw = obs[5]# from obs
        # current_pos, current_yaw = np.asarray(current_pos), np.asarray(current_yaw)
        # current_state = np.append(current_pos, current_yaw)
        # action_[:4] = action_[:4] * self.dv + current_state

        self.act(action_)
        print('aaaaaaaaaaaaaa')
        # print(f'the action_ is {action_}')

        obs = self.get_obs()
        # print(f'the ee is {obs[:3]}')

        # print(f'the box is {obs[6:9]}')

        # ! determine whether the distance is appropriate

        r, done = self.reward_func(obs, action_)

        self.step_counter += 1
        if self.step_counter >= self.max_step:
            done = True

        return obs, r, done, {}

    def reward_func(self, obs, action_):

        reward_ = 0
        x, y, z = obs[:3]
        # cur_box_pos = obs[6:6+self.num_objects*3]
        cur_box_pos = obs[6:(6 + 3)]
        ee_yaw = obs[5]
        # box_yaw = obs[5 + self.num_objects * 3 + 3]
        obj_yaw = obs[5 + 3 + 3]

        gripper_a = action_[4]
        gripper_a //= 0.5001

        # get_objects = None
        # if gripper_a == 1:
        #     get_objects = self.gripper_control()
        # if get_objects == True:
        #     test_distance = 0.03
        #     test_pos = [x, y, z + test_distance]
        #     ik_angle = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=test_pos, maxNumIterations=2000,
        #                                         targetOrientation=p.getQuaternionFromEuler([0, 1.57, action_[3]]))
        #     for i in [0, 2, 3]:
        #         p.setJointMotorControl2(self.arm_id, i, p.POSITION_CONTROL, targetPosition=ik_angle[i], force=4.1,
        #                                 maxVelocity=4.8)
        #     p.setJointMotorControl2(self.arm_id, 1, p.POSITION_CONTROL, targetPosition=ik_angle[1], force=8.2,
        #                             maxVelocity=4.8)
        #     p.setJointMotorControl2(self.arm_id, 4, p.POSITION_CONTROL, targetPosition=ik_angle[4], force=1.5,
        #                             maxVelocity=6.4)
        #     for i in range(60):
        #         # self.images = self.get_image()
        #         p.stepSimulation()
        #         if self.is_render:
        #             time.sleep(self.slep_t)

        #     new_obs = self.get_obs()
        #     new_box_pos = new_obs[6:6 + 3]
        #     if (new_box_pos[2] - cur_box_pos[2]) > test_distance - 0.01:
        #         get_objects = True
        #     else:
        #         get_objects = False
        #         logger.info('This "True" signal is unreal, the flag is still false!')
        #         time.sleep(3)

        ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # distance = np.linalg.norm(obs[0:3] - obs[6:6+self.num_objects*3])
        distance = np.linalg.norm(obs[:3] - obs[6:(6 + 3)])
        logger.debug(f'the distance between ee and box is {distance}')

        boundary = bool(x < self.x_low_obs - self.table_boundary or x > self.x_high_obs + self.table_boundary
                        or y < self.y_low_obs - self.table_boundary or y > self.y_high_obs + self.table_boundary
                        or z < self.z_low_obs - 0.005 or z > self.z_high_obs + self.table_boundary)
        box_boundary = bool(cur_box_pos[0] < self.x_low_obs - self.table_boundary or cur_box_pos[
            0] > self.x_high_obs + self.table_boundary
                            or cur_box_pos[1] < self.y_low_obs - self.table_boundary or cur_box_pos[
                                1] > self.y_high_obs + self.table_boundary
                            or cur_box_pos[2] < self.z_low_obs - self.table_boundary or cur_box_pos[
                                2] > self.z_high_obs + self.table_boundary)

        top_decision = bool(abs(x - cur_box_pos[0]) < self.x_grasp_interval and
                            abs(y - cur_box_pos[1]) < self.y_grasp_interval and
                            (0.03 < z - cur_box_pos[2]) < 0.15)

        grasp_decision = bool(abs(x - cur_box_pos[0]) < self.x_grasp_interval and
                              abs(y - cur_box_pos[1]) < self.y_grasp_interval and
                              abs(z - cur_box_pos[2]) < self.z_grasp_interval)

        # elif self.step_counter > self.kMaxEpisodeSteps:
        #     r = -0.1
        #     logger.info('times up')
        #     self.terminated = True
        self.terminated = False

        # r = ((self.x_high_obs - self.x_low_obs) / 3 - abs(x - cur_box_pos[0])) * 5
        # self.r += r
        # r1 = ((self.x_high_obs - self.x_low_obs) / 3 - abs(x - cur_box_pos[0])) * 5
        #
        # r2 = ((self.y_high_obs - self.y_low_obs) / 3 - abs(y - cur_box_pos[1])) * 5
        #
        # r3 = ((self.z_high_obs - self.z_low_obs) / 3 - abs(z - cur_box_pos[2] - 0.03)) * 50

        # if distance < 0.03:
        #     r = 0.0001
        #     self.r += r
        #     logger.info('next to the box')
        #     self.terminated = False

        # if grasp_decision:
        #     r = 0.01
        #     self.r += r
        #     logger.info('this position is appropriate to grasp, keep it!')
        #     self.terminated = False

        # if top_decision:
        #     r = 0.001
        #     self.r += r
        #     logger.info('on the top of box')
        #     self.terminated = False

        if abs(obj_yaw - ee_yaw) < 0.05:
            r = 0.01
            logger.debug('the yaw is same')
            self.terminated = False

        # if get_objects == False:
        #     r = -1
        #     logger.info('grasp failed, the reward is -1!')
        #     self.terminated = True

        elif boundary:
            r = -1
            logger.info('hit the border')
            print(obs[:3])
            # print(f'xyz is {x},{y},{z}')
            self.terminated = True

        elif box_boundary:
            r = -3
            logger.info('the box hit the border')
            self.terminated = True

        # elif get_objects == True:
        #     r = 10
        #     logger.info('get the box, the reward is 10!')
        #     # time.sleep(3)
        #     self.terminated = True

        # elif self.decision_flag == False:
        #     r = -1
        #     logger.info('it is too early to grasp')
        #     self.terminated = True

        else:
            r = 0

        cube_state = np.copy(obs[[6,7,8,11]])
        ee_state = np.copy(obs[[0,1,2,5]])
        cube_state[2] += 0.03 # cube height: 0.03 m

        reward_ = 0.4 - np.linalg.norm(cube_state - ee_state)
        # print("step reward", reward_)

        return reward_, self.terminated

    def get_obs(self):
        # Get end-effector obs
        self.ee_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
        self.ee_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))
        self.ee_pos.astype(np.float32)
        self.ee_ori.astype(np.float32)
        logger.debug(self.ee_ori)

        # Get box obs!
        self.box_pos, self.box_ori = [], []
        for i in range(len(self.obj_idx)):
            box_pos = np.asarray(p.getBasePositionAndOrientation(self.obj_idx[i])[0])
            box_ori = np.asarray(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.obj_idx[0])[1]))
            self.box_pos = np.append(self.box_pos, box_pos).astype(np.float32)
            self.box_ori = np.append(self.box_ori, box_ori).astype(np.float32)
        logger.debug(f'self.box_pos = {self.box_pos}')
        logger.debug(f'self.box_ori = {self.box_ori}')

        # Get Joint angle
        Joint_info_list = p.getJointStates(self.arm_id, self.joints_index)
        self.joints_angle = []
        for i in range(len(Joint_info_list)):
            self.joints_angle.append(Joint_info_list[i][0])
        self.joints_angle = np.asarray(self.joints_angle)
        self.joints_angle[len(self.joints_angle) - 2:] = abs(self.joints_angle[len(self.joints_angle) - 2:]) // 0.01601
        self.joints_angle = self.joints_angle.astype(np.float32)

        # ee_pos = 3, ee_ori = 3, box_pos = 3 * num_objects, box_ori = 3 * num_objects, joints_angle = 7
        self.obs = np.concatenate([self.ee_pos, self.ee_ori, self.box_pos, self.box_ori, self.joints_angle])
        self.obs = self.obs.astype(np.float32)
        logger.debug(f'the shape of obs is {self.obs.size}')

        return self.obs

    def get_image(self, gray_flag=False, resize_flag=False):
        # reset camera
        (width, length, image, _, _) = p.getCameraImage(width=960,
                                                        height=720,
                                                        viewMatrix=self.view_matrix,
                                                        projectionMatrix=self.projection_matrix,
                                                        renderer=p.ER_BULLET_HARDWARE_OPENGL)
        if gray_flag:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if resize_flag:
            image = cv2.resize(image, (self.kImageSize['width'], self.kImageSize['height']))[None, :, :] / 255.
            return image

        return image


if __name__ == '__main__':

    mode = 2

    if mode == 1:  # ! use the slider module

        env = Arm_env(max_step=3, is_render=True, num_objects=1)
        num_item = 2
        num_epoch = 3
        env.slep_t = 1 / 240
        num_step = 400

        Debug_para = []

        Debug_para.append(p.addUserDebugParameter("x", 0, 1, 0))
        Debug_para.append(p.addUserDebugParameter("y", 0, 1, 0.5))
        Debug_para.append(p.addUserDebugParameter("z", 0, 1, 0))
        Debug_para.append(p.addUserDebugParameter("yaw", 0, 1, 0.5))
        Debug_para.append(p.addUserDebugParameter("gripper", 0, 1, 0.5))

        # 5 Joints and 1 End-effector
        for epoch in range(num_epoch):
            state = env.reset()
            epoch_r = 0

            for i_step in range(num_step):
                print(i_step)
                a = []
                # get parameters
                for j in range(5):
                    a.append(p.readUserDebugParameter(Debug_para[j]))
                a = np.asarray(a)
                env.slider_act(a)

                # epoch_r += r

    elif mode == 2:  # ! use the random action_

        env = Arm_env(max_step=3, is_render=True, num_objects=1)
        obs = env.reset()
        for _ in range(10000):
            a = env.action_space.sample()
            print('bbbbbbbbbbbbbbb')
            # a = [0.05, -0.15, 0.005, 0.8985989, 0.4]
            state_, reward, done, _ = env.step(a)

            # time.sleep(100)
            if done:
                print('reset')
                obs = env.reset()

    elif mode == 3:

        num_2x2 = 2
        num_2x3 = 2
        num_2x4 = 2
        total_offset = [0.15, 0, 0.006]
        grasp_order = [2, 1, 0]
        gap_item = 0.015
        gap_block = 0.02
        from_virtual = False
        random_offset = True
        real_operate = True
        obs_order = 'real_image_obj'

        env = Arm_env(is_render=True)
        env.get_parameters(num_2x2=num_2x2, num_2x3=num_2x3, num_2x4=num_2x4,
                           total_offset=total_offset, grasp_order=grasp_order,
                           gap_item=gap_item, gap_block=gap_block, from_virtual=from_virtual,
                           real_operate=real_operate, obs_order=obs_order, random_offset=random_offset)
