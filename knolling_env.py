import time
import logging

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

logger=EasyLog(log_level=logging.INFO)

class Arm_env(gym.Env):

    def __init__(self, is_render=True,  num_objects=1):

        self.kImageSize = {'width': 480, 'height': 480}

        self.step_counter = 0

        self.urdf_path = 'urdf'
        self.pybullet_path = pd.getDataPath()
        self.is_render = is_render

        self.x_low_obs = 0
        self.x_high_obs = 0.3
        self.y_low_obs = -0.2
        self.y_high_obs = 0.2
        self.z_low_obs = 0.001
        self.z_high_obs = 0.35

        self.x_low_act = -0.5
        self.x_high_act = 0.5
        self.y_low_act = -0.5
        self.y_high_act = 0.5
        self.z_low_act = -0.5
        self.z_high_act = 0.5
        self.yaw_low_act = -2
        self.yaw_high_act = 2
        self.gripper_low_act = 0
        self.gripper_high_act = 1

        self.friction = 0.99
        self.num_objects = num_objects
        # self.action_space = np.asarray([np.pi/3, np.pi / 6, np.pi / 4, np.pi / 2, np.pi])
        # self.shift = np.asarray([-np.pi/6, -np.pi/12, 0, 0, 0])
        self.ik_space = np.asarray([0.3, 0.4, 0.06, np.pi]) # x, y, z, yaw
        self.ik_space_shift = np.asarray([0, -0.2, 0, -np.pi/2])

        self.slep_t = 1./240
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
            distance=0.38,
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

        self.action_space = spaces.Box(low=np.array([self.x_low_act, self.y_low_act, self.z_low_act, self.yaw_low_act, self.gripper_low_act]),
                                        high=np.array([self.x_high_act, self.y_high_act, self.z_high_act, self.yaw_high_act, self.gripper_high_act]),
                                        dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.ones(19)*np.inf,
                                            high=np.ones(19)*np.inf,
                                            dtype=np.float32)
        self.seed()
        self.reset()            
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        self.r = 0
        self.step_counter = 0
        
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_low_obs],
            lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_low_obs],
            lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_high_obs, self.z_low_obs],
            lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_high_obs, self.z_low_obs],
            lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_low_obs])

        baseid = p.loadURDF(os.path.join(self.urdf_path, "base.urdf"), basePosition=[0, 0, -0.05], useFixedBase=1, flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        self.arm_id = p.loadURDF(os.path.join(self.urdf_path, "robot_arm928/robot_arm1.urdf"),
                                 basePosition=[-.08, 0, 0.02], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        textureId = p.loadTexture(os.path.join(self.urdf_path, "table/table.png"))
        p.changeDynamics(baseid, -1, lateralFriction=self.friction, spinningFriction=0.02, rollingFriction=0.002)
        p.changeDynamics(self.arm_id, 7, lateralFriction=self.friction, spinningFriction=0.02, rollingFriction=0.002)
        p.changeDynamics(self.arm_id, 8, lateralFriction=self.friction, spinningFriction=0.02, rollingFriction=0.002)
        p.changeVisualShape(baseid, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=textureId)


        # Generate the pos and orin of objects randomly.
        self.obj_idx = []
        for i in range(self.num_objects):
            rdm_pos = [random.uniform(self.x_low_obs, self.x_high_obs), random.uniform(self.y_low_obs, self.y_high_obs),
                       0.01]
            rdm_ori = [0, 0, random.uniform(-math.pi / 2, math.pi / 2)]
            self.obj_idx.append(p.loadURDF(os.path.join(self.urdf_path, "box/box%d.urdf" % i), basePosition=rdm_pos,
                                      baseOrientation=p.getQuaternionFromEuler(rdm_ori),flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
            p.changeDynamics(self.obj_idx[i], -1, lateralFriction=self.friction, spinningFriction=0.02,
                             rollingFriction=0.002)
            logger.debug(f'this is the urdf id: {self.obj_idx}')

        #! initiate the position
        # TBD use xyz pos to initialize robot arm (IK)
        p.setJointMotorControlArray(self.arm_id, [0, 1, 2, 3, 4, 7, 8], p.POSITION_CONTROL,
                                    targetPositions=[0, -0.48627556248779596, 1.1546790099090924, 0.7016159753143177, 0, 0, 0],
                                    forces=[10] * 7)
        for _ in range(40):
            p.stepSimulation()
        return self.get_obs()

    def act(self, action):

        dv = 0.1
        # action = np.array([0.1, 0.1, 0.1, 1.57, 0.5])
        logger.debug(f'action is: {action}')
        current_obs = p.getLinkState(self.arm_id, 9)[0]
        current_yaw = p.getEulerFromQuaternion(p.getLinkState(self.arm_id,9)[1])[2]
        logger.debug(f'current yaw is: {current_yaw}')
        current_obs, current_yaw = np.asarray(current_obs), np.asarray(current_yaw)
        current_state = np.append(current_obs, current_yaw)
        logger.debug(f'current state is: {current_state}')
        action[:4] = action[:4] * dv + current_state
        logger.debug(f'action is: {action}')

        ik_angle = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=action[:3], maxNumIterations=2000,
                                                targetOrientation=p.getQuaternionFromEuler([0, 1.57, action[3]]))
        for i in [0,2,3]:
            p.setJointMotorControl2(self.arm_id, i, p.POSITION_CONTROL, targetPosition=ik_angle[i], force=4.1, maxVelocity=4.8)
        p.setJointMotorControl2(self.arm_id, 1, p.POSITION_CONTROL, targetPosition=ik_angle[1], force=8.2, maxVelocity=4.8)
        p.setJointMotorControl2(self.arm_id, 4, p.POSITION_CONTROL, targetPosition=ik_angle[4], force=1.5, maxVelocity=6.4)
        
        for i in range(10):
            self.images = self.get_image()
            p.stepSimulation()
            if self.is_render:
                time.sleep(self.slep_t)

        #! don't forget the decision of grasp!
        self.decision_flag = False
        cur_ee_pos = p.getLinkState(self.arm_id, 9)[0]
        cur_box_pos = p.getBasePositionAndOrientation(self.obj_idx[0])[0]


        grasp_decision = bool(abs(new_arm_obs[0] - new_box_pos[0]) < (self.x_high_obs - self.x_low_obs)/15 and
        abs(new_arm_obs[1] - new_box_pos[1]) < (self.y_high_obs - self.y_low_obs)/15 and
        (0.005 < new_arm_obs[2]-new_box_pos[2])< 0.03)

        self.gripper_flag = None
        if grasp_decision:
            self.decision_flag = True
            a_gripper = action[4]//0.5001
            # logger.info('choose to grasp')
            self.gripper_flag = self.gripper_control(a_gripper)
        else:
            p.setJointMotorControlArray(self.arm_id, [7, 8], p.POSITION_CONTROL, targetPositions=[0, 0])
            # logger.info('choose not to grasp')
            # self.gripper_flag = None


    def gripper_control(self, cmds):
        flag = False
        # open gripper
        if cmds == 0:
            logger.info(f'the distance is ok but the arm does not decide to grasp, the flag is {flag}')
            p.setJointMotorControlArray(self.arm_id, [7, 8], p.POSITION_CONTROL, targetPositions=[0, 0])
        elif cmds == 1:
            while True:
                cur_pos = np.asarray(p.getJointStates(self.arm_id, [7, 8]))[:, 0]
                tar_pos = np.add(cur_pos, [0.032/20, 0.032/20])
                # print(tar_pos)
                p.setJointMotorControlArray(self.arm_id, [7, 8], p.POSITION_CONTROL, targetPositions=tar_pos)

                for i in range(20):
                    p.stepSimulation()
                    time.sleep(self.slep_t)

                obs_pos = np.asarray(p.getJointStates(self.arm_id, [7, 8]))[:, 0]
                # print(obs_pos)
                if abs(obs_pos[1] - tar_pos[1]) > 0.0005:
                    if obs_pos[1] >= 0.03186:
                        break
                    flag = True
                    logger.info(f'get it, the flag is {flag}')
                    break
                else:
                    flag = False
                    logger.info(f'decided to grasp, not catch the box, the flag is {flag}')
                    break
        return flag

    def step(self, a):
        # self.act(a)

        #! sample some actions: x y z yaw
        self.act(a)

        obs = self.get_obs()

        r, done = self.reward_func(obs)

        self.step_counter += 1

        return obs, r, done, {}

    def reward_func(self, obs):

        self.ee_pos = p.getLinkState(self.arm_id, 9)[0]
        self.ee_ori = p.getEulerFromQuaternion(p.getLinkState(self.arm_id,9)[1])
        self.box_pos = p.getBasePositionAndOrientation(self.obj_idx[0])[0]
        self.box_ori = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.obj_idx[0])[1])
        x = self.ee_pos[0]
        y = self.ee_pos[1]
        z = self.ee_pos[2]
        ee_yaw = self.ee_ori[2]
        box_yaw = self.box_ori[2]

        ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        square_dx = (self.ee_pos[0] - self.ee_ori[0]) ** 2
        square_dy = (self.ee_pos[1] - self.ee_ori[1]) ** 2
        square_dz = (self.ee_pos[2] - self.ee_ori[2]) ** 2
        self.distance = math.sqrt(square_dx + square_dy + square_dz)

        self.terminated = False

        terminated = bool(x < self.x_low_obs or x > self.x_high_obs
                          or y < self.y_low_obs or y > self.y_high_obs
                          or z < self.z_low_obs or z > self.z_high_obs)
        
        top_decision = bool(abs(self.ee_pos[0] - self.box_pos[0]) < (self.x_high_obs - self.x_low_obs)/10 and
        abs(self.ee_pos[1] - self.box_pos[1]) < (self.y_high_obs - self.y_low_obs)/10 and
        (0.03 < self.ee_pos[2]-self.box_pos[2])< 0.15)
        
        if self.gripper_flag == False:
            r = -1
            self.terminated = True

        elif self.step_counter > self.kMaxEpisodeSteps:
            r = -0.1
            logger.info('times up')
            self.terminated = True

        elif terminated:
            r = -0.1
            logger.info('hit the border')
            self.terminated = True

        elif self.gripper_flag ==True:
            r = 10
            logger.info('get the box, the reward is 10!')
            self.terminated = True

        elif self.decision_flag == True:
            r = 1
            logger.info('locate the position of grasp')
            self.terminated = False
        
        # elif self.distance < 0.1:
        #     r = 0.0001
        #     logger.info('next to the box')
        #     self.terminated = False
        
        elif top_decision:
            r = 0.05
            logger.info('on the top of box')
            self.terminated = False

        elif abs(box_yaw - ee_yaw) < 0.05:
            r = 0.0001
            logger.debug('the yaw is same')
            self.terminated = False

        else:
            r = 0
            self.terminated = False
        
        self.r += r

        return self.r, self.terminated


    def get_obs(self):
        # Get end-effector obs
        self.ee_pos = p.getLinkState(self.arm_id, 9)[0]
        self.ee_ori = p.getEulerFromQuaternion(p.getLinkState(self.arm_id,9)[1])
        self.ee_pos, self.ee_ori = np.asarray(self.ee_pos), np.asarray(self.ee_ori)
        self.ee_pos.astype(np.float32)
        self.ee_ori.astype(np.float32)
        # logger.debug(self.ee_ori)

        # only one box!
        self.box_pos = p.getBasePositionAndOrientation(self.obj_idx[0])[0]
        self.box_ori = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.obj_idx[0])[1])
        self.box_pos, self.box_ori = np.asarray(self.box_pos), np.asarray(self.box_ori)
        self.box_pos.astype(np.float32)
        self.box_ori.astype(np.float32)
        # logger.debug(self.box_pos)

        Joint_info_list = p.getJointStates(self.arm_id, self.joints_index)
        self.joints_angle = []
        for i in range(len(Joint_info_list)):
            self.joints_angle.append(Joint_info_list[i][0])
        self.joints_angle = np.asarray(self.joints_angle)
        self.joints_angle[len(self.joints_angle)-2:] = abs(self.joints_angle[len(self.joints_angle)-2:])//0.01601
        self.joints_angle = self.joints_angle.astype(np.float32)
        
        obs = np.concatenate([self.ee_pos, self.ee_ori, self.box_pos, self.box_ori, self.joints_angle])
        logger.debug(f'the shape of obs is {obs.size}')
        obs = obs.astype(np.float32)
        return obs

    def get_image(self, gray_flag = False, resize_flag=False):
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

    env = Arm_env(is_render=True, num_objects=1)

    num_item = 2
    num_epoch = 3
    env.slep_t = 1 / 240
    num_step = 400

    Debug_para = []

    Debug_para.append(p.addUserDebugParameter("x" , 0, 1, 0))
    Debug_para.append(p.addUserDebugParameter("y" , 0, 1, 0.5))
    Debug_para.append(p.addUserDebugParameter("z" , 0, 1, 0))
    Debug_para.append(p.addUserDebugParameter("yaw" , 0, 1, 0.5))
    Debug_para.append(p.addUserDebugParameter("gripper" , 0, 1, 0.5))

    # 5 Joints and 1 End-effector
    for epoch in range(num_epoch):
        state = env.reset()
        epoch_r = 0

        for i_step in range(num_step):
            print(i_step)
            # a = np.random.uniform(0,1,size = 4)
            a = []
            # get parameters
            for j in range(5):
                a.append(p.readUserDebugParameter(Debug_para[j]))
            a = np.asarray(a)
            # a *= 0.5 * np.sin(i_step/np.pi)
            obs, r, done, _ = env.step(a)
            # print(obs)
            epoch_r += r
