import time

import pybullet as p
import pybullet_data as pd
import os
import gym
# import cv2
import numpy as np
import random
import math


class Arm_env(gym.Env):

    def __init__(self, is_render=True, num_objects=1):

        self.kImageSize = {'width': 480, 'height': 480}

        self.urdf_path = 'urdf'
        self.pybullet_path = pd.getDataPath()
        self.is_render = is_render

        self.x_low_obs = 0
        self.x_high_obs = 0.3
        self.y_low_obs = -0.2
        self.y_high_obs = 0.2
        self.z_low_obs = 0.001
        self.z_high_obs = 0.55
        self.friction = 0.99
        self.num_objects = num_objects
        # self.action_space = np.asarray([np.pi/3, np.pi / 6, np.pi / 4, np.pi / 2, np.pi])
        # self.shift = np.asarray([-np.pi/6, -np.pi/12, 0, 0, 0])
        self.ik_space = np.asarray([0.3, 0.4, 0.06, np.pi]) # x, y, z, yaw
        self.ik_space_shift = np.asarray([0,-0.2,0, -np.pi/2])

        self.slep_t = 0
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
            cameraTargetPosition=[0.40, 0, 0.05],
            distance=0.40,
            yaw=90,
            pitch=-45,
            roll=0,
            upAxisIndex=2)
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_parameters['fov'],
            aspect=self.camera_parameters['width'] /
                   self.camera_parameters['height'],
            nearVal=self.camera_parameters['near'],
            farVal=self.camera_parameters['far'])

        p.configureDebugVisualizer(lightPosition=[5, 0, 5])
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                     cameraYaw=90,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[0.55, 0, 0.4])
        p.setAdditionalSearchPath(pd.getDataPath())

    def reset(self):

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

        # for i in range(num):
        #     exec('box{} = 1'.format(i))
        # print(box0)

        baseid = p.loadURDF("urdf/base.urdf", basePosition=[0, 0, -0.05], useFixedBase=1)
        self.arm_id = p.loadURDF(os.path.join(self.urdf_path, "robot_arm928/robot_arm1.urdf"),
                                 basePosition=[-.08, 0, 0.02], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        textureId = p.loadTexture("table/table.png")
        p.changeDynamics(baseid, -1, lateralFriction=self.friction, spinningFriction=0.02, rollingFriction=0.002)
        p.changeVisualShape(baseid, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=textureId)

        # Generate the pos and orin of objects randomly.
        obj_idx = []
        for i in range(self.num_objects):
            rdm_pos = [random.uniform(self.x_low_obs, self.x_high_obs), random.uniform(self.y_low_obs, self.y_high_obs),
                       0.01]
            rdm_ori = [0, 0, random.uniform(-math.pi / 2, math.pi / 2)]
            obj_idx.append(p.loadURDF(os.path.join(self.urdf_path, "box/box%d.urdf" % i), basePosition=rdm_pos,
                                      baseOrientation=p.getQuaternionFromEuler(rdm_ori)))

        # box1_min, box1_max = p.getAABB(box1_id)

        p.setJointMotorControlArray(self.arm_id, [0, 1, 2, 3, 4, 7, 8], p.POSITION_CONTROL,
                                    targetPositions=[0, -0.48627556248779596, 1.3546790099090924, 0.7016159753143177, 0, 0, 0],
                                    forces=[10] * 7)

        for _ in range(40):
            p.stepSimulation()
        return self.get_obs()

    def act(self, a_pos):
        a_pos[:4] = a_pos[:4]*self.ik_space+self.ik_space_shift
        # a_joint = a_pos[:5]  # * self.action_space + self.shift
        ik_angle = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=a_pos[:3], maxNumIterations=2000,
                                                targetOrientation=p.getQuaternionFromEuler([0, 1.57, a_pos[3]]))
        # Joint execution
        for i in range(5):
            p.setJointMotorControl2(self.arm_id, i, p.POSITION_CONTROL, targetPosition=ik_angle[i], force=10,
                                    maxVelocity=4)

        # Gripper execution
        a_gripper = a_pos[4]//0.5001
        p.setJointMotorControlArray(self.arm_id, [7, 8],
                                    p.POSITION_CONTROL,
                                    targetPositions=[a_gripper, a_gripper])

        for i in range(40):
            self.images = self.get_image()
            p.stepSimulation()
            time.sleep(self.slep_t)

    def step(self, a):
        self.act(a)

        obs = self.get_obs()

        r = 1

        done = False

        return obs, r, done, {}

    def get_obs(self):
        # Get end-effector obs
        self.ee_pos = p.getLinkState(self.arm_id, 9)[0]
        self.ee_ori = p.getEulerFromQuaternion(p.getLinkState(self.arm_id,9)[1])
        self.ee_pos, self.ee_ori = np.asarray(self.ee_pos), np.asarray(self.ee_ori)

        # for zzz in range(p.getNumJoints(self.arm_id)):
        #     joint_id = zzz
        #     joint_info = p.getJointInfo(self.arm_id, joint_id)
        #     print(joint_info)
        self.joints_angle = np.asarray(p.getJointStates(self.arm_id, self.joints_index))[:, 0]

        self.joints_angle[len(self.joints_angle)-2:] = abs(self.joints_angle[len(self.joints_angle)-2:])//0.01601
        obs = np.concatenate([self.ee_pos, self.ee_ori, self.joints_angle])

        return obs

    def get_image(self):
        # reset camera
        (width, length, px, _, _) = p.getCameraImage(width=960,
                                                     height=720,
                                                     viewMatrix=self.view_matrix,
                                                     projectionMatrix=self.projection_matrix,
                                                     renderer=p.ER_BULLET_HARDWARE_OPENGL)
        # print(width, length)
        return px

    # def _process_image(self, image):
    #     """Convert the RGB pic to gray pic and add a channel 1
    #
    #     Args:
    #         image ([type]): [description]
    #     """
    #
    #     if image is not None:
    #         image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #         image = cv2.resize(image, (self.kImageSize['width'], self.kImageSize['height']))[None, :, :] / 255.
    #         return image
    #     else:
    #         return np.zeros((1, self.kImageSize['width'], self.kImageSize['height']))


if __name__ == '__main__':

    env = Arm_env(is_render=True, num_objects=1)

    num_item = 2
    num_epoch = 3
    env.slep_t = 1 / 240
    num_step = 40

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
            print(obs)
            epoch_r += r
