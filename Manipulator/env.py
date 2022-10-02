import pybullet as p
import pybullet_data as pd
import os
import gym
import cv2
import numpy as np
import random
import math
import reorder

class Arm_env(gym.Env):

    def __init__(self, is_render=True):

        self.kImageSize = {'width': 480, 'height': 480}

        self.urdf_path = 'urdf'
        self.pybullet_path = pd.getDataPath()
        self.is_render = is_render

        self.x_low_obs = 0.15
        self.x_high_obs = 0.55
        self.y_low_obs = -0.2
        self.y_high_obs = 0.2
        self.z_low_obs = 0
        self.z_high_obs = 0.55
        self.friction = 0.99
        self.friction = 0.99

        # 5 6 9不用管，固定的！
        self.init_joint_positions = [0, 0, -1.57, 0, 0, 0, 0, 0, 0, 0]
        
        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
            
        self.camera_parameters = {
            'width': 960.,
            'height': 720,
            'fov': 69,
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
            pitch=-75,
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
                                     cameraYaw=0,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[0.55, -0.35, 0.4])

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

        p.loadURDF(os.path.join(self.pybullet_path, "plane.urdf"), basePosition=[0, 0, -0.65])
        self.arm_id = p.loadURDF(os.path.join(self.urdf_path, "robot_arm928/robot_arm1.urdf"), useFixedBase=True)
        table_id = p.loadURDF(os.path.join(self.pybullet_path, "table/table.urdf"), basePosition=[0, 0, -0.65])

        p.changeDynamics(table_id, -1, lateralFriction=self.friction,spinningFriction=0.02,rollingFriction=0.002)
        p.changeVisualShape(table_id, -1, rgbaColor=[1, 1, 1, 1])
        
        box1_pos = [random.uniform(self.x_low_obs, self.x_high_obs), random.uniform(self.y_low_obs, self.y_high_obs), 0.01]
        box2_pos = [random.uniform(self.x_low_obs, self.x_high_obs), random.uniform(self.y_low_obs, self.y_high_obs), 0.01]
        box3_pos = [random.uniform(self.x_low_obs, self.x_high_obs), random.uniform(self.y_low_obs, self.y_high_obs), 0.01]
        box4_pos = [random.uniform(self.x_low_obs, self.x_high_obs), random.uniform(self.y_low_obs, self.y_high_obs), 0.01]
        box1_ori = [0, 0, random.uniform(-math.pi/2, math.pi/2)]
        box2_ori = [0, 0, random.uniform(-math.pi/2, math.pi/2)]
        box3_ori = [0, 0, random.uniform(-math.pi/2, math.pi/2)]
        box4_ori = [0, 0, random.uniform(-math.pi/2, math.pi/2)]
        box1_qua = p.getQuaternionFromEuler(box1_ori)
        box2_qua = p.getQuaternionFromEuler(box2_ori)
        box3_qua = p.getQuaternionFromEuler(box3_ori)
        box4_qua = p.getQuaternionFromEuler(box4_ori)

        box1_id = p.loadURDF(os.path.join(self.urdf_path, "box/box1.urdf"), basePosition = box1_pos, baseOrientation = box1_qua)
        box2_id = p.loadURDF(os.path.join(self.urdf_path, "box/box2.urdf"), basePosition = box2_pos, baseOrientation = box2_qua)
        box3_id = p.loadURDF(os.path.join(self.urdf_path, "box/box3.urdf"), basePosition = box3_pos, baseOrientation = box3_qua)
        box4_id = p.loadURDF(os.path.join(self.urdf_path, "box/box4.urdf"), basePosition = box4_pos, baseOrientation = box4_qua)
        
        position = []
        box1_min, box1_max = p.getAABB(box1_id)
        box2_min, box2_max = p.getAABB(box2_id)
        box3_min, box3_max = p.getAABB(box3_id)
        box4_min, box4_max = p.getAABB(box4_id)
        position.append(list(box1_min + box1_max))
        position.append(list(box2_min + box2_max))
        position.append(list(box3_min + box3_max))
        position.append(list(box4_min + box4_max))
        orientation = []
        orientation.append(list(box1_ori))
        orientation.append(list(box2_ori))
        orientation.append(list(box3_ori))
        orientation.append(list(box4_ori))

        new_position, new_orientation = reorder.configuration(position, orientation)
        
        box1_qua = p.getQuaternionFromEuler(new_orientation[0])
        box2_qua = p.getQuaternionFromEuler(new_orientation[1])
        box3_qua = p.getQuaternionFromEuler(new_orientation[2])
        box4_qua = p.getQuaternionFromEuler(new_orientation[3])

        p.resetBasePositionAndOrientation(box1_id, new_position[0], box1_qua)
        p.resetBasePositionAndOrientation(box2_id, new_position[1], box2_qua)
        p.resetBasePositionAndOrientation(box3_id, new_position[2], box3_qua)
        p.resetBasePositionAndOrientation(box4_id, new_position[3], box4_qua)

        self.num_joints = p.getNumJoints(self.arm_id)

        # for zzz in range(p.getNumJoints(self.arm_id)):
        #     joint_id = zzz
        #     joint_info = p.getJointInfo(self.arm_id, joint_id)
        #     print(joint_info)

        for i in range(self.num_joints):
            p.resetJointState(
                self.arm_id,
                jointIndex=i,
                targetValue=self.init_joint_positions[i],
            )

        self.robot_pos_obs = p.getLinkState(self.arm_id, self.num_joints - 1)[4]

        # reset camera
        (width, length, px, _, _) = p.getCameraImage(width=960,
                                                    height=720,
                                                    viewMatrix=self.view_matrix,
                                                    projectionMatrix=self.projection_matrix,
                                                    renderer=p.ER_BULLET_HARDWARE_OPENGL)
        self.images = px
        # print(width, length)


        self.images = self.images[:, :, :3]

        p.setJointMotorControlArray(self.arm_id, [0, 1, 2, 3, 4, 7, 8], p.POSITION_CONTROL,
                                    targetPositions=[0, -np.pi / 2, np.pi / 2, 0, 0, 0, 0])

        # p.stepSimulation()
        
        return self._process_image(self.images)

    def act(self, a_pos):
        p.setJointMotorControlArray(self.arm_id, [0, 1, 2, 3, 4], p.POSITION_CONTROL, targetPositions=a_pos,
                                    targetVelocities=[1, 1, 1, 1, 1])


    def step(self, a):
        self.act(a)




    
    def _process_image(self, image):
        """Convert the RGB pic to gray pic and add a channel 1

        Args:
            image ([type]): [description]
        """

        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.resize(image, (self.kImageSize['width'], self.kImageSize['height']))[None, :, :] / 255.
            # print("process_image的图像")
            # print(image)
            return image
        else:
            return np.zeros((1, self.kImageSize['width'], self.kImageSize['height']))

if __name__ == '__main__':

    env = Arm_env(is_render=True)
    
    num_item = 2

    state = env.reset()
    while True:
        p.stepSimulation()
    # print(state.shape)
    # img = state[0] # 去除图像两个多余的维度
    # print(img)

    # plt.imshow(img, cmap='gray')
    # plt.show()