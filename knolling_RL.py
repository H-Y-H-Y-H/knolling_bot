import numpy as np
import pyrealsense2 as rs
from items_real import sort
import pybullet_data as pd
import math
from turdf import *
import socket
import cv2
from cam_obs import *
import torch
import torch.nn as nn
import torch.nn.functional as F

# RL environment
from gym import spaces
from gym.utils import seeding
import gym
# RL environment

torch.manual_seed(42)
# np.random.seed(100)
# random.seed(100)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 3)

    def forward(self, x):
        # define forward pass
        x = self.fc1(x)
        # x = torch.sigmoid(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # nn.Dropout()
        x = self.fc4(x)
        return x

    def loss(self, pred, target):
        value = (pred - target) ** 2
        return torch.mean(value)

class Arm(gym.Env):

    def __init__(self, is_render=True):

        self.kImageSize = {'width': 480, 'height': 480}
        self.urdf_path = 'urdf'
        self.pybullet_path = pd.getDataPath()
        self.is_render = is_render
        if self.is_render:
            p.connect(p.GUI, options="--width=1280 --height=720")
        else:
            p.connect(p.DIRECT)

        self.num_motor = 5

        self.table_boundary = 0.05

        self.lateral_friction = 1
        self.spinning_friction = 1
        self.rolling_friction = 0

        self.correct = np.array([[0.015, 0.015, 0.012],
                                [0.023, 0.015, 0.012],
                                [0.033, 0.015, 0.012],
                                [0.01518, 0.09144, 0.01524]])

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
                                    aspect=self.camera_parameters['width'] / self.camera_parameters['height'],
                                    nearVal=self.camera_parameters['near'],
                                    farVal=self.camera_parameters['far'])

        p.configureDebugVisualizer(lightPosition=[5, 0, 5])
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                     cameraYaw=45,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[0.1, 0, 0])
        p.setAdditionalSearchPath(pd.getDataPath())

    def get_parameters(self, num_2x2=0, num_2x3=0, num_2x4=0, num_pencil=0,
                       total_offset=[0.1, 0, 0.006], grasp_order=[1, 0, 2],
                       gap_item=0.03, gap_block=0.02,
                       real_operate=False, obs_order='1', random_offset = False,
                       max_step = 10):

        self.num_2x2 = num_2x2
        self.num_2x3 = num_2x3
        self.num_2x4 = num_2x4
        self.num_pencil = num_pencil
        self.total_offset = total_offset
        self.grasp_order = grasp_order
        self.gap_item = gap_item
        self.gap_block = gap_block
        self.real_operate = real_operate
        self.obs_order = obs_order
        self.random_offset = random_offset
        self.num_list = np.array([self.num_2x2, self.num_2x3, self.num_2x4, self.num_pencil])
        self.num = self.num_2x2 + self.num_2x3 + self.num_2x4 + self.num_pencil

        # parameterrs of RL
        self.low_act = -np.ones(5)
        self.high_act = np.ones(5)
        self.action_space = spaces.Box(low=self.low_act, high=self.high_act, dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.ones(len(self.num_list) * 6 + 4) * np.inf, high=np.ones(len(self.num_list) * 6 + 4) * np.inf, dtype=np.float32)
        self.max_step = max_step
        self.low_scale = np.array([0.05, -0.15, 0.006, - np.pi / 2, 0])
        self.high_scale = np.array([0.25, 0.15, 0.05, np.pi / 2, 1])
        self.x_low_obs = self.low_scale[0]
        self.x_high_obs = self.high_scale[0]
        self.y_low_obs = self.low_scale[1]
        self.y_high_obs = self.high_scale[1]
        self.z_low_obs = self.low_scale[2]
        self.z_high_obs = self.high_scale[2]

    def get_obs(self, order):

        def get_rl_obs():

            ee_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
            ee_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))
            ee_ori = np.array(ee_ori[2])

            new_pos_before, new_ori_before = [], []
            # test_ori = []
            for i in range(len(self.obj_idx)):
                new_pos_before += p.getBasePositionAndOrientation(self.obj_idx[i])[0][:2] # select x and y in pos, discard z
                new_ori_before.append(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.obj_idx[i])[1])[2]) # select yaw in ori, discard roll and pitch
                test_ori.append(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.obj_idx[i])[1]))
            new_pos_before = np.asarray(new_pos_before)
            new_ori_before = np.asarray(new_ori_before)
            # test_ori = np.asarray(test_ori)

            # manipulator_before = np.concatenate((new_pos_before, new_ori_before), axis=1)
            new_xyz_list = self.xyz_list.reshape(-1)

            # print('this is ee pos', ee_pos)
            # print('this is ee ori', ee_ori)
            # print(test_ori)
            # print('this is pos', new_pos_before)
            # print('this is ori', new_ori_before)
            observation = np.concatenate((ee_pos, ee_ori, new_pos_before, new_ori_before, new_xyz_list), axis=None)
            # print('this is obs', observation)

            return observation

        def get_images():

            (width, length, image, _, _) = p.getCameraImage(width=640,
                                                            height=480,
                                                            viewMatrix=self.view_matrix,
                                                            projectionMatrix=self.projection_matrix,
                                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)
            return image

        def get_sim_image_obs():

            z = 0
            roll = 0
            pitch = 0
            (width, length, image, _, _) = p.getCameraImage(width=640,
                                                            height=480,
                                                            viewMatrix=self.view_matrix,
                                                            projectionMatrix=self.projection_matrix,
                                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)
            # # show in window
            # temp = np.copy(image[:, :, 0])
            # image[:, :, 0] = np.copy(image[:, :, 2])
            # image[:, :, 2] = temp
            # print(image.shape)
            # cv2.imshow("Comparison between Chaotic Configuration and Trim Configuration", image)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            img = image[:, :,:3]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cv2.imshow("Comparison between Chaotic Configuration and Trim Configuration", img)
            cv2.waitKey()
            cv2.destroyAllWindows()

            add = int((640-480)/2)
            img = cv2.copyMakeBorder(img, add, add, 0, 0,cv2.BORDER_CONSTANT, None, value = 0)
            results = np.asarray(detect(img))
            results = np.asarray(results[:,:5]).astype(np.float32)
            # structure: x,y,yaw,length,width

            index = []
            correct = []
            for i in range(len(self.grasp_order)):
                correct.append(self.xyz_list[self.all_index[i][0]])
            correct = np.asarray(correct)
            for i in range(len(correct)):
                for j in range(len(results)):
                    if np.linalg.norm(correct[i][:2] - results[j][3:5]) < 0.001:
                        index.append(j)
            manipulator_before = []
            for i in index:
                manipulator_before.append([results[i][0], results[i][1], z, roll, pitch, results[i][2]])
            manipulator_before = np.asarray(manipulator_before)
            new_xyz_list = self.xyz_list

            print('this is the result of dectection \n', results)
            print('this is manipulator before after the detection \n', manipulator_before)

            return manipulator_before, new_xyz_list

        def get_lego_obs():

            # sequence: pos before, ori before
            new_pos_before, new_ori_before = [], []
            for i in range(len(self.obj_idx)):
                new_pos_before.append(p.getBasePositionAndOrientation(self.obj_idx[i])[0])
                new_ori_before.append(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.obj_idx[i])[1]))

            manipulator_before = np.concatenate((new_pos_before, new_ori_before), axis=1)
            new_xyz_list = self.xyz_list

            return manipulator_before, new_xyz_list

        def get_real_image_obs():

            pipeline = rs.pipeline()
            config = rs.config()

            # Get device product line for setting a supporting resolution
            pipeline_wrapper = rs.pipeline_wrapper(pipeline)
            pipeline_profile = config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            device_product_line = str(device.get_info(rs.camera_info.product_line))

            found_rgb = False
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True
                    break
            if not found_rgb:
                print("The demo requires Depth camera with Color sensor")
                exit(0)

            # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

            if device_product_line == 'L500':
                config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
            else:
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
            # Start streaming
            pipeline.start(config)

            for _ in range(100):
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                # depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                color_image = np.asanyarray(color_frame.get_data())

                color_colormap_dim = color_image.shape
                resized_color_image = color_image

                # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                # cv2.imshow('RealSense', resized_color_image)
                # cv2.imwrite("img.png",resized_color_image[112:368, 192:448])
                add = int((640 - 480) / 2)
                resized_color_image = cv2.copyMakeBorder(resized_color_image, add, add, 0, 0, cv2.BORDER_CONSTANT,
                                                         None, value=0)
                cv2.imwrite("129_testpip.png", resized_color_image)

                cv2.waitKey(1)

            img = cv2.imread("129_testpip.png")

            manipulator_before = [i for i in range(len(self.manipulator_after) + 1)]
            while len(manipulator_before) != len(self.manipulator_after):
                print('we need to detect the num and kinds of lego bricks')
                results = np.asarray(detect(img))
                results = np.asarray(results[:, :5]).astype(np.float32)
                # results[:, 2] = results[:, 2] * math.pi / 180
                # structure: x,y,yaw,length,width

                all_index = []
                new_xyz_list = []
                kind = []
                new_results = []
                z = 0
                roll = 0
                pitch = 0
                num = 0
                for i in range(len(self.correct)):
                    kind_index = []
                    for j in range(len(results)):
                        if np.linalg.norm(self.correct[i][:2] - results[j][3:5]) < 0.003:
                            kind_index.append(num)
                            new_xyz_list.append(self.correct[i])
                            num += 1
                            if i in kind:
                                pass
                            else:
                                kind.append(i)
                            new_results.append(results[j])
                        else:
                            print('detect failed!!!')
                    if len(kind_index) != 0:
                        all_index.append(kind_index)
                new_xyz_list = np.asarray(new_xyz_list)
                new_results = np.asarray(new_results)
                print(new_results)
                print(all_index)

                manipulator_before = []
                for i in range(len(all_index)):
                    for j in range(len(all_index[i])):
                        manipulator_before.append([new_results[all_index[i][j]][0], new_results[all_index[i][j]][1], z, roll, pitch, new_results[all_index[i][j]][2]])
                manipulator_before = np.asarray(manipulator_before)

            print('this is the result of dectection \n', results)
            print('this is manipulator before after the detection \n', manipulator_before)

            return manipulator_before, new_xyz_list

        if order == 'sim_obj':
            manipulator_before, new_xyz_list = get_lego_obs()
            return manipulator_before, new_xyz_list
        elif order == 'sim_image_obj':
            manipulator_before, new_xyz_list = get_sim_image_obs()
            return manipulator_before, new_xyz_list
        elif order == 'rl_obj':
            obs = get_rl_obs()
            return obs
        elif order == 'images':
            image = get_images()
            return image
        elif order == 'real_image_obj':
            manipulator_before, new_xyz_list = get_real_image_obs()
            return manipulator_before, new_xyz_list

    def reset(self):

        self.step_counter = 0

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

        textureid = p.loadTexture(os.path.join(self.urdf_path, "table/table.png"))
        p.changeDynamics(baseid, -1, lateralFriction=1, spinningFriction=1, rollingFriction=0.002, linearDamping=0.5, angularDamping=0.5)
        p.changeDynamics(self.arm_id, 7, lateralFriction=1, spinningFriction=1, rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.arm_id, 8, lateralFriction=1, spinningFriction=1, rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeVisualShape(baseid, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=textureid)

        # get the standard xyz and corresponding index from files in the computer
        items_sort = sort()
        self.obj_idx = []
        if self.real_operate == False:
            self.xyz_list, _, _, self.all_index = items_sort.get_data_virtual(self.grasp_order, self.num_2x2,
                                                                              self.num_2x3, self.num_2x4,
                                                                              self.num_pencil)
            restrict = np.max(self.xyz_list)
            gripper_height = 0.012
            last_pos = np.array([[0, 0, 1]])
            for i in range(len(self.grasp_order)):
                for j in range(self.num_list[self.grasp_order[i]]):

                    rdm_pos = np.array([random.uniform(self.x_low_obs, self.x_high_obs),
                                        random.uniform(self.y_low_obs, self.y_high_obs), 0.006])
                    ori = [0, 0, random.uniform(-math.pi / 2, math.pi / 2)]
                    check_list = np.zeros(last_pos.shape[0])

                    while 0 in check_list:
                        rdm_pos = [random.uniform(self.x_low_obs, self.x_high_obs),
                                   random.uniform(self.y_low_obs, self.y_high_obs), 0.006]
                        for z in range(last_pos.shape[0]):
                            if np.linalg.norm(last_pos[z] - rdm_pos) < restrict + gripper_height:
                                check_list[z] = 0
                            else:
                                check_list[z] = 1

                    last_pos = np.append(last_pos, [rdm_pos], axis=0)
                    self.obj_idx.append(
                        p.loadURDF(os.path.join(self.urdf_path, f"item_{self.grasp_order[i]}/{j}.urdf"),
                                   basePosition=rdm_pos,
                                   baseOrientation=p.getQuaternionFromEuler(ori), useFixedBase=False,
                                   flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
        else:
            self.xyz_list, pos_before, ori_before, self.all_index, self.kind = items_sort.get_data_real()
            for i in range(len(self.kind)):
                for j in range(len(self.all_index[i])):
                    self.obj_idx.append(
                        p.loadURDF(os.path.join(self.urdf_path, f"item_{self.kind[i]}/{j}.urdf"),
                                   basePosition=pos_before[self.all_index[i][j]],
                                   baseOrientation=p.getQuaternionFromEuler(ori_before[self.all_index[i][j]]), useFixedBase=False,
                                   flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))

        print(self.obj_idx)
        for i in range(len(self.obj_idx)):
            p.changeDynamics(self.obj_idx[i], -1, restitution=30)

        # set the initial pos of the arm
        ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=[0, 0, 0.06],
                                                  maxNumIterations=200,
                                                  targetOrientation=p.getQuaternionFromEuler([0, math.pi / 2, 0]))
        for motor_index in range(self.num_motor):
            p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                    targetPosition=ik_angles0[motor_index], maxVelocity=7)
        p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL, targetPosition=0.025, force=10)
        p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL, targetPosition=0.025, force=10)

        for i in range(60):
            p.stepSimulation()
        return self.get_obs(self.obs_order)

    def change_config(self):  # this is main function!!!!!!!!!

        p.resetSimulation()
        p.setGravity(0, 0, -10)

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

        textureid = p.loadTexture(os.path.join(self.urdf_path, "table/table.png"))
        p.changeDynamics(baseid, -1, lateralFriction=self.lateral_friction, frictionAnchor=True)
        p.changeDynamics(self.arm_id, 7, lateralFriction=self.lateral_friction, frictionAnchor=True)
        p.changeDynamics(self.arm_id, 8, lateralFriction=self.lateral_friction, frictionAnchor=True)
        p.changeVisualShape(baseid, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=textureid)

        # get the standard xyz and corresponding index from files in the computer
        items_sort = sort()
        if self.real_operate == False:
            self.xyz_list, _, _, self.all_index = items_sort.get_data_virtual(self.grasp_order, self.num_2x2,
                                                                              self.num_2x3, self.num_2x4,
                                                                              self.num_pencil)
        else:
            self.xyz_list, _, _, self.all_index, self.kind = items_sort.get_data_real()
        print(f'this is standard trim xyz list\n {self.xyz_list}')
        print(f'this is standard trim index list\n {self.all_index}')

        def calculate_items(item_num, item_xyz):

            min_xy = np.ones(2) * 100
            best_item_config = []

            if item_num % 2 == 0:
                fac = []  # 定义一个列表存放因子
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

        def calculate_block():

            min_result = []
            best_config = []
            for i in range(len(self.all_index)):
                item_index = self.all_index[i]
                item_xyz = self.xyz_list[item_index, :]
                item_num = len(item_index)
                xy, config = calculate_items(item_num, item_xyz)
                print(f'this is min xy {xy}')
                min_result.append(list(xy))
                print(f'this is the best item config {config}')
                best_config.append(list(config))
            min_result = np.asarray(min_result).reshape(-1, 2)
            min_xy = np.copy(min_result)
            best_config = np.asarray(best_config).reshape(-1, 2)
            print(best_config)

            # 安排总的摆放
            iteration = 500
            all_num = best_config.shape[0]
            all_x = 100
            all_y = 100

            if all_num % 2 == 0:
                fac = []  # 定义一个列表存放因子
                for i in range(1, all_num + 1):
                    if all_num % i == 0:
                        fac.append(i)
                        continue
            else:
                fac = [1, all_num]

            for i in range(iteration):

                sequence = np.random.choice(best_config.shape[0], size=len(self.all_index), replace=False)
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

                        # insert 'whether to rotate' here
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

            return reorder_block(best_config, best_all_config, best_rotate_flag, best_min_xy)

        def reorder_item(best_config, start_pos, index_block, item_index, item_xyz, index_flag):

            # initiate the pos and ori
            # we don't analysis these imported oris
            # we directly define the ori is 0 or 90 degree, depending on the algorithm.
            item_pos = np.zeros([len(item_index), 3])
            item_ori = np.zeros([len(item_index), 3])
            # print(item_pos)
            num_2x2_row = best_config[index_block][0]
            num_2x2_column = best_config[index_block][1]
            index_2x2 = np.arange(item_pos.shape[0]).reshape(num_2x2_row, num_2x2_column)

            # the initial position of the first items

            if index_flag == True:

                temp = np.copy(item_xyz[:, 0])
                item_xyz[:, 0] = item_xyz[:, 1]
                item_xyz[:, 1] = temp
                item_ori[:, 2] = math.pi / 2
                # print(item_ori)
                temp = num_2x2_row
                num_2x2_row = num_2x2_column
                num_2x2_column = temp
                index_2x2 = index_2x2.transpose()
            else:

                item_ori[:, 2] = 0
                # print(item_ori)

            start_pos[0] = start_pos[0] + item_xyz[0][0] / 2
            start_pos[1] = start_pos[1] + item_xyz[0][1] / 2
            # print(f'this is try start {start_pos}')

            for j in range(num_2x2_row):
                for k in range(num_2x2_column):
                    x_2x2 = start_pos[0] + (item_xyz[index_2x2[j][k]][0]) * j + self.gap_item * j
                    y_2x2 = start_pos[1] + (item_xyz[index_2x2[j][k]][1]) * k + self.gap_item * k
                    item_pos[index_2x2[j][k]][0] = x_2x2
                    item_pos[index_2x2[j][k]][1] = y_2x2
            # print(item_pos)

            return item_ori, item_pos

        def reorder_block(best_config, best_all_config, best_rotate_flag, min_xy):

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
            # print(f'this is start_y {start_y}')d

            # determine the start position per item
            item_pos = np.zeros([len(self.xyz_list), 3])
            item_ori = np.zeros([len(self.xyz_list), 3])
            # print(self.xyz_list[self.all_index[0]])
            # print(self.all_index)
            for m in range(num_all_row):
                for n in range(num_all_column):
                    item_index = self.all_index[
                        best_all_config[m][n]]  # !!!!!!!!!!!!!!!!!!!!!determine the index of blocks
                    # print('try', item_index)
                    item_xyz = self.xyz_list[item_index, :]
                    # print('try', item_xyz)
                    start_pos = np.asarray([start_x[m], start_y[n]])
                    index_block = best_all_config[m][n]
                    index_flag = best_rotate_flag[m][n]

                    ori, pos = reorder_item(best_config, start_pos, index_block, item_index, item_xyz,
                                                    index_flag)
                    # print('tryori', ori)
                    # print('trypos', pos)
                    item_pos[item_index] = pos
                    item_ori[item_index] = ori

            return item_pos, item_ori  # pos_list, ori_list

        # determine the center of the tidy configuration
        self.items_pos_list, self.items_ori_list = calculate_block()
        x_low = np.min(self.items_pos_list, axis=0)[0]
        x_high = np.max(self.items_pos_list, axis=0)[0]
        y_low = np.min(self.items_pos_list, axis=0)[1]
        y_high = np.max(self.items_pos_list, axis=0)[1]
        center = np.array([(x_low + x_high) / 2, (y_low + y_high) / 2, 0])
        x_length = abs(x_high - x_low)
        y_length = abs(y_high - y_low)
        print(x_low, x_high, y_low, y_high)
        if self.random_offset == True:
            self.total_offset = np.array([random.uniform(self.x_low_obs + x_length / 2, self.x_high_obs - x_length / 2),
                                          random.uniform(self.y_low_obs + y_length / 2, self.y_high_obs - y_length / 2), 0.006])
        else:
            pass
        self.items_pos_list = self.items_pos_list + self.total_offset - center
        self.manipulator_after = np.concatenate((self.items_pos_list, self.items_ori_list), axis=1)
        print(self.manipulator_after)

        # import urdf and assign the trim pos & ori
        items_names = globals()
        self.obj_idx = []
        if self.real_operate == False:
            for i in range(len(self.grasp_order)):
                items_names[f'index_{self.grasp_order[i]}'] = self.all_index[i]
                items_names[f'num_{self.grasp_order[i]}'] = len(items_names[f'index_{self.grasp_order[i]}'])
                items_names[f'pos_{self.grasp_order[i]}'] = self.items_pos_list[items_names[f'index_{self.grasp_order[i]}'], :]
                items_names[f'ori_{self.grasp_order[i]}'] = self.items_ori_list[items_names[f'index_{self.grasp_order[i]}'], :]
                for j in range(self.num_list[self.grasp_order[i]]):
                    self.obj_idx.append(p.loadURDF(os.path.join(self.urdf_path, f"item_{self.grasp_order[i]}/{j}.urdf"),
                                                   basePosition=items_names[f'pos_{self.grasp_order[i]}'][j],
                                                   baseOrientation=p.getQuaternionFromEuler(items_names[f'ori_{self.grasp_order[i]}'][j]),
                                                   useFixedBase=True,
                                                   flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
        else:
            for i in range(len(self.kind)):
                # items_names[f'index_{self.grasp_order[i]}'] = self.all_index[i]
                # test!!
                index = self.all_index[i]
                pos = self.items_pos_list[index, :]
                ori = self.items_ori_list[index, :]
                for j in range(len(self.all_index[i])):
                    self.obj_idx.append(p.loadURDF(os.path.join(self.urdf_path, f"item_{self.kind[i]}/{j}.urdf"),
                                                   basePosition=pos[j],
                                                   baseOrientation=p.getQuaternionFromEuler(ori[j]),
                                                   useFixedBase=True,
                                                   flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
        # while 1:
        #     p.stepSimulation()
        return self.get_obs('images')

    def act(self, action_):

        def move(cur_pos, cur_ori, tar_pos, tar_ori):

            plot_cmd = []
            cmd_xyz = []
            cmd_ori = []

            if self.real_operate == True:
                if np.linalg.norm(tar_pos - np.array([0, 0, tar_pos[2]])) > 0.25:
                    tar_pos[0] = tar_pos[0] + 0.012
                    # cur_pos[0] = cur_pos[0] + 0.012
                    if tar_pos[1] < 0:
                        tar_pos[1] = tar_pos[1] - 0.004
                        # cur_pos[1] = cur_pos[1] - 0.008
                    else:
                        tar_pos[1] = tar_pos[1] + 0.004
                        # cur_pos[1] = cur_pos[1] + 0.008
                    print('add far')
                elif 0.20 < np.linalg.norm(tar_pos - np.array([0, 0, tar_pos[2]])) <= 0.25:
                    tar_pos[0] = tar_pos[0] + 0.008
                    # cur_pos[0] = cur_pos[0] + 0.008
                    if tar_pos[1] < 0:
                        tar_pos[1] = tar_pos[1] - 0.002
                        # cur_pos[1] = cur_pos[1] - 0.003
                    else:
                        tar_pos[1] = tar_pos[1] + 0.002
                        # cur_pos[1] = cur_pos[1] + 0.003
                    print('add medium far')
                elif 0.17 < np.linalg.norm(tar_pos - np.array([0, 0, tar_pos[2]])) <= 0.20:
                    tar_pos[0] = tar_pos[0] + 0.005
                    # cur_pos[0] = cur_pos[0] + 0.008
                    if tar_pos[1] < 0:
                        tar_pos[1] = tar_pos[1] - 0.002
                        # cur_pos[1] = cur_pos[1] - 0.003
                    else:
                        tar_pos[1] = tar_pos[1] + 0.002
                        # cur_pos[1] = cur_pos[1] + 0.003
                    print('add medium')
                elif 0.12 < np.linalg.norm(tar_pos - np.array([0, 0, tar_pos[2]])) <= 0.17:
                    tar_pos[0] = tar_pos[0] - 0.003
                    # cur_pos[0] = cur_pos[0] - 0.003
                    if tar_pos[1] < 0:
                        tar_pos[1] = tar_pos[1] - 0.002
                        # cur_pos[1] = cur_pos[1] - 0.003
                    else:
                        tar_pos[1] = tar_pos[1] + 0.002
                    print('add close')
            else:
                sim_height = -0.008

            tar_ori = np.array([0, np.pi / 2, tar_ori])
            if tar_ori[2] > 1.58:
                tar_ori[2] = tar_ori[2] - np.pi
            elif tar_ori[2] < -1.58:
                tar_ori[2] = tar_ori[2] + np.pi

            current_pos = np.copy(cur_pos)
            current_ori = np.copy(cur_ori)
            target_pos = np.copy(tar_pos)
            target_ori = np.copy(tar_ori)

            if abs(cur_pos[0] - tar_pos[0]) < 0.001 and abs(cur_pos[1] - tar_pos[1]) < 0.001:
                # vertical, choose a small slice
                move_slice = 0.006
            else:
                # horizontal, choose a large slice
                move_slice = 0.005
            distance = np.linalg.norm(tar_pos - cur_pos)
            num_step = np.ceil(distance / move_slice)

            step_pos = (target_pos - cur_pos) / num_step
            step_ori = (target_ori - cur_ori) / num_step

            if self.real_operate == True:
                while True:
                    tar_pos = cur_pos + step_pos
                    tar_ori = cur_ori + step_ori
                    cmd_xyz.append(tar_pos)
                    cmd_ori.append(tar_ori)

                    ik_angles_sim = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=tar_pos + np.array([0, 0, sim_height]),
                                                              maxNumIterations=200,
                                                              targetOrientation=p.getQuaternionFromEuler(tar_ori))

                    for motor_index in range(self.num_motor):
                        p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                                targetPosition=ik_angles_sim[motor_index], maxVelocity=2.5)
                    for i in range(20):
                        p.stepSimulation()

                    if abs(target_pos[0] - tar_pos[0]) < 0.001 and abs(target_pos[1] - tar_pos[1]) < 0.001 and abs(target_pos[2] - tar_pos[2]) < 0.001 and \
                            abs(target_ori[0] - tar_ori[0]) < 0.001 and abs(target_ori[1] - tar_ori[1]) < 0.001 and abs(target_ori[2] - tar_ori[2]) < 0.001:
                        break
                    cur_pos = tar_pos
                    cur_ori = tar_ori

                ######################### nn offset #######################
                cmd_xyz = np.asarray(cmd_xyz)
                cmd_ori = np.asarray(cmd_ori)
                new_cmd_xyz = Cartesian_offset_nn(cmd_xyz)
                print('this is tar', target_pos)
                print('this is cur', current_pos)
                if abs(target_pos[2] - current_pos[2]) > 0.02 and abs(target_pos[0] - current_pos[0]) < 0.0005 and abs(target_pos[1] - current_pos[1]) < 0.0005:
                    new_cmd_xyz[:, :2] = cmd_xyz[:, :2]
                    print('vertical grasp!!!')
                else:
                    new_cmd_xyz[:, 0] = cmd_xyz[:, 0]
                    print('horizontal grasp!!!')
                ######################### nn offset #######################

                for i in range(len(new_cmd_xyz)):
                    ik_angles_real = p.calculateInverseKinematics(self.arm_id, 9,
                                                                  targetPosition=new_cmd_xyz[i] + np.array([0, 0, real_height]),
                                                                  maxNumIterations=200,
                                                                  targetOrientation=p.getQuaternionFromEuler(
                                                                      cmd_ori[i]))

                    # print('this is motor angle', ik_angles_real)
                    angle_real = np.asarray(real_cmd2tarpos(rad2cmd(ik_angles_real[0:5])), dtype=np.float32)
                    plot_cmd.append(angle_real)

                plot_step = np.arange(num_step)
                plot_cmd = np.asarray(plot_cmd)
                print('this is the shape of cmd', plot_cmd.shape)
                print(len(plot_cmd.reshape(-1)))
                test_bytes = plot_cmd.tobytes()
                conn.sendall(plot_cmd.tobytes())
                # print('waiting the manipulator')
                plot_real = conn.recv(4096)
                # print('received')
                plot_real = np.frombuffer(plot_real, dtype=np.float32)
                plot_real = plot_real.reshape(-1, 6)
                with open(file="Cartisian_data/cmd.txt", mode="a", encoding="utf-8") as f:
                    plot_cmd = plot_cmd.tolist()
                    # print(plot_cmd)
                    for i in range(len(plot_cmd)):
                        list_zzz = [str(j) for j in plot_cmd[i]]
                        f.writelines(' '.join(list_zzz))
                        f.write('\n')
                with open(file="Cartisian_data/real.txt", mode="a", encoding="utf-8") as f:
                    plot_real = plot_real.tolist()
                    for i in range(len(plot_real)):
                        list_zzz = [str(j) for j in plot_real[i]]
                        f.writelines(' '.join(list_zzz))
                        f.write('\n')
                with open(file="Cartisian_data/step.txt", mode="a", encoding="utf-8") as f:
                    list_zzz = [str(j) for j in plot_step]
                    f.writelines(' '.join(list_zzz))
                    f.write('\n')

            else:
                while True:
                    tar_pos = cur_pos + step_pos
                    tar_ori = cur_ori + step_ori
                    ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=tar_pos + np.array([0, 0, sim_height]),
                                                              maxNumIterations=200,
                                                              targetOrientation=p.getQuaternionFromEuler(tar_ori))
                    for motor_index in range(self.num_motor):
                        p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                                targetPosition=ik_angles0[motor_index], maxVelocity=2.5)
                    for i in range(30):
                        p.stepSimulation()
                        # time.sleep(1 / 240)
                    if abs(target_pos[0] - tar_pos[0]) < 0.001 and abs(target_pos[1] - tar_pos[1]) < 0.001 and abs(
                            target_pos[2] - tar_pos[2]) < 0.001 and \
                            abs(target_ori[0] - tar_ori[0]) < 0.001 and abs(target_ori[1] - tar_ori[1]) < 0.001 and abs(
                        target_ori[2] - tar_ori[2]) < 0.001:
                        break
                    cur_pos = tar_pos
                    cur_ori = tar_ori

        def gripper(gap):
            if self.real_operate == True:
                if gap > 0.0265:
                    pos_real = np.asarray([[gap, gap]], dtype=np.float32)
                elif gap <= 0.0265:
                    pos_real = np.asarray([[0, 0]], dtype=np.float32)
                # print('gripper', pos_real)
                conn.sendall(pos_real.tobytes())
                # print(f'this is the cmd pos {pos_real}')
                p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL, targetPosition=gap, force=10)
                p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL, targetPosition=gap, force=10)

                real_pos = conn.recv(4096)
                real_pos = np.frombuffer(real_pos, dtype=np.float32)

            else:
                p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL, targetPosition=gap, force=10)
                p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL, targetPosition=gap, force=10)
            for i in range(30):
                p.stepSimulation()
                time.sleep(1 / 120)

        last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
        last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))
        action_ = (action_ - self.low_act) / (self.high_act - self.low_act) * (
                    self.high_scale - self.low_scale) + self.low_scale
        print('this is action', action_) # x, y, z, yaw, grasp?

        # self.ik_angle = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=action_[:3], maxNumIterations=3000,
        #                                              lowerLimits=self.ik_low, upperLimits=self.ik_high,
        #                                              jointRanges=[3.14, 3.14, 3.14, 3.14, 3.14, 20, 20],
        #                                              restPoses=[0, -0.48627556248779596, 1.1546790099090924,
        #                                                         0.7016159753143177, 0, 0, 0],
        #                                              targetOrientation=p.getQuaternionFromEuler([0, 1.57, action_[3]]))
        if action_[4] <= 0.7:
            print('move')
            move(last_pos, last_ori, action_[:3], action_[3])
        else:
            print('grasp')
            gripper(0.0273)

    def reward_func(self, observation, action):

        ee_pos = observation[:3]
        ee_ori = np.array([0, np.pi / 2, observation[3]])
        obj_pos = observation[4:4 + 2 * self.num].reshape(-1, 2)
        obj_pos = np.concatenate((obj_pos, np.zeros((self.num, 1))), axis=1)
        obj_ori = observation[4 + 2 * self.num:4 + 3 * self.num].reshape(-1, 1)
        obj_ori = np.concatenate((np.zeros((self.num, 2)), obj_ori), axis=1)
        obj_xyz = observation[4 + 3 * self.num:].reshape(-1, 3)

        print('this is the shape of obs', observation.shape)
        print('this is ee pos and ori\n', ee_pos, '\n', ee_ori)
        print('this is obj pos and ori\n', obj_pos, '\n', obj_ori)
        print('this is obj xyz\n', obj_xyz)

        # if np.linalg.norm()

        self.terminated = False
        r = 1

        return r, self.terminated

    def planning(self, order, conn, real_height, sim_height, evaluation):

        def get_start_end():  # generating all trajectories of all items in normal condition
            z = 0
            roll = 0
            pitch = 0
            manipulator_before, new_xyz_list = self.get_obs(self.obs_order)

            # sequence pos_before, ori_before, pos_after, ori_after
            start_end = []
            for i in range(len(self.obj_idx)):
                start_end.append([manipulator_before[i][0], manipulator_before[i][1], z, roll, pitch, manipulator_before[i][5],
                                  self.manipulator_after[i][0], self.manipulator_after[i][1], z, roll, pitch, self.manipulator_after[i][5]])
            start_end = np.asarray((start_end))
            return start_end

        def Cartesian_offset_nn(xyz_input):

            input_sc = [[-0.01, -0.201, -0.01],
                        [0.30, 0.201, 0.0601]]
            output_sc = [[-0.01, -0.201, -0.01],
                        [0.30, 0.201, 0.0601]]

            scaler_output = MinMaxScaler()
            scaler_input = MinMaxScaler()
            scaler_output.fit(output_sc)
            scaler_input.fit(input_sc)

            # model = Net().to(device)
            # model.load_state_dict(torch.load("nn_calibration/model_pt_xyz/combine_all003005_free_far015025003005_free.pt"))
            # # print(model)
            # model.eval()
            with torch.no_grad():
                xyz_input_scaled = scaler_input.transform(xyz_input).astype(np.float32)
                xyz_input_scaled = torch.from_numpy(xyz_input_scaled)
                xyz_input_scaled = xyz_input_scaled.to(device)
                pred_xyz = model.forward(xyz_input_scaled)
                # print(pred_angle)
                pred_xyz = pred_xyz.cpu().data.numpy()
                xyz_output = scaler_output.inverse_transform(pred_xyz)
                # # !!!!!!!!!!!!!!!!!!!! unify the differences!!!!!!!!!!!!!!!!!!!!!!
                # if split_flag == True:
                #     angle_output = np.insert(angle_output, 0, values=cmd[:, 0], axis=1)
                #     angle_output = np.column_stack((angle_output, cmd[:, 5]))
                #     angle_output[:, 1] = angle_output[:, 2]
                # else:
                #     angle_output[:, 1] = angle_output[:, 2]
                # # !!!!!!!!!!!!!!!!!!!! unify the differences!!!!!!!!!!!!!!!!!!!!!!

            # for i in range(len(xyz_input[0])):
            #     plt.subplot(2, 3, i+1)
            #     plt.plot(np.arange(len(xyz_input)), xyz_output[:, i], label='output cmd')
            #     plt.plot(np.arange(len(xyz_input)), xyz_input[:, i], label='input cmd')
            #     plt.title(f'motor{i}')
            #     plt.legend()
            # plt.suptitle("Comparison of output and input")
            # plt.show()

            return xyz_output

        def move(cur_pos, cur_ori, tar_pos, tar_ori):

            plot_cmd = []
            cmd_xyz = []
            cmd_ori = []

            if self.real_operate == True:
                if np.linalg.norm(tar_pos - np.array([0, 0, tar_pos[2]])) > 0.25:
                    tar_pos[0] = tar_pos[0] + 0.012
                    # cur_pos[0] = cur_pos[0] + 0.012
                    if tar_pos[1] < 0:
                        tar_pos[1] = tar_pos[1] - 0.004
                        # cur_pos[1] = cur_pos[1] - 0.008
                    else:
                        tar_pos[1] = tar_pos[1] + 0.004
                        # cur_pos[1] = cur_pos[1] + 0.008
                    print('add far')
                elif 0.20 < np.linalg.norm(tar_pos - np.array([0, 0, tar_pos[2]])) <= 0.25:
                    tar_pos[0] = tar_pos[0] + 0.008
                    # cur_pos[0] = cur_pos[0] + 0.008
                    if tar_pos[1] < 0:
                        tar_pos[1] = tar_pos[1] - 0.002
                        # cur_pos[1] = cur_pos[1] - 0.003
                    else:
                        tar_pos[1] = tar_pos[1] + 0.002
                        # cur_pos[1] = cur_pos[1] + 0.003
                    print('add medium far')
                elif 0.17 < np.linalg.norm(tar_pos - np.array([0, 0, tar_pos[2]])) <= 0.20:
                    tar_pos[0] = tar_pos[0] + 0.005
                    # cur_pos[0] = cur_pos[0] + 0.008
                    if tar_pos[1] < 0:
                        tar_pos[1] = tar_pos[1] - 0.002
                        # cur_pos[1] = cur_pos[1] - 0.003
                    else:
                        tar_pos[1] = tar_pos[1] + 0.002
                        # cur_pos[1] = cur_pos[1] + 0.003
                    print('add medium')
                elif 0.12 < np.linalg.norm(tar_pos - np.array([0, 0, tar_pos[2]])) <= 0.17:
                    tar_pos[0] = tar_pos[0] - 0.003
                    # cur_pos[0] = cur_pos[0] - 0.003
                    if tar_pos[1] < 0:
                        tar_pos[1] = tar_pos[1] - 0.002
                        # cur_pos[1] = cur_pos[1] - 0.003
                    else:
                        tar_pos[1] = tar_pos[1] + 0.002
                    print('add close')
            else:
                pass

            if tar_ori[2] > 1.58:
                tar_ori[2] = tar_ori[2] - np.pi
            elif tar_ori[2] < -1.58:
                tar_ori[2] = tar_ori[2] + np.pi

            current_pos = np.copy(cur_pos)
            current_ori = np.copy(cur_ori)
            target_pos = np.copy(tar_pos)
            target_ori = np.copy(tar_ori)

            if abs(cur_pos[0] - tar_pos[0]) < 0.001 and abs(cur_pos[1] - tar_pos[1]) < 0.001:
                # vertical, choose a small slice
                move_slice = 0.006
            else:
                # horizontal, choose a large slice
                move_slice = 0.005
            distance = np.linalg.norm(tar_pos - cur_pos)
            num_step = np.ceil(distance / move_slice)

            step_pos = (target_pos - cur_pos) / num_step
            step_ori = (target_ori - cur_ori) / num_step

            if self.real_operate == True:
                while True:
                    tar_pos = cur_pos + step_pos
                    tar_ori = cur_ori + step_ori
                    cmd_xyz.append(tar_pos)
                    cmd_ori.append(tar_ori)

                    ik_angles_sim = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=tar_pos + np.array([0, 0, sim_height]),
                                                              maxNumIterations=200,
                                                              targetOrientation=p.getQuaternionFromEuler(tar_ori))

                    for motor_index in range(self.num_motor):
                        p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                                targetPosition=ik_angles_sim[motor_index], maxVelocity=2.5)
                    for i in range(20):
                        p.stepSimulation()

                    if abs(target_pos[0] - tar_pos[0]) < 0.001 and abs(target_pos[1] - tar_pos[1]) < 0.001 and abs(target_pos[2] - tar_pos[2]) < 0.001 and \
                            abs(target_ori[0] - tar_ori[0]) < 0.001 and abs(target_ori[1] - tar_ori[1]) < 0.001 and abs(target_ori[2] - tar_ori[2]) < 0.001:
                        break
                    cur_pos = tar_pos
                    cur_ori = tar_ori

                ######################### nn offset #######################
                cmd_xyz = np.asarray(cmd_xyz)
                cmd_ori = np.asarray(cmd_ori)
                new_cmd_xyz = Cartesian_offset_nn(cmd_xyz)
                print('this is tar', target_pos)
                print('this is cur', current_pos)
                if abs(target_pos[2] - current_pos[2]) > 0.02 and abs(target_pos[0] - current_pos[0]) < 0.0005 and abs(target_pos[1] - current_pos[1]) < 0.0005:
                    new_cmd_xyz[:, :2] = cmd_xyz[:, :2]
                    print('vertical grasp!!!')
                else:
                    new_cmd_xyz[:, 0] = cmd_xyz[:, 0]
                    print('horizontal grasp!!!')
                ######################### nn offset #######################

                for i in range(len(new_cmd_xyz)):
                    ik_angles_real = p.calculateInverseKinematics(self.arm_id, 9,
                                                                  targetPosition=new_cmd_xyz[i] + np.array([0, 0, real_height]),
                                                                  maxNumIterations=200,
                                                                  targetOrientation=p.getQuaternionFromEuler(
                                                                      cmd_ori[i]))

                    # print('this is motor angle', ik_angles_real)
                    angle_real = np.asarray(real_cmd2tarpos(rad2cmd(ik_angles_real[0:5])), dtype=np.float32)
                    plot_cmd.append(angle_real)

                plot_step = np.arange(num_step)
                plot_cmd = np.asarray(plot_cmd)
                print('this is the shape of cmd', plot_cmd.shape)
                print(len(plot_cmd.reshape(-1)))
                test_bytes = plot_cmd.tobytes()
                conn.sendall(plot_cmd.tobytes())
                # print('waiting the manipulator')
                plot_real = conn.recv(4096)
                # print('received')
                plot_real = np.frombuffer(plot_real, dtype=np.float32)
                plot_real = plot_real.reshape(-1, 6)
                with open(file="Cartisian_data/cmd.txt", mode="a", encoding="utf-8") as f:
                    plot_cmd = plot_cmd.tolist()
                    # print(plot_cmd)
                    for i in range(len(plot_cmd)):
                        list_zzz = [str(j) for j in plot_cmd[i]]
                        f.writelines(' '.join(list_zzz))
                        f.write('\n')
                with open(file="Cartisian_data/real.txt", mode="a", encoding="utf-8") as f:
                    plot_real = plot_real.tolist()
                    for i in range(len(plot_real)):
                        list_zzz = [str(j) for j in plot_real[i]]
                        f.writelines(' '.join(list_zzz))
                        f.write('\n')
                with open(file="Cartisian_data/step.txt", mode="a", encoding="utf-8") as f:
                    list_zzz = [str(j) for j in plot_step]
                    f.writelines(' '.join(list_zzz))
                    f.write('\n')

            else:
                while True:
                    tar_pos = cur_pos + step_pos
                    tar_ori = cur_ori + step_ori
                    ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=tar_pos + np.array([0, 0, sim_height]),
                                                              maxNumIterations=200,
                                                              targetOrientation=p.getQuaternionFromEuler(tar_ori))
                    for motor_index in range(self.num_motor):
                        p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                                targetPosition=ik_angles0[motor_index], maxVelocity=2.5)
                    for i in range(30):
                        p.stepSimulation()
                        # time.sleep(1 / 240)
                    if abs(target_pos[0] - tar_pos[0]) < 0.001 and abs(target_pos[1] - tar_pos[1]) < 0.001 and abs(
                            target_pos[2] - tar_pos[2]) < 0.001 and \
                            abs(target_ori[0] - tar_ori[0]) < 0.001 and abs(target_ori[1] - tar_ori[1]) < 0.001 and abs(
                        target_ori[2] - tar_ori[2]) < 0.001:
                        break
                    cur_pos = tar_pos
                    cur_ori = tar_ori

        def gripper(gap):
            if self.real_operate == True:
                if gap > 0.0265:
                    pos_real = np.asarray([[gap, gap]], dtype=np.float32)
                elif gap <= 0.0265:
                    pos_real = np.asarray([[0, 0]], dtype=np.float32)
                # print('gripper', pos_real)
                conn.sendall(pos_real.tobytes())
                # print(f'this is the cmd pos {pos_real}')
                p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL, targetPosition=gap, force=10)
                p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL, targetPosition=gap, force=10)

                real_pos = conn.recv(4096)
                real_pos = np.frombuffer(real_pos, dtype=np.float32)

            else:
                p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL, targetPosition=gap, force=10)
                p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL, targetPosition=gap, force=10)
            for i in range(30):
                p.stepSimulation()
                # time.sleep(1 / 120)

        def knolling():

            start_end = get_start_end()

            rest_pos = np.array([0, 0, 0.05])
            rest_ori = np.array([0, 1.57, 0])
            offset_low = np.array([0, 0, 0])
            offset_high = np.array([0, 0, 0.04])

            for i in range(len(self.obj_idx)):

                trajectory_pos_list = [[0.025],
                                       offset_high + start_end[i][:3],
                                       offset_low + start_end[i][:3],
                                       [0.0273],
                                       offset_high + start_end[i][:3],
                                       offset_high + start_end[i][6:9],
                                       offset_low + start_end[i][6:9],
                                       [0.025],
                                       offset_high + start_end[i][6:9]]

                trajectory_ori_list = [rest_ori + start_end[i][3:6],
                                       rest_ori + start_end[i][3:6],
                                       rest_ori + start_end[i][3:6],
                                       [0.0273],
                                       rest_ori + start_end[i][3:6],
                                       rest_ori + start_end[i][9:12],
                                       rest_ori + start_end[i][9:12],
                                       [0.025],
                                       rest_ori + start_end[i][9:12]]
                last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
                last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))
                for j in range(len(trajectory_pos_list)):

                    if len(trajectory_pos_list[j]) == 3:
                        move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j])
                        last_pos = np.copy(trajectory_pos_list[j])
                        last_ori = np.copy(trajectory_ori_list[j])

                    elif len(trajectory_pos_list[j]) == 1:
                        gripper(trajectory_pos_list[j][0])

            # back to the reset pos and ori
            ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=[0, 0, 0.06],
                                                      maxNumIterations=200,
                                                      targetOrientation=p.getQuaternionFromEuler(
                                                          [0, math.pi / 2, 0]))
            for motor_index in range(self.num_motor):
                p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                        targetPosition=ik_angles0[motor_index], maxVelocity=7)
            for i in range(80):
                p.stepSimulation()
                # self.images = self.get_image()
                # time.sleep(1 / 120)

        def check_accuracy_sim():
            manipulator_before, new_xyz_list = self.get_obs(self.obs_order)
            print('this is before', manipulator_before)
            print('this is after', self.manipulator_after)
            print('this is new xyz', new_xyz_list)
            print('this is self xyz', self.xyz_list)
            # all_distance_sim = 0
            # for i in range(len(manipulator_before)):
            #     all_distance_sim += np.linalg.norm(manipulator_before[i][:3] - self.manipulator_after[i][:3])
            #     if np.linalg.norm(manipulator_before[i][:3] - self.manipulator_after[i][:3]) > 0.01:
            #         print(manipulator_before[i][:3])
            #         print(self.manipulator_after[i][:3])
            #         image_error = self.get_obs('images')
            #         print('error happened when checking!')
            #         temp = np.copy(image_error[:, :, 0])
            #         image_error[:, :, 0] = np.copy(image_error[:, :, 2])
            #         image_error[:, :, 2] = temp
            #         cv2.imwrite(f'./Error_images/error_{evaluation}.png', image_error)
            #         break

            error_flag = False
            for i in range(len(self.manipulator_fter)):
                for j in range(len(manipulator_before)):
                    if np.linalg.norm(self.manipulator_after[i][:3] - manipulator_before[j][:3]) < 0.015 and np.linalg.norm(self.xyz_list[i] - new_xyz_list[j]) < 0.002:
                        error_flag = False
                        print('find it')
                        break
                    else:
                        error_flag = True
            if error_flag == True:
                print('Error!!!')
            print('this is all distance between messy and neat in simulation environment')

        def check_accuracy_real():
            manipulator_before, new_xyz_list = self.get_obs(self.obs_order)
            all_distance_real = 0
            for i in range(len(manipulator_before)):
                all_distance_real += np.linalg.norm(manipulator_before[i][:3] - self.manipulator_after[i][:3])
            print('this is all distance between messy and neat in real world')

        if order == 1:
            # clean_desk()
            pass
        elif order == 2:
            # clean_item()
            pass
        elif order == 3:
            knolling()
        elif order == 4:
            check_accuracy_sim()

    def step(self, action_):

        ########################################## Hard Code #############################################

        if self.real_operate == True:

            with open(file="Cartisian_data/cmd.txt", mode="w", encoding="utf-8") as f:
                f.truncate(0)
            with open(file="Cartisian_data/real.txt", mode="w", encoding="utf-8") as f:
                f.truncate(0)
            with open(file="Cartisian_data/step.txt", mode="w", encoding="utf-8") as f:
                f.truncate(0)

            HOST = "192.168.0.186"  # Standard loopback interface address (localhost)
            PORT = 8880  # Port to listen on (non-privileged ports are > 1023)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((HOST, PORT))
            # It should be an integer from 1 to 65535, as 0 is reserved. Some systems may require superuser privileges if the port number is less than 4096.
            # associate the socket with a specific network interface
            s.listen()
            print(f"Waiting for connection...\n")
            conn, addr = s.accept()
            print(conn)
            print(f"Connected by {addr}")
            table_surface_height = -0.007
            sim_table_surface_height = -0.005
            num_motor = 5
            # ! reset the pos in both real and sim
            reset_pos = [0.015, 0, 0.1]
            ik_angles = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=reset_pos,
                                                     maxNumIterations=300,
                                                     targetOrientation=p.getQuaternionFromEuler(
                                                         [0, 1.57, 0]))
            reset_real = np.asarray(real_cmd2tarpos(rad2cmd(ik_angles[0:5])), dtype=np.float32)
            conn.sendall(reset_real.tobytes())

            for i in range(num_motor):
                p.setJointMotorControl2(self.arm_id, i, p.POSITION_CONTROL, targetPosition=ik_angles[i],
                                        maxVelocity=3)
            for _ in range(200):
                p.stepSimulation()
                time.sleep(1 / 12)
        else:
            conn = None
            table_surface_height = -0.007
            sim_table_surface_height = -0.008

        # self.planning(1, conn, table_surface_height, sim_table_surface_height, evaluation)
        # self.planning(2, conn, table_surface_height, sim_table_surface_height, evaluation)
        # self.planning(3, conn, table_surface_height, sim_table_surface_height, evaluation)
        # self.planning(4, conn, table_surface_height, sim_table_surface_height, evaluation)

        if self.real_operate == True:
            end = np.array([0], dtype=np.float32)
            conn.sendall(end.tobytes())

        ########################################## Hard Code #############################################

        self.act(action_)
        obs = self.get_obs(self.obs_order)
        r, done = self.reward_func(obs, action_)

        self.step_counter += 1
        if self.step_counter >= self.max_step:
            print('done!!!!!!!!!!!')
            done = True

        return obs, r, done, {}

if __name__ == '__main__':

    random_flag = True
    command = 'train'  # image virtual real

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("Device:", device)
    model = Net().to(device)
    model.load_state_dict(torch.load("Test_and_Calibration/nn_calibration/model_pt_xyz/combine_all_free_001005_far_free_001005_flank_free_001005_useful.pt"))
    # print(model)
    model.eval()

    if command == 'image':
        num_2x2 = 2
        num_2x3 = 4
        num_2x4 = 3
        num_pencil = 0
        total_offset = [0.1, -0.1, 0.006]
        grasp_order = [1, 0, 2]
        gap_item = 0.015
        gap_block = 0.02
        from_virtual = True
        real_operate = False
        obs_order = 'images'

        env = Arm(is_render=True)
        env.get_parameters(num_2x2=num_2x2, num_2x3=num_2x3, num_2x4=num_2x4,
                           total_offset=total_offset, grasp_order=grasp_order,
                           gap_item=gap_item, gap_block=gap_block, from_virtual=from_virtual)

        image_chaotic = env.reset()
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

        cv2.line(new_img, (int(new_img.shape[1] / 2), 0), (int(new_img.shape[1] / 2), new_img.shape[0]), (0, 0, 0), 20)
        cv2.imshow("Comparison between Chaotic Configuration and Trim Configuration", new_img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    if command == 'knolling':

        num_2x2 = 2
        num_2x3 = 2
        num_2x4 = 2
        total_offset = [0.15, 0, 0.006]
        grasp_order = [2,1,0]
        gap_item = 0.015
        gap_block = 0.02
        random_offset = True
        real_operate = False
        obs_order = 'rl_obj'

        max_step = 5

        env = Arm(is_render=True)
        env.get_parameters(num_2x2=num_2x2, num_2x3=num_2x3, num_2x4=num_2x4,
                           total_offset=total_offset, grasp_order=grasp_order,
                           gap_item=gap_item, gap_block=gap_block,
                           real_operate=real_operate, obs_order=obs_order, random_offset=random_offset,
                           max_step=max_step)
        evaluations = 100

        for i in range(evaluations):
            image_trim = env.change_config()
            a = env.action_space.sample()
            _ = env.reset()
            env.step(i)

    if command == 'train':
        num_2x2 = 2
        num_2x3 = 2
        num_2x4 = 2
        total_offset = [0.15, 0, 0.006]
        grasp_order = [2, 1, 0]
        gap_item = 0.015
        gap_block = 0.02
        random_offset = True
        real_operate = False
        obs_order = 'rl_obj'

        max_step = 5

        env = Arm(is_render=True)
        env.get_parameters(num_2x2=num_2x2, num_2x3=num_2x3, num_2x4=num_2x4,
                           total_offset=total_offset, grasp_order=grasp_order,
                           gap_item=gap_item, gap_block=gap_block,
                           real_operate=real_operate, obs_order=obs_order, random_offset=random_offset,
                           max_step=max_step)
        evaluations = 100

        obs = env.reset()
        for _ in range(evaluations):
            a = env.action_space.sample()
            # print('bbbbbbbbbbbbbbb')
            # a = [0.05, -0.15, 0.005, 0.8985989, 0.4]
            state_, reward, done, _ = env.step(a)
            # time.sleep(100)
            if done:
                print('reset')
                obs = env.reset()