import numpy as np

from items import sort
import pybullet_data as pd
import math
from turdf import *
import socket
import cv2
from cam_obs import *
import matplotlib.pyplot as plt

random.seed(10)
np.random.seed(10)

class Arm:

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

        self.low_scale = np.array([0.05, -0.15, 0.006, - np.pi / 2, 0])
        self.high_scale = np.array([0.25, 0.15, 0.05, np.pi / 2, 0.4])
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
                       from_virtual=True, real_operate=False, obs_order='1'):

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
        self.num_list = np.array([self.num_2x2, self.num_2x3, self.num_2x4, self.num_pencil])

    def get_obs(self, order):

        def get_joints_obs():

            pass

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
            # show in window
            temp = np.copy(image[:, :, 0])
            image[:, :, 0] = np.copy(image[:, :, 2])
            image[:, :, 2] = temp
            print(image.shape)
            cv2.imshow("Comparison between Chaotic Configuration and Trim Configuration", image)
            cv2.waitKey()
            cv2.destroyAllWindows()

            img = image[:, :,:3]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            add = int((640-480)/2)
            img = cv2.copyMakeBorder(img, add, add, 0, 0,cv2.BORDER_CONSTANT, None, value = 0)
            results = np.asarray(detect(img))
            results = np.asarray(results[:,:5]).astype(np.float32)

            index = []
            correct = []
            for i in range(len(self.grasp_order)):
                correct.append(self.xyz_list[self.all_index[i][0]])
            correct = np.asarray(correct)
            for i in range(len(correct)):
                for j in range(len(results)):
                    if np.linalg.norm(correct[i][:2] - results[j][3:5]) < 0.001:
                        index.append(j)
            print(index)
            manipulator_before = []
            for i in index:
                manipulator_before.append([results[i][0], results[i][1], z, roll, pitch, results[i][2]])
            manipulator_before = np.asarray(manipulator_before)
            print(results)
            print(manipulator_before)
            new_xyz_list = self.xyz_list

            return manipulator_before, new_xyz_list

        def get_lego_obs():

            # sequence: pos before, ori before
            manipulator_before = []
            new_pos_before, new_ori_before = [], []
            new_xyz_list = []

            for i in range(len(self.obj_idx)):
                new_pos_before.append(p.getBasePositionAndOrientation(self.obj_idx[i])[0])
                new_ori_before.append(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.obj_idx[i])[1]))
            for i in self.grasp_order:
                for j in range(len(self.all_index[i])):
                    manipulator_before.append(new_pos_before[self.all_index[i][j]] + new_ori_before[self.all_index[i][j]])
                    new_xyz_list.append(self.xyz_list[self.all_index[i][j]])

            manipulator_before = np.concatenate((new_pos_before, new_ori_before), axis=1)
            new_xyz_list = self.xyz_list

            return manipulator_before, new_xyz_list

        def get_real_image_obs():

            pass

        if order == 'sim_obj':
            manipulator_before, new_xyz_list = get_lego_obs()
            return manipulator_before, new_xyz_list
        elif order == 'sim_image_obj':
            manipulator_before, new_xyz_list = get_sim_image_obs()
            return manipulator_before, new_xyz_list
        elif order == 'joint':
            get_joints_obs()
        elif order == 'images':
            image = get_images()
            return image
        elif order == 'real_image_obj':
            get_real_image_obs()

    def reset(self):

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
        items_sort = sort(self.num_2x2, self.num_2x3, self.num_2x4, self.num_pencil)
        self.xyz_list, _, _, self.all_index = items_sort.get_data(self.grasp_order)

        restrict = np.max(self.xyz_list)
        gripper_height = 0.012
        self.obj_idx = []
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
                    p.loadURDF(os.path.join(self.urdf_path, f"item_{self.grasp_order[i]}/{j}.urdf"), basePosition=rdm_pos,
                               baseOrientation=p.getQuaternionFromEuler(ori), useFixedBase=False,
                               flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))

        print(self.obj_idx)
        for i in range(len(self.obj_idx)):
            p.changeDynamics(self.obj_idx[i], -1, restitution=30)

        # set the initial pos of the arm
        ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=[0, 0, 0.066],
                                                  maxNumIterations=200,
                                                  targetOrientation=p.getQuaternionFromEuler([0, math.pi / 2, 0]))
        for motor_index in range(self.num_motor):
            p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                    targetPosition=ik_angles0[motor_index], maxVelocity=7)
        for i in range(60):
            p.stepSimulation()
        return self.get_obs('images')

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
        items_sort = sort(self.num_2x2, self.num_2x3, self.num_2x4, self.num_pencil)
        self.xyz_list, _, _, self.all_index = items_sort.get_data(self.grasp_order)
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

        def reorder_items():

            min_result = []
            best_config = []
            for i in range(len(self.grasp_order)):
                item_index = self.all_index[i]
                item_xyz = self.xyz_list[item_index, :]
                item_num = len(item_index)
                xy, config = calculate_items(item_num, item_xyz)
                # print(f'this is min xy {xy}')
                min_result.append(list(xy))
                # print(f'this is the best item config {config}')
                best_config.append(list(config))
            min_result = np.asarray(min_result).reshape(-1, 2)
            min_xy = np.copy(min_result)
            best_config = np.asarray(best_config).reshape(-1, 2)
            # print(min_xy)
            # print(best_config)

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
                # print(fac)
            else:
                fac = [1, all_num]

            for i in range(iteration):

                sequence = np.random.choice(best_config.shape[0], size=len(self.grasp_order), replace=False)
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

        def calculate_block(best_config, start_pos, index_block, item_index, item_xyz, index_flag):

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

                    ori, pos = calculate_block(best_config, start_pos, index_block, item_index, item_xyz,
                                                    index_flag)
                    # print('tryori', ori)
                    # print('trypos', pos)
                    item_pos[item_index] = pos
                    item_ori[item_index] = ori

            return item_pos, item_ori  # pos_list, ori_list

        # determine the center of the tidy configuration
        self.items_pos_list, self.items_ori_list = reorder_items()
        x_low = np.min(self.items_pos_list, axis=0)[0]
        x_high = np.max(self.items_pos_list, axis=0)[0]
        y_low = np.min(self.items_pos_list, axis=0)[1]
        y_high = np.max(self.items_pos_list, axis=0)[1]
        center = np.array([(x_low + x_high) / 2, (y_low + y_high) / 2, 0])
        self.items_pos_list = self.items_pos_list + self.total_offset - center
        self.manipulator_after = np.concatenate((self.items_pos_list, self.items_ori_list), axis=1)
        print(self.manipulator_after)

        # import urdf and assign the trim pos & ori
        items_names = globals()
        self.obj_idx = []
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
        # while 1:
        #     p.stepSimulation()
        return self.get_obs('images')

    def planning(self, order, conn, height):

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

        def move(cur_pos, cur_ori, tar_pos, tar_ori, move_slice):

            plot_cmd = []
            plot_real = []
            target_pos = np.copy(tar_pos)
            target_ori = np.copy(tar_ori)
            #!!!!!!!!!!!
            # if abs(tar_pos[0] - cur_pos[0]) < 0.001 and abs(tar_pos[1] - cur_pos[1]) < 0.001:
            #     num_step = 15
            # elif abs(tar_pos[2] - cur_pos[2]) < 0.001:
            #     num_step = 10
            # num_step = np.linalg.norm(tar_pos - cur_pos) // move_slice
            step_pos = (tar_pos - cur_pos) / move_slice
            step_ori = (tar_ori - cur_ori) / move_slice
            #!!!!!!!!!!!
            if self.real_operate == True:
                while True:
                    tar_pos = cur_pos + step_pos
                    tar_ori = cur_ori + step_ori

                    ik_angles_real = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=tar_pos + np.array([0, 0, height]),
                                                             maxNumIterations=200,
                                                             targetOrientation=p.getQuaternionFromEuler(tar_ori))
                    ik_angles_sim = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=tar_pos,
                                                              maxNumIterations=200,
                                                              targetOrientation=p.getQuaternionFromEuler(tar_ori))
                    angle_real = np.asarray(real_cmd2tarpos(rad2cmd(ik_angles_real[0:5])), dtype=np.float32)
                    plot_cmd.append(angle_real)
                    conn.sendall(angle_real.tobytes())
                    for motor_index in range(self.num_motor):
                        p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                                targetPosition=ik_angles_sim[motor_index], maxVelocity=0.5)
                    for i in range(40):
                        p.stepSimulation()
                        # time.sleep(1 / 240)

                    real_pos = conn.recv(1024)
                    real_pos = np.frombuffer(real_pos, dtype=np.float32)
                    plot_real.append(real_pos)

                    if abs(target_pos[0] - tar_pos[0]) < 0.001 and abs(target_pos[1] - tar_pos[1]) < 0.001 and abs(target_pos[2] - tar_pos[2]) < 0.001 and \
                            abs(target_ori[0] - tar_ori[0]) < 0.001 and abs(target_ori[1] - tar_ori[1]) < 0.001 and abs(target_ori[2] - tar_ori[2]) < 0.001:
                        break
                    cur_pos = tar_pos
                    cur_ori = tar_ori

                plot_step = np.arange(move_slice)
                plot_cmd = np.asarray(plot_cmd)
                plot_real = np.asarray(plot_real)
                print(np.asarray(plot_cmd[:, 0]).reshape(-1))
                np.savetxt('Cartisian_data/cmd.txt', plot_cmd)
                np.savetxt('Cartisian_data/step.txt', plot_step)
                np.savetxt('Cartisian_data/real.txt', plot_real)

            else:
                while True:
                    tar_pos = cur_pos + step_pos
                    tar_ori = cur_ori + step_ori
                    ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=tar_pos,
                                                              maxNumIterations=200,
                                                              targetOrientation=p.getQuaternionFromEuler(tar_ori))
                    for motor_index in range(self.num_motor):
                        p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                                targetPosition=ik_angles0[motor_index], maxVelocity=0.5)
                    for i in range(40):
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
                    pos_real = np.asarray([gap, gap], dtype=np.float32)
                elif gap <= 0.0265:
                    pos_real = np.asarray([0, 0], dtype=np.float32)
                print('gripper', pos_real)
                conn.sendall(pos_real.tobytes())
                # print(f'this is the cmd pos {pos_real}')
                p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL, targetPosition=gap, force=10)
                p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL, targetPosition=gap, force=10)

                real_pos = conn.recv(1024)
                real_pos = np.frombuffer(real_pos, dtype=np.float32)

            else:
                p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL, targetPosition=gap, force=10)
                p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL, targetPosition=gap, force=10)
            for i in range(30):
                p.stepSimulation()
                # time.sleep(1 / 120)

        def clean_desk():

            barricade_pos = []
            barricade_index = []
            manipulator_before, new_xyz_list = self.get_obs(self.obs_order)
            for i in range(len(manipulator_before)):
                for j in range(len(self.manipulator_after)):
                    restrict_item_i = np.sqrt((new_xyz_list[i][0]) ** 2 + (new_xyz_list[i][1]) ** 2)
                    restrict_item_j = np.sqrt((new_xyz_list[j][0]) ** 2 + (new_xyz_list[j][1]) ** 2)
                    if np.linalg.norm(self.manipulator_after[j][:3] - manipulator_before[i][:3]) < restrict_item_i / 2 + restrict_item_j / 2:
                        if i not in barricade_index:
                            print('We need to clean the desktop to provide an enough space')
                            barricade_pos.append(manipulator_before[i][:3])
                            barricade_index.append(i)
            barricade_pos = np.asarray(barricade_pos)

            x_high = np.max(self.manipulator_after[:, 0])
            x_low = np.min(self.manipulator_after[:, 0])
            y_high = np.max(self.manipulator_after[:, 1])
            y_low = np.min(self.manipulator_after[:, 1])
            p.addUserDebugLine(lineFromXYZ=[x_low, y_low, 0.006], lineToXYZ=[x_high, y_low, 0.006])
            p.addUserDebugLine(lineFromXYZ=[x_low, y_low, 0.006], lineToXYZ=[x_low, y_high, 0.006])
            p.addUserDebugLine(lineFromXYZ=[x_high, y_high, 0.006], lineToXYZ=[x_high, y_low, 0.006])
            p.addUserDebugLine(lineFromXYZ=[x_high, y_high, 0.006], lineToXYZ=[x_low, y_high, 0.006])

            while len(barricade_index) > 0:

                # pos
                offset_low = np.array([0, 0, -0.002])
                offset_high = np.array([0, 0, 0.04])
                # ori
                rest_ori = np.array([0, 1.57, 0])
                # axis and direction
                if y_high - y_low > x_high - x_low:
                    offset_rectangle = np.array([0, 0, math.pi / 2])
                    axis = 'x_axis'
                    if (x_high + x_low) / 2 > (self.x_high_obs + self.x_low_obs) / 2:
                        direction = 'negative'
                        offset_horizontal = np.array([np.max(self.xyz_list) - 0.005, 0, 0])
                    else:
                        direction = 'positive'
                        offset_horizontal = np.array([-(np.max(self.xyz_list) - 0.005), 0, 0])
                else:
                    offset_rectangle = np.array([0, 0, 0])
                    axis = 'y_axis'
                    if (y_high + y_low) / 2 > (self.y_high_obs + self.y_low_obs) / 2:
                        direction = 'negative'
                        offset_horizontal = np.array([0, np.max(self.xyz_list) - 0.005, 0])
                    else:
                        direction = 'positive'
                        offset_horizontal = np.array([0, -(np.max(self.xyz_list) - 0.005), 0])

                trajectory_pos_list = []
                trajectory_ori_list = []
                print(barricade_index)
                for i in range(len(barricade_index)):
                    if axis == 'x_axis':
                        if direction == 'positive':
                            print('x,p')
                            terminate = np.array([x_high, barricade_pos[i][1], barricade_pos[i][2]])
                        elif direction == 'negative':
                            print('x,n')
                            terminate = np.array([x_low, barricade_pos[i][1], barricade_pos[i][2]])
                    elif axis == 'y_axis':
                        if direction == 'positive':
                            print('y,p')
                            terminate = np.array([barricade_pos[i][0], y_high, barricade_pos[i][2]])
                        elif direction == 'negative':
                            print('y,n')
                            terminate = np.array([barricade_pos[i][0], y_low, barricade_pos[i][2]])

                    trajectory_pos_list.append([0.02959])
                    trajectory_pos_list.append(barricade_pos[i] + offset_high + offset_horizontal)
                    trajectory_pos_list.append(barricade_pos[i] + offset_low + offset_horizontal)
                    trajectory_pos_list.append(offset_low - offset_horizontal + terminate)
                    trajectory_pos_list.append(offset_high - offset_horizontal + terminate)

                    trajectory_ori_list.append(rest_ori + offset_rectangle)
                    trajectory_ori_list.append(rest_ori + offset_rectangle)
                    trajectory_ori_list.append(rest_ori + offset_rectangle)
                    trajectory_ori_list.append(rest_ori + offset_rectangle)
                    trajectory_ori_list.append(rest_ori + offset_rectangle)

                # reset the manipulator to read the image
                trajectory_pos_list.append([0, 0, 0.066])
                trajectory_ori_list.append([0, math.pi / 2, 0])

                last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
                last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))
                move_slice = 30
                for j in range(len(trajectory_pos_list)):

                    if len(trajectory_pos_list[j]) == 3:
                        move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j], move_slice)
                        last_pos = np.copy(trajectory_pos_list[j])
                        last_ori = np.copy(trajectory_ori_list[j])
                    elif len(trajectory_pos_list[j]) == 1:
                        gripper(trajectory_pos_list[j][0])

                break_flag = False
                barricade_pos = []
                barricade_index = []
                manipulator_before, new_xyz_list = self.get_obs(self.obs_order)
                for i in range(len(manipulator_before)):
                    for j in range(len(self.manipulator_after)):
                        restrict_item_i = np.sqrt((new_xyz_list[i][0]) ** 2 + (new_xyz_list[i][1]) ** 2)
                        restrict_item_j = np.sqrt((new_xyz_list[j][0]) ** 2 + (new_xyz_list[j][1]) ** 2)
                        if np.linalg.norm(self.manipulator_after[j][:3] - manipulator_before[i][:3]) < restrict_item_i / 2 + restrict_item_j / 2:
                            if i not in barricade_index:
                                print('We still need to clean the desktop to provide an enough space')
                                barricade_pos.append(manipulator_before[i][:3])
                                barricade_index.append(i)
                                break_flag = True
                                break
                    if break_flag == True:
                        break
                barricade_pos = np.asarray(barricade_pos)
            else:
                print('nothing to clean')
                pass
            print('clean end, this is the new pos')

        def clean_item():

            gripper_width = 0.0165
            gripper_height = 0.012
            restrict_gripper_diagonal = np.sqrt(gripper_width ** 2 + gripper_height ** 2)
            crowded_pos = []
            crowded_ori = []
            crowded_index = []
            manipulator_before, new_xyz_list = self.get_obs(self.obs_order)
            print(manipulator_before)

            # define the cube which is crowded
            for i in range(len(manipulator_before)):
                for j in range(len(manipulator_before)):
                    restrict_item_i = np.sqrt((new_xyz_list[i][0]) ** 2 + (new_xyz_list[i][1]) ** 2)
                    restrict_item_j = np.sqrt((new_xyz_list[j][0]) ** 2 + (new_xyz_list[j][1]) ** 2)
                    if 0.0001 < np.linalg.norm(manipulator_before[j][:3] - manipulator_before[i][:3]) < restrict_item_i / 2 + restrict_item_j / 2:
                        if i not in crowded_index and j not in crowded_index:
                            print('We need to clean the items surrounding it to provide an enough space')
                            crowded_pos.append(manipulator_before[i][:3])
                            crowded_ori.append(manipulator_before[i][3:6])
                            crowded_pos.append(manipulator_before[j][:3])
                            crowded_ori.append(manipulator_before[j][3:6])
                            crowded_index.append(i)
                            crowded_index.append(j)
                        if i in crowded_index and j not in crowded_index:
                            print('We need to clean the items surrounding it to provide an enough space')
                            crowded_pos.append(manipulator_before[j][:3])
                            crowded_ori.append(manipulator_before[j][3:6])
                            crowded_index.append(j)
            crowded_pos = np.asarray(crowded_pos)

            while len(crowded_index) > 0:
                # pos
                offset_low = np.array([0, 0, -0.004])
                offset_high = np.array([0, 0, 0.04])
                # ori
                rest_ori = np.array([0, 1.57, 0])

                trajectory_pos_list = []
                trajectory_ori_list = []
                for i in range(len(crowded_index)):
                    break_flag = False
                    once_flag = False
                    x = new_xyz_list[crowded_index[i]][0] / 2
                    y = new_xyz_list[crowded_index[i]][1] / 2
                    theta = manipulator_before[crowded_index[i]][5]
                    vertex_1 = np.array(
                        [(x * math.cos(theta)) - (y * math.sin(theta)), (x * math.sin(theta)) + (y * math.cos(theta)), 0])
                    vertex_2 = np.array(
                        [(-x * math.cos(theta)) - (y * math.sin(theta)), (-x * math.sin(theta)) + (y * math.cos(theta)), 0])
                    vertex_3 = np.array([(-x * math.cos(theta)) - (-y * math.sin(theta)),
                                         (-x * math.sin(theta)) + (-y * math.cos(theta)), 0])
                    vertex_4 = np.array(
                        [(x * math.cos(theta)) - (-y * math.sin(theta)), (x * math.sin(theta)) + (-y * math.cos(theta)), 0])
                    point_1 = vertex_1 + np.array(
                        [(gripper_width * math.cos(theta)) - (gripper_height * math.sin(theta)),
                         (gripper_width * math.sin(theta)) + (gripper_height * math.cos(theta)), 0])
                    point_2 = vertex_2 + np.array(
                        [(-gripper_width * math.cos(theta)) - (gripper_height * math.sin(theta)),
                         (-gripper_width * math.sin(theta)) + (gripper_height * math.cos(theta)), 0])
                    point_3 = vertex_3 + np.array(
                        [(-gripper_width * math.cos(theta)) - (-gripper_height * math.sin(theta)),
                         (-gripper_width * math.sin(theta)) + (-gripper_height * math.cos(theta)), 0])
                    point_4 = vertex_4 + np.array(
                        [(gripper_width * math.cos(theta)) - (-gripper_height * math.sin(theta)),
                         (gripper_width * math.sin(theta)) + (-gripper_height * math.cos(theta)), 0])
                    sequence_point = np.array([point_1, point_2, point_3, point_4])

                    print(crowded_index)
                    print(crowded_index[i])
                    t = 0
                    for j in range(len(sequence_point)):
                        vertex_break_flag = False
                        for k in range(len(manipulator_before)):
                            restrict_item_k = np.sqrt((new_xyz_list[k][0]) ** 2 + (new_xyz_list[k][1]) ** 2)
                            if 0.001 < np.linalg.norm(sequence_point[0] + crowded_pos[i] - manipulator_before[k][:3]) < restrict_item_k/2 + restrict_gripper_diagonal/2:
                                p.addUserDebugPoints([sequence_point[0] + crowded_pos[i]], [[0.1, 0, 0]], pointSize=5)
                                print("this vertex doesn't work")
                                vertex_break_flag = True
                                break
                        if vertex_break_flag == False:
                            print("this vertex is ok")
                            once_flag = True
                            break
                        else:
                            # should change the vertex and try again
                            sequence_point = np.roll(sequence_point, -1, axis=0)
                            print(sequence_point)
                            t += 1
                        if t == len(sequence_point):
                            # all vertex of this cube fail, should change the cube
                            break_flag = True
                    # problem, change another crowded cube
                    if break_flag == True:
                        continue

                    trajectory_pos_list.append([0.02959])
                    trajectory_pos_list.append(crowded_pos[i] + offset_high + sequence_point[0])
                    trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[0])
                    trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[1])
                    trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[2])
                    trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[3])
                    trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[0])
                    trajectory_pos_list.append(crowded_pos[i] + offset_high + sequence_point[0])
                    # reset the manipulator to read the image
                    trajectory_pos_list.append([0, 0, 0.066])

                    trajectory_ori_list.append(rest_ori)
                    trajectory_ori_list.append(rest_ori + crowded_ori[i])
                    trajectory_ori_list.append(rest_ori + crowded_ori[i])
                    trajectory_ori_list.append(rest_ori + crowded_ori[i])
                    trajectory_ori_list.append(rest_ori + crowded_ori[i])
                    trajectory_ori_list.append(rest_ori + crowded_ori[i])
                    trajectory_ori_list.append(rest_ori + crowded_ori[i])
                    trajectory_ori_list.append(rest_ori + crowded_ori[i])
                    # reset the manipulator to read the image
                    trajectory_ori_list.append([0, math.pi / 2, 0])

                    # only once!
                    if once_flag == True:
                        break

                last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
                last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))
                move_slice = 10
                for j in range(len(trajectory_pos_list)):
                    if len(trajectory_pos_list[j]) == 3:
                        move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j], move_slice)
                        last_pos = np.copy(trajectory_pos_list[j])
                        last_ori = np.copy(trajectory_ori_list[j])
                    elif len(trajectory_pos_list[j]) == 1:
                        gripper(trajectory_pos_list[j][0])

                # check the environment again and update the pos and ori of cubes
                crowded_pos = []
                crowded_ori = []
                crowded_index = []
                manipulator_before, new_xyz_list = self.get_obs(self.obs_order)
                print('this is the clean item')
                print(manipulator_before)
                for i in range(len(manipulator_before)):
                    for j in range(len(manipulator_before)):
                        restrict_item_i = np.sqrt((new_xyz_list[i][0]) ** 2 + (new_xyz_list[i][1]) ** 2)
                        restrict_item_j = np.sqrt((new_xyz_list[j][0]) ** 2 + (new_xyz_list[j][1]) ** 2)
                        if 0.0001 < np.linalg.norm(manipulator_before[j][:3] - manipulator_before[i][:3]) < restrict_item_i / 2 + restrict_item_j / 2:
                            if i not in crowded_index and j not in crowded_index:
                                print('We need to clean the items surrounding it to provide an enough space')
                                crowded_pos.append(manipulator_before[i][:3])
                                crowded_ori.append(manipulator_before[i][3:6])
                                crowded_pos.append(manipulator_before[j][:3])
                                crowded_ori.append(manipulator_before[j][3:6])
                                crowded_index.append(i)
                                crowded_index.append(j)
                            if i in crowded_index and j not in crowded_index:
                                print('We need to clean the items surrounding it to provide an enough space')
                                crowded_pos.append(manipulator_before[j][:3])
                                crowded_ori.append(manipulator_before[j][3:6])
                                crowded_index.append(j)
                crowded_pos = np.asarray(crowded_pos)
            else:
                print('nothing around the item')
                pass
            print('clean end, this is the new pos')

        if order == 1:
            clean_desk()

        elif order == 2:
            clean_item()

        elif order == 3:
            start_end = get_start_end()

            rest_pos = np.array([0, 0, 0.05])
            rest_ori = np.array([0, 1.57, 0])
            offset_low = np.array([0, 0, -0.002])
            offset_high = np.array([0, 0, 0.04])

            for i in range(len(self.obj_idx)):

                trajectory_pos_list = [[0.025],
                                       offset_high + start_end[i][:3],
                                       offset_low + start_end[i][:3],
                                       [0.0273],
                                       offset_high + start_end[i][:3],
                                       offset_high + start_end[i][6:9],
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
                                       rest_ori + start_end[i][9:12],
                                       [0.025],
                                       rest_ori + start_end[i][9:12]]
                last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
                last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))

                move_slice = 10
                for j in range(len(trajectory_pos_list)):

                    if len(trajectory_pos_list[j]) == 3:
                        move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j], move_slice)
                        last_pos = np.copy(trajectory_pos_list[j])
                        last_ori = np.copy(trajectory_ori_list[j])

                    elif len(trajectory_pos_list[j]) == 1:
                        gripper(trajectory_pos_list[j][0])

                for _ in range(30):
                    p.stepSimulation()
                    # self.images = self.get_image()
                    # time.sleep(1 / 120)
            while 1:
                p.stepSimulation()

    def step(self):

        if self.real_operate == True:

            HOST = "192.168.0.186"  # Standard loopback interface address (localhost)
            PORT = 8880  # Port to listen on (non-privileged ports are > 1023)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((HOST, PORT))
            # It should be an integer from 1 to 65535, as 0 is reserved. Some systems may require superuser privileges if the port number is less than 1024.
            # associate the socket with a specific network interface
            s.listen()
            print(f"Waiting for connection...\n")
            conn, addr = s.accept()
            print(conn)
            print(f"Connected by {addr}")
            reset(self.arm_id)
            table_surface_height = 0.02
            num_motor = 5
            # ! reset the pos in both real and sim
            reset_pos = [0.05, 0, table_surface_height + 0.04]
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
                time.sleep(1 / 48)
        else:
            conn = None
            table_surface_height = 0.02

        self.planning(1, conn, table_surface_height)
        self.planning(2, conn, table_surface_height)
        self.planning(3, conn, table_surface_height)

if __name__ == '__main__':

    random_flag = True
    command = 'virtual'  # image virtual real

    if command == 'image':
        num_2x2 = 4
        num_2x3 = 5
        num_2x4 = 3
        num_pencil = 0
        total_offset = [0.1, 0, 0.006]
        grasp_order = [1, 0, 2]
        gap_item = 0.03
        gap_block = 0.02
        from_virtual = True
        real_operate = False
        obs_order = 'sim_image_obj'

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

    if command == 'virtual':

        num_2x2 = 5
        num_2x3 = 5
        num_2x4 = 5
        num_pencil = 0
        total_offset = [0.15, 0, 0.006]
        grasp_order = [1,0,2]
        gap_item = 0.03
        gap_block = 0.02
        from_virtual = True
        real_operate = True
        obs_order = 'sim_obj'

        env = Arm(is_render=False)
        env.get_parameters(num_2x2=num_2x2, num_2x3=num_2x3, num_2x4=num_2x4,
                           total_offset=total_offset, grasp_order=grasp_order,
                           gap_item=gap_item, gap_block=gap_block, from_virtual=from_virtual,
                           real_operate=real_operate, obs_order=obs_order)

        _ = env.change_config()
        _ = env.reset()

        env.step()