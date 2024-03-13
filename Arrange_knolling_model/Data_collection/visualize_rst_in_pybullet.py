import numpy as np
import json
# from Arrange_knolling_model.Data_collection.sort_data_collection_multi_solution import Sort_objects
import pybullet_data as pd
import random
# from turdf import *
import pybullet as p
import os
import cv2
# from cam_obs_yolov8 import *
import torch
import sys
sys.path.append('../train_and_test')
sys.path.append('../../')
from model_structure import *
import pandas
import matplotlib.pyplot as plt
# from urdfpy import URDF

from arrange_policy import configuration_zzz

torch.manual_seed(42)
np.random.seed(100)
random.seed(100)

color_list = [[20, 20, 20,1],
              [20, 20, 20,1],
              [20, 20, 20,1],
              [20, 20, 20,1],
              [100,100,100,1],
              [100,100,100,1],
              [100,100,100,1],
              [200, 200,200,1],
              [200, 200,200,1],
              [200, 200,200,1]]
color_list = np.array(color_list,dtype=float)
color_list[:,:3] = color_list[:,:3]/255
def create_box(body_name: str,
               position: np.ndarray,
               orientation: np.ndarray = np.array([0, 0, 0, 1]),
               rgba_color=None,
               size=None,
               mass=0.1,
               ) -> None:
    """
    Create a box.
    """
    length = size[0]
    width = size[1]
    height = size[2]

    visual_kwargs = {
        "rgbaColor": rgba_color if rgba_color is not None else [np.random.random(), np.random.random(),
                                                                np.random.random(), 1],
        "halfExtents": [length / 2, width / 2, height / 2]
    }
    collision_kwargs = {
        "halfExtents": [length / 2, width / 2, height / 2]
    }

    _create_geometry(body_name,
                          geom_type=p.GEOM_BOX,
                          mass=mass,
                          position=position,
                          orientation=orientation,
                          lateral_friction=1.0,
                          contact_damping=1.0,
                          contact_stiffness=50000,
                          visual_kwargs=visual_kwargs,
                          collision_kwargs=collision_kwargs)


def _create_geometry(
        body_name: str,
        geom_type: int,
        mass: float = 0.0,
        position=None,
        orientation=None,
        ghost: bool = False,
        lateral_friction=None,
        spinning_friction=None,
        contact_damping=None,
        contact_stiffness=None,
        visual_kwargs={},
        collision_kwargs={},
) -> None:
    """Create a geometry.

    Args:
        body_name (str): The name of the body. Must be unique in the sim.
        geom_type (int): The geometry type. See p.GEOM_<shape>.
        mass (float, optional): The mass in kg. Defaults to 0.
        position (np.ndarray, optional): The position, as (x, y, z). Defaults to [0, 0, 0].
        orientation (np.ndarray, optional): The orientation. Defaults to [0, 0, 0, 1]
        ghost (bool, optional): Whether the body can collide. Defaults to False.
        lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
            value. Defaults to None.
        spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
            value. Defaults to None.
        visual_kwargs (dict, optional): Visual kwargs. Defaults to {}.
        collision_kwargs (dict, optional): Collision kwargs. Defaults to {}.
    """
    position = position if position is not None else np.zeros(3)
    orientation = orientation if orientation is not None else np.array([0, 0, 0, 1])

    baseVisualShapeIndex = p.createVisualShape(geom_type, **visual_kwargs)
    if not ghost:
        baseCollisionShapeIndex = p.createCollisionShape(geom_type, **collision_kwargs)
    else:
        baseCollisionShapeIndex = -1
    box_id = p.createMultiBody(
        baseVisualShapeIndex=baseVisualShapeIndex,
        baseCollisionShapeIndex=baseCollisionShapeIndex,
        baseMass=mass,
        basePosition=position,
        baseOrientation=orientation
    )


class Arm:
    def __init__(self, is_render=True):

        self.kImageSize = {'width': 480, 'height': 480}
        self.urdf_path = '../../ASSET/urdf/'
        self.dataset_path = '../../../knolling_dataset/sundry_301/'

        self.pybullet_path = pd.getDataPath()
        self.is_render = is_render
        if self.is_render:
            # p.connect(p.GUI, options="--width=1280 --height=720")
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.num_motor = 5

        self.low_scale = np.array([0.03, -0.14, 0.0, - np.pi / 2, 0])
        self.high_scale = np.array([0.27, 0.14, 0.05, np.pi / 2, 0.4])
        self.low_act = -np.ones(5)
        self.high_act = np.ones(5)
        self.x_low_obs = self.low_scale[0]
        self.x_high_obs = self.high_scale[0]
        self.y_low_obs = self.low_scale[1]
        self.y_high_obs = self.high_scale[1]
        self.z_low_obs = self.low_scale[2]
        self.z_high_obs = self.high_scale[2]
        self.table_boundary = 0.03

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

        if random.uniform(0, 1) > 0.5:
            p.configureDebugVisualizer(lightPosition=[random.randint(1, 3), random.randint(1, 2), 5])
        else:
            p.configureDebugVisualizer(lightPosition=[random.randint(1, 3), random.randint(-2, -1), 5])
        p.configureDebugVisualizer(lightPosition=[random.randint(1, 3), random.randint(1, 2), 5],
                                   shadowMapResolution=8192, shadowMapIntensity=np.random.randint(5, 8) / 10)
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                     cameraYaw=45,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[0.1, 0, 0])
        p.setAdditionalSearchPath(pd.getDataPath())

    def get_parameters(self, box_num=None, area_num=None, ratio_num=None, boxes_index=None,
                       total_offset=None, configuration=None,
                       gap_item=0.03, gap_block=0.02,
                       real_operate=False, obs_order='1',
                       random_offset=False, check_detection_loss=None, obs_img_from=None, use_yolo_pos=True,
                       item_odd_prevent=None, block_odd_prevent=None, upper_left_max = None, forced_rotate_box=None):

        # self.lego_num = lego_num
        self.total_offset = total_offset
        self.area_num = area_num
        self.ratio_num = ratio_num
        self.gap_item = gap_item
        self.gap_block = gap_block
        self.real_operate = real_operate
        self.obs_order = obs_order
        self.random_offset = random_offset
        self.num_list = box_num
        self.check_detection_loss = check_detection_loss
        self.obs_img_from = obs_img_from
        self.use_yolo_pos = use_yolo_pos
        self.boxes_index = boxes_index
        self.configuration = configuration
        self.item_odd_prevent = item_odd_prevent
        self.block_odd_prevent = block_odd_prevent
        self.upper_left_max = upper_left_max
        self.forced_rotate_box = forced_rotate_box

    def get_obs(self, order, evaluation):

        def get_images():
            (width, length, image, _, _) = p.getCameraImage(width=640,
                                                            height=480,
                                                            viewMatrix=self.view_matrix,
                                                            projectionMatrix=self.projection_matrix,
                                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)
            return image

        if order == 'images':
            image = get_images()
            return image

    def is_too_close(self, new_pos, existing_objects, min_distance=0.01):
        for _, obj_pos in existing_objects:
            if np.linalg.norm(np.array(new_pos) - np.array(obj_pos)) < min_distance:
                return True
        return False


    def label2image(self, labels_data,labels_name=None):

        labels_data = labels_data.reshape(-1, 7)

        pos_data = np.concatenate((labels_data[:, :2], np.ones((len(labels_data), 1)) * 0.003), axis=1)

        lw_data = labels_data[:, 2:5]
        # ori_data = labels_data[:, 3:6]
        ori_data = np.zeros((len(lw_data), 3))
        color_index = labels_data[:, -1]
        class_index = labels_data[:, -2]

        # Converting dictionary keys to integers
        dict_map = {i: v for i, (k, v) in enumerate(color_dict.items())}

        # Mapping array values to dictionary values
        rdm_color_index = np.random.choice(10, len(color_index))
        mapped_color_values = []
        for i in range(len(color_index)):
            mapped_color_values.append(dict_map[color_index[i]][rdm_color_index[i]])

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

        baseid = p.loadURDF(self.urdf_path + "plane_zzz.urdf", useFixedBase=1,
                            flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        textureId = p.loadTexture(self.urdf_path + "floor_white.png")
        # p.changeVisualShape(baseid, -1, textureUniqueId=textureId,
        #                     rgbaColor=[np.random.uniform(0.9, 1), np.random.uniform(0.9, 1), np.random.uniform(0.9, 1),
        #                                1])
        p.changeDynamics(baseid, -1, lateralFriction=1, spinningFriction=1, rollingFriction=0.002, linearDamping=0.5,
                         angularDamping=0.5)
        ################### recover urdf boxes based on lw_data ###################
        obj_name = []
        for i in range(len(lw_data)):
            # print(f'this is matching urdf{j}')
            # print(pos_data[i])
            # print(lw_data[i])
            # print(ori_data[i])
            pos_data[i, 2] += 0.006
            pos_data[i, 1] += y_axis_shift
            pos_data[i,0] += 0.016
            if lw_data[i, 0] < lw_data[i, 1]:
                ori_data[i, 2] += np.pi / 2

            obj_name.append(f'object_{i}')
            # create_box(f'object_{i}', pos_data[i], p.getQuaternionFromEuler(ori_data[i]), size=lw_data[i])
            p.loadURDF(self.dataset_path + 'urdf_file/' + labels_name[i] + '.urdf',
                                         basePosition=pos_data[i],
                                         baseOrientation=p.getQuaternionFromEuler(ori_data[i]), useFixedBase=False,
                                         flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

            p.changeVisualShape(p.getBodyUniqueId(i+1), -1, rgbaColor=mapped_color_values[i] + [1])


        ################### recover objects  ###################

        # object_idx = []
        # print('position\n', pos_data)
        # print('lwh\n', lw_data)
        # for i in range(len(labels_name)):
        #
        #     # object_path = self.dataset_path + 'generated_stl/' + labels_name[i][:-2] + '/'
        #     object_path = self.dataset_path + 'generated_stl/' + labels_name[i][:-2] + '/' + labels_name[i]
        #     object_csv = object_path + '.csv'
        #     # object_stl = object_path + '.stl'
        #
        #     print(f'this is matching urdf{i}')
        #     csv_lwh = np.asarray(pandas.read_csv(object_csv).iloc[0, [3, 4, 5]].values) * 0.001
        #     pos_data[i, 2] = csv_lwh[2] / 2
        #     pos_data[i, 1] += y_axis_shift
        #
        #     if lw_data[i, 0] < lw_data[i, 1]:
        #         ori_data[i, 2] += np.pi / 2
        #     object_idx.append(p.loadURDF(self.dataset_path + 'urdf_file/' + labels_name[i] + '.urdf',
        #                                  basePosition=pos_data[i],
        #                                  baseOrientation=p.getQuaternionFromEuler(ori_data[i]), useFixedBase=False,
        #                                  flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
        #     p.changeVisualShape(p.getBodyUniqueId(i + 1), -1, rgbaColor=mapped_color_values[i] + [1])

        # neat_img = self.get_obs('images', None)


        ################### recover urdf boxes based on lw_data ###################

        # shutil.rmtree(save_urdf_path_one_img)
        for i in range(100):
            p.stepSimulation()

        return self.get_obs('images', None)

    def reset(self):

        # get the standard xyz and corresponding index from files in the computer
        items_sort = Sort_objects(configuration=self.configuration)
        self.obj_idx = []
        if self.real_operate == False:
            self.xyz_list, _, _, self.all_index, self.transform_flag = items_sort.get_data_virtual(self.area_num,
                                                                                                   self.ratio_num,
                                                                                                   self.num_list,
                                                                                                   self.boxes_index)
            restrict = np.max(self.xyz_list)
            gripper_height = 0.012
            last_pos = np.array([[0, 0, 1]])

            ############## collect ori and pos to calculate the error of detection ##############
            collect_ori = []
            collect_pos = []
            ############## collect ori and pos to calculate the error of detection ##############

            for i in range(len(self.all_index)):
                for j in range(len(self.all_index[i])):
                    #         pass
                    # for i in range(len(self.grasp_order)):
                    #     for j in range(self.num_list[self.grasp_order[i]]):

                    rdm_pos = np.array([random.uniform(self.x_low_obs, self.x_high_obs),
                                        random.uniform(self.y_low_obs, self.y_high_obs), 0.0])
                    ori = [0, 0, random.uniform(0, np.pi)]
                    # ori = [0, 0, 0]
                    collect_ori.append(ori)
                    check_list = np.zeros(last_pos.shape[0])

                    while 0 in check_list:
                        rdm_pos = [random.uniform(self.x_low_obs, self.x_high_obs),
                                   random.uniform(self.y_low_obs, self.y_high_obs), 0.0]
                        for z in range(last_pos.shape[0]):
                            if np.linalg.norm(last_pos[z] - rdm_pos) < restrict + gripper_height:
                                check_list[z] = 0
                            else:
                                check_list[z] = 1
                    collect_pos.append(rdm_pos)

                    last_pos = np.append(last_pos, [rdm_pos], axis=0)

            collect_ori = np.asarray(collect_ori)
            collect_pos = np.asarray(collect_pos)
            self.check_ori = collect_ori[:, 2]
            self.check_pos = collect_pos[:, :2]
        for i in range(60):
            p.stepSimulation()
        # return self.get_obs('images', None)

        ################# change the sequence of data based on the max area of single box #####################
        box_order = np.argsort(self.xyz_list[:, 0] * self.xyz_list[:, 1])[::-1]
        self.check_pos = self.check_pos[box_order]
        self.check_ori = self.check_ori[box_order]
        self.xyz_list = self.xyz_list[box_order]
        ################# change the sequence of data based on the max area of single box #####################


        return self.check_pos, self.check_ori, self.xyz_list[:, :2], self.transform_flag

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

        baseid = p.loadURDF(self.urdf_path + "plane_zzz.urdf", basePosition=[0, 0, 0], useFixedBase=1,
                            flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        # self.arm_id = p.loadURDF(self.urdf_path + "robot_arm928/robot_arm_fixed.urdf",
        #                          basePosition=[-0.08, 0, 0.02], useFixedBase=True,
        #                          flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        textureId = p.loadTexture(self.urdf_path + "floor_1.png")
        p.changeDynamics(baseid, -1, lateralFriction=self.lateral_friction, frictionAnchor=True)
        # p.changeDynamics(self.arm_id, 7, lateralFriction=self.lateral_friction, frictionAnchor=True)
        # p.changeDynamics(self.arm_id, 8, lateralFriction=self.lateral_friction, frictionAnchor=True)
        # p.changeVisualShape(baseid, -1, textureUniqueId=textureId,
        #                     rgbaColor=[np.random.uniform(0.9, 1), np.random.uniform(0.9, 1), np.random.uniform(0.9, 1),
        #                                1])

        # get the standard xyz and corresponding index from files in the computer
        items_sort = Sort_objects(configuration=self.configuration)
        if self.real_operate == False:
            self.xyz_list, _, _, self.all_index, self.transform_flag = items_sort.get_data_virtual(self.area_num,
                                                                                                   self.ratio_num,
                                                                                                   self.num_list,
                                                                                                   self.boxes_index)
        calculate_reorder = configuration_zzz(self.xyz_list, self.all_index, self.gap_item, self.gap_block, self.transform_flag, self.configuration,
                                              self.item_odd_prevent, self.block_odd_prevent, self.upper_left_max, self.forced_rotate_box)

        # determine the center of the tidy configuration
        self.items_pos_list, self.items_ori_list = calculate_reorder.calculate_block()
        # the sequence of self.items_pos_list, self.items_ori_list are the same as those in self.xyz_list
        x_low = np.min(self.items_pos_list, axis=0)[0]
        x_high = np.max(self.items_pos_list, axis=0)[0]
        y_low = np.min(self.items_pos_list, axis=0)[1]
        y_high = np.max(self.items_pos_list, axis=0)[1]
        center = np.array([(x_low + x_high) / 2, (y_low + y_high) / 2, 0])
        x_length = abs(x_high - x_low)
        y_length = abs(y_high - y_low)
        # print(x_low, x_high, y_low, y_high)
        if self.random_offset == True:
            self.total_offset = np.array([random.uniform(self.x_low_obs + x_length / 2, self.x_high_obs - x_length / 2),
                                          random.uniform(self.y_low_obs + y_length / 2, self.y_high_obs - y_length / 2),
                                          0.0])
        else:
            pass
        self.items_pos_list = self.items_pos_list + self.total_offset
        self.manipulator_after = np.concatenate((self.items_pos_list, self.items_ori_list), axis=1)

        # return self.get_obs('images', None)

        ################# change the sequence of data based on the max area of single box #####################
        box_order = np.argsort(self.xyz_list[:, 0] * self.xyz_list[:, 1])[::-1]
        self.items_pos_list = self.items_pos_list[box_order]
        self.items_ori_list = self.items_ori_list[box_order]
        self.xyz_list = self.xyz_list[box_order]
        ################# change the sequence of data based on the max area of single box #####################

        return self.items_pos_list[:, :2], self.items_ori_list[:, 2], self.xyz_list[:, :2], self.transform_flag


if __name__ == '__main__':

    info_per_object = 7
    start_evaluations = 0
    end_evaluations =   1000000
    step_num = 10
    save_point = np.linspace(int((end_evaluations - start_evaluations) / step_num + start_evaluations), end_evaluations, step_num)

    y_axis_shift = -0.17 + 0.016
    num_samples = 100

    SHIFT_DATA_SET = 0

    method_list = ['flowing-moon-116', 'eternal-sweep-1', 'restful-sweep-1',
                   'dazzling-sweep-1', 'sage-sweep-1', 'magic-tree-145']

    name = method_list[0] + '_min'
    n_all_solutions = 1  # num_gaussian**object_num

    for object_num in range(10,11):

        data_path = f'../train_and_test/results/{name}/{SHIFT_DATA_SET}/'

        images_log_path = data_path+'output_images%d/'%object_num
        os.makedirs(images_log_path, exist_ok=True)

        gt_data = np.loadtxt(data_path + '/num_%d_gt.txt' % object_num)
        obj_name_data = np.loadtxt(data_path + '/obj_name_%d.txt' % object_num, dtype=str)

        pred_data          = []
        ms_min_sample_loss = []
        ll_loss            = []
        overlap_loss       = []
        pos_loss           = []
        entropy_loss       = []
        for i in range(n_all_solutions):
            pred_data         .append(np.loadtxt(data_path + f'/num_{object_num}_pred_{i}.txt')    )
            ms_min_sample_loss.append(np.loadtxt(data_path+f'ms_min_sample_loss{object_num}.csv'))
            ll_loss           .append(np.loadtxt(data_path+f'll_loss{object_num}.csv')           )
            overlap_loss      .append(np.loadtxt(data_path + f'overlap_loss{object_num}.csv')    )
            pos_loss          .append(np.loadtxt(data_path + f'pos_loss{object_num}.csv')        )
            entropy_loss      .append(np.loadtxt(data_path + f'v_entropy_loss{object_num}.csv')  )
        pred_data = np.asarray(pred_data)
        all_loss = [ms_min_sample_loss,
                    ll_loss,
                    overlap_loss,
                    pos_loss,
                    entropy_loss]
        all_loss_name = ['ms_min_sample_loss',
                    'll_loss',
                    'overlap_loss',
                    'pos_loss',
                    'entropy_loss']



        env = Arm(is_render=False)
        with open('../../ASSET/urdf/object_color/rgb_info.json') as f:
            color_dict = json.load(f)
        env.get_parameters(box_num=object_num)

        new_data = []
        for i in range(num_samples):
            print(f'this is data {i}')
            all_ready = False
            image_list = []
            images_log_f_path = images_log_path+'%d/'%i
            os.makedirs(images_log_f_path,exist_ok=True)
            for solu_i in range(n_all_solutions+1):
                print("solution:",solu_i)
                if solu_i==0:
                    #this is gt data
                    data = gt_data[:, :object_num * info_per_object]

                else:
                    data = pred_data[solu_i-1][:, :object_num * info_per_object]
                one_img_data = data[i][:info_per_object*object_num].reshape(-1, info_per_object)
                box_order = np.lexsort((one_img_data[:, 1], one_img_data[:, 0]))
                one_img_data = one_img_data[box_order].reshape(-1,)
                if len(box_order) == 1:
                    one_img_data_obj = [obj_name_data[i]]
                else:
                    one_img_data_obj = obj_name_data[i][box_order]
                new_data.append(one_img_data)
                image = env.label2image(one_img_data,one_img_data_obj)
                # image_list.append(image[..., :3])
                plt.imsave(images_log_f_path+f"{solu_i}.jpg",image[...,:3])
            # image_comb = np.hstack((image_list[0],image_list[1]))

            # cv2.imwrite(images_log_path+'%d.png'%j,image_comb)
            # image_list = []
            #
            # cv2.namedWindow('zzz', 0)
            # cv2.resizeWindow('zzz', 1280, 960)
            # image= cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # cv2.imshow("zzz", image)
            # data_reshape = data[j].reshape(-1, 7)
            # pred_pos = torch.tensor(data_reshape[:, :2], device=device).unsqueeze(0)
            # length_width = torch.tensor(data_reshape[:, 2:4], device=device).unsqueeze(0)
            #
            # cv2.waitKey()
            # cv2.destroyAllWindows()

