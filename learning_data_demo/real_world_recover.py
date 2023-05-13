import numpy as np
import pyrealsense2 as rs
import pybullet_data as pd
import pybullet as p
import torch
import random
import os
from urdfpy import URDF
import cv2

torch.manual_seed(42)
np.random.seed(202)
random.seed(202)

class Arm:

    def __init__(self, is_render=True):

        self.kImageSize = {'width': 480, 'height': 480}
        self.urdf_path = '../urdf/'
        self.pybullet_path = pd.getDataPath()
        self.is_render = is_render
        if self.is_render:
            p.connect(p.GUI, options="--width=1280 --height=720")
            # p.connect(p.GUI)
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

        if random.uniform(0,1) > 0.5:
            p.configureDebugVisualizer(lightPosition=[random.randint(1,3), random.randint(1,2), 5])
        else:
            p.configureDebugVisualizer(lightPosition=[random.randint(1,3), random.randint(-2, -1), 5])
        p.configureDebugVisualizer(lightPosition=[random.randint(1, 3), random.randint(1, 2), 5],
                                   shadowMapResolution=8192, shadowMapIntensity=np.random.randint(5, 8) / 10)
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                     cameraYaw=45,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[0.1, 0, 0])
        p.setAdditionalSearchPath(pd.getDataPath())

    def get_parameters(self, lego_num=None, area_num=None, ratio_num=None, boxes_index=None,
                       total_offset=None, evaluations=None,
                       gap_item=0.03, gap_block=0.02,
                       real_operate=False, obs_order='1',
                       random_offset = False, check_detection_loss=None, obs_img_from=None, use_lego_urdf=True,
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
        self.num_list = lego_num
        self.check_detection_loss = check_detection_loss
        self.obs_img_from = obs_img_from
        self.use_lego_urdf = use_lego_urdf
        self.boxes_index = boxes_index
        self.evaluations = evaluations
        self.configuration = None
        self.item_odd_prevent = item_odd_prevent
        self.block_odd_prevent = block_odd_prevent
        self.upper_left_max = upper_left_max
        self.forced_rotate_box = forced_rotate_box

        self.correct = np.array([[0.016, 0.016, 0.012],
                                 [0.020, 0.016, 0.012],
                                 [0.020, 0.020, 0.012],
                                 [0.024, 0.016, 0.012],
                                 [0.024, 0.020, 0.012],
                                 [0.024, 0.024, 0.012],
                                 [0.028, 0.016, 0.012],
                                 [0.028, 0.020, 0.012],
                                 [0.028, 0.024, 0.012],
                                 [0.032, 0.016, 0.012],
                                 [0.032, 0.020, 0.012],
                                 [0.032, 0.024, 0.012]])

    def get_obs(self, order):

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

    def recover(self, labels_data, urdf_path, img_indx, num_lego):

        lego_xy = np.concatenate((labels_data[:, :2], np.zeros((len(labels_data), 1))), axis=1)
        lego_lw = labels_data[:, 2:4]
        lego_ori = np.concatenate((np.zeros((len(labels_data), 2)), labels_data[:, 4].reshape(-1, 1)), axis=1)

        temp_box = URDF.load('../urdf/box_generator/template.urdf')
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        temp_img_urdf_path = urdf_path + 'num_%d/img_%d' % (num_lego, img_indx)
        os.makedirs(temp_img_urdf_path, exist_ok=True)

        for i in range(len(labels_data)):
            temp_box.links[0].collisions[0].origin[2, 3] = 0
            length = lego_lw[i, 0]
            width = lego_lw[i, 1]
            height = 0.012
            temp_box.links[0].visuals[0].geometry.box.size = [length, width, height]
            temp_box.links[0].collisions[0].geometry.box.size = [length, width, height]
            temp_box.save(urdf_path + 'img_%d/lego_demo_%d.urdf' % (img_indx, i))

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

        baseid = p.loadURDF(self.urdf_path + "plane_1.urdf", basePosition=[0, -0.2, 0], useFixedBase=1,
                            flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        self.arm_id = p.loadURDF(os.path.join(self.urdf_path, "robot_arm928/robot_arm1.urdf"),
                                 basePosition=[-0.08, 0, 0.02], useFixedBase=True,
                                 flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        textureId = p.loadTexture(self.urdf_path + "img_1.png")
        p.changeDynamics(baseid, -1, lateralFriction=1, spinningFriction=1, rollingFriction=0.002, linearDamping=0.5, angularDamping=0.5)
        p.changeDynamics(self.arm_id, 7, lateralFriction=1, spinningFriction=1, rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.arm_id, 8, lateralFriction=1, spinningFriction=1, rollingFriction=0, linearDamping=0, angularDamping=0)
        # p.changeVisualShape(baseid, -1, textureUniqueId=textureId,rgbaColor=[np.random.uniform(0.9,1), np.random.uniform(0.9,1),np.random.uniform(0.9,1), 1])
        p.changeVisualShape(baseid, -1, textureUniqueId=textureId)

        for i in range(100):
            p.stepSimulation()

        # get the standard xyz and corresponding index from files in the computer
        self.lego_idx = []

        # these data has defined in function change_config, we don't need to define them twice!!!
        # self.xyz_list, pos_before, ori_before, self.all_index, self.kind = items_sort.get_data_real()
        num_lego = 0
        sim_pos = np.copy(lego_xy)
        sim_pos[:, 2] += 0.006
        for i in range(len(lego_lw)):
            self.lego_idx.append(
                p.loadURDF(urdf_path + 'img_%d/lego_demo_%d.urdf' % (img_indx, i),
                           basePosition=sim_pos[i],
                           baseOrientation=p.getQuaternionFromEuler(lego_ori[i]), useFixedBase=False,
                           flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
            r = np.random.uniform(0, 0.9)
            g = np.random.uniform(0, 0.9)
            b = np.random.uniform(0, 0.9)
            p.changeVisualShape(self.lego_idx[num_lego], -1, rgbaColor=(r, g, b, 1))
            num_lego += 1

        # data_before = np.concatenate((self.pos_before[:, :2], self.xyz_list[:, :2], self.ori_before[:, 2].reshape(-1, 1)), axis=1)

        return self.get_obs('images')

if __name__ == '__main__':

    env = Arm(is_render=True)

    num_lego = 6
    num_image = 5

    target_path = '../learning_data_demo/cfg_0/'
    img_sim_path = target_path + 'images_after_sim/num_%d/' % num_lego
    os.makedirs(img_sim_path, exist_ok=True)
    temp_urdf_path = target_path + 'urdf/num_%d/' % num_lego
    os.makedirs(temp_urdf_path, exist_ok=True)

    data = np.loadtxt(target_path + 'labels_after/num_%d.txt' % (num_lego))


    for i in range(num_image):
        one_img_data = data[i].reshape(-1, 5)
        # for i in range(evaluations):
        image = env.recover(one_img_data, temp_urdf_path, i, num_lego)

        image = image[..., :3]
        # print('this is shape of image', image.shape)
        # image = np.transpose(image, (2, 0, 1))
        # temp = image[:, :, 2]
        # image[:, :, 2] = image[:, :, 0]
        # image[:, :, 0] = temp
        cv2.namedWindow('zzz', 0)
        cv2.imshow("zzz", image)
        cv2.waitKey()
        cv2.destroyAllWindows()