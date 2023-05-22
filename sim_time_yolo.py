from items_real_learning import Sort_objects
from cam_obs_learning import plot_and_transform
from knolling_configuration import configuration_zzz
import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from urdfpy import URDF
import pybullet_data as pd
import math
import random
# from turdf import *
import socket
import pybullet as p
import os

manual_measure = False
movie = True

class PosePredictor(DetectionPredictor):

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes,
                                        nc=len(self.model.names))

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, shape)
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(
                Results(orig_img=orig_img,
                        path=img_path,
                        names=self.model.names,
                        boxes=pred[:, :6],
                        keypoints=pred_kpts))
        return results

def label2image(labels_data, img_index, save_urdf_path):

    is_render = False
    if is_render:
        # p.connect(p.GUI, options="--width=1280 --height=720")
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    urdf_path = './urdf/'
    # print(index_flag)
    # index_flag = index_flag.reshape(2, -1)
    labels_data = labels_data.reshape(-1, 5)
    pos_data = labels_data[:, :2]
    pos_data = np.concatenate((pos_data, np.zeros(len(pos_data)).reshape(-1, 1)), axis=1)
    lw_data = labels_data[:, 2:4]
    ori_data = labels_data[:, 4]
    ori_data = np.concatenate((np.zeros((len(ori_data), 2)), ori_data.reshape(-1, 1)), axis=1)

    p.resetSimulation()
    p.setGravity(0, 0, -9.8)

    baseid = p.loadURDF(urdf_path + "plane_zzz.urdf", basePosition=[0, -0.2, 0], useFixedBase=1,
                        flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
    # self.arm_id = p.loadURDF(self.urdf_path + "robot_arm928/robot_arm1.urdf",
    #                          basePosition=[-0.08, 0, 0.02], useFixedBase=True,
    #                          flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

    textureId = p.loadTexture(urdf_path + "img_1.png")
    p.changeDynamics(baseid, -1, lateralFriction=1, spinningFriction=1, rollingFriction=0.002, linearDamping=0.5,
                     angularDamping=0.5)
    # p.changeDynamics(self.arm_id, 7, lateralFriction=1, spinningFriction=1, rollingFriction=0, linearDamping=0, angularDamping=0)
    # p.changeDynamics(self.arm_id, 8, lateralFriction=1, spinningFriction=1, rollingFriction=0, linearDamping=0, angularDamping=0)
    p.changeVisualShape(baseid, -1, textureUniqueId=textureId,
                        rgbaColor=[np.random.uniform(0.9, 1), np.random.uniform(0.9, 1), np.random.uniform(0.9, 1),
                                   1])

    ################### recover urdf boxes based on lw_data ###################
    boxes = []
    xyz_list = []
    new_pos_data = []
    new_ori_data = []
    # for i in range(len(index_flag[0])):
    #     boxes.append(URDF.load('../urdf/box_generator/box_%d.urdf' % index_flag[0, i]))
    #     xyz_list.append(boxes[i].links[0].visuals[0].geometry.box.size)

    temp_box = URDF.load('./urdf/box_generator/template.urdf')
    save_urdf_path_one_img = save_urdf_path + 'img_%d/' % img_index
    os.makedirs(save_urdf_path_one_img, exist_ok=True)
    for i in range(len(lw_data)):
        temp_box.links[0].collisions[0].origin[2, 3] = 0
        length = lw_data[i, 0]
        width = lw_data[i, 1]
        height = 0.012
        temp_box.links[0].visuals[0].geometry.box.size = [length, width, height]
        temp_box.links[0].collisions[0].geometry.box.size = [length, width, height]
        temp_box.links[0].visuals[0].material.color = [np.random.random(), np.random.random(), np.random.random(), 1]
        temp_box.save(save_urdf_path_one_img + 'box_%d.urdf' % (i))

    lego_idx = []
    for i in range(len(lw_data)):
        print(f'this is matching urdf{i}')
        print(pos_data[i])
        print(lw_data[i])
        print(ori_data[i])
        pos_data[i, 2] += 0.006
        lego_idx.append(p.loadURDF(save_urdf_path_one_img + 'box_%d.urdf' % (i),
                       basePosition=pos_data[i],
                       baseOrientation=p.getQuaternionFromEuler(ori_data[i]), useFixedBase=False,
                       flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
    ################### recover urdf boxes based on lw_data ###################

    camera_parameters = {
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
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.15, 0, 0],
        distance=0.4,
        yaw=90,
        pitch=-90,
        roll=0,
        upAxisIndex=2)
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=camera_parameters['fov'],
        aspect=camera_parameters['width'] / camera_parameters['height'],
        nearVal=camera_parameters['near'],
        farVal=camera_parameters['far'])


    (width, length, image, _, _) = p.getCameraImage(width=640,
                                                    height=480,
                                                    viewMatrix=view_matrix,
                                                    projectionMatrix=projection_matrix,
                                                    renderer=p.ER_BULLET_HARDWARE_OPENGL)
    image = image[..., :3]
    # print('this is shape of image', image.shape)
    # image = np.transpose(image, (2, 0, 1))
    # temp = image[:, :, 2]
    # image[:, :, 2] = image[:, :, 0]
    # image[:, :, 0] = temp
    # cv2.namedWindow('zzz', 0)
    # cv2.imshow("zzz", image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return image

def label_sort(results):
    item_pos = results[:, :2]
    item_lw = results[:, 2:4]
    item_ori = results[:, 4]

    ##################### generate customize boxes based on the result of yolo ######################
    temp_box = URDF.load('./urdf/box_generator/template.urdf')
    for i in range(len(results)):
        temp_box.links[0].collisions[0].origin[2, 3] = 0
        length = item_lw[i, 0]
        width = item_lw[i, 1]
        height = 0.012
        temp_box.links[0].visuals[0].geometry.box.size = [length, width, height]
        temp_box.links[0].collisions[0].geometry.box.size = [length, width, height]
        temp_box.save('./urdf/knolling_box/knolling_box_%d.urdf' % i)
    ##################### generate customize boxes based on the result of yolo ######################

    category_num = int(area_num * ratio_num + 1)
    s = item_lw[:, 0] * item_lw[:, 1]
    s_min, s_max = np.min(s), np.max(s)
    s_range = np.linspace(s_max, s_min, int(area_num + 1))
    lw_ratio = item_lw[:, 0] / item_lw[:, 1]
    ratio_min, ratio_max = np.min(lw_ratio), np.max(lw_ratio)
    ratio_range = np.linspace(ratio_max, ratio_min, int(ratio_num * 2 + 1))
    ratio_range_high = np.linspace(ratio_max, 1, int(ratio_num + 1))
    ratio_range_low = np.linspace(1 / ratio_max, 1, int(ratio_num + 1))

    # ! initiate the number of items
    all_index = []
    new_item_lw = []
    new_item_pos = []
    new_item_ori = []
    new_urdf_index = []
    transform_flag = []
    rest_index = np.arange(len(item_lw))
    index = 0

    for i in range(area_num):
        for j in range(ratio_num):
            kind_index = []
            for m in range(len(item_lw)):
                if m not in rest_index:
                    continue
                elif s_range[i] >= s[m] >= s_range[i + 1]:
                    # if ratio_range_high[j] >= lw_ratio[m] >= ratio_range_high[j + 1]:
                    transform_flag.append(0)
                    # print(f'boxes{m} matches in area{i}, ratio{j}!')
                    kind_index.append(index)
                    new_item_lw.append(item_lw[m])
                    new_item_pos.append(item_pos[m])
                    new_item_ori.append(item_ori[m])
                    new_urdf_index.append(m)
                    index += 1
                    rest_index = np.delete(rest_index, np.where(rest_index == m))
            if len(kind_index) != 0:
                all_index.append(kind_index)

    new_item_lw = np.asarray(new_item_lw).reshape(-1, 2)
    new_item_pos = np.asarray(new_item_pos)
    new_item_ori = np.asarray(new_item_ori)
    new_item_pos = np.concatenate((new_item_pos, np.zeros((len(new_item_pos), 1))), axis=1)
    new_item_ori = np.concatenate((np.zeros((len(new_item_pos), 2)), new_item_ori.reshape(len(new_item_ori), 1)),
                                  axis=1)
    transform_flag = np.asarray(transform_flag)
    if len(rest_index) != 0:
        # we should implement the rest of boxes!
        rest_xyz = item_lw[rest_index]
        new_item_lw = np.concatenate((new_item_lw, rest_xyz), axis=0)
        all_index.append(list(np.arange(index, len(item_lw))))
        transform_flag = np.append(transform_flag, np.zeros(len(item_lw) - index))

    new_item_lw = np.concatenate((new_item_lw, np.array([0.012] * len(new_item_lw)).reshape(len(new_item_lw), 1)),
                                 axis=1)
    return new_item_lw, new_item_pos, new_item_ori, all_index, transform_flag, new_urdf_index

def predict(cfg=DEFAULT_CFG, use_python=False, data_path=None, model_path=None):

    sim_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/datasets/yolo_dataset_test_519/'
    num_sim = 4000

    for i in range(num_sim):
        origin_img = cv2.imread(sim_root + 'images/val/%012d.png' % int(i + 16000))
        sim_label = np.loadtxt(sim_root + 'labels/val/%012d.txt' % int(i + 16000))
        sim_result = []
        for j in range(len(sim_label)):
            tar_xylw = np.copy(sim_label[j, 1:5])
            tar_keypoints = np.copy((sim_label[j, 5:]).reshape(-1, 3)[:, :2])
            tar_label = '0: "target"'

            # plot target
            origin_img, t_result = plot_and_transform(im=origin_img, box=tar_xylw, label='0: target',
                                               color=(255, 255, 0), txt_color=(255, 255, 255), index=j,
                                               scaled_xylw=tar_xylw, keypoints=tar_keypoints, use_xylw=False,
                                               truth_flag=True)
            sim_result.append(t_result)

        sim_result = np.asarray(sim_result)

        #######################################################################################################################
        distance = np.linalg.norm(sim_result[:, :2] - origin_point, axis=1)
        order = np.argsort(distance)
        sim_result = sim_result[order]
        print('yolo sim_result after changing the sequence', sim_result)
        #######################################################################################################################

        cv2.imwrite(sim_root + 'yolo_result_images/%012d.png' % int(i + 16000), origin_img)

        new_item_lw, new_item_pos_before, new_item_ori_before, all_index, transform_flag, new_urdf_index = label_sort(
            sim_result)

        np.savetxt(sim_root + 'labels_before_knolling/%012d.txt' % int(i + 16000), sim_result)

        calculate_reorder = configuration_zzz(new_item_lw, all_index, gap_item, gap_block,
                                              transform_flag, configuration,
                                              item_odd_prevent, block_odd_prevent, upper_left_max,
                                              forced_rotate_box)

        # determine the center of the tidy configuration
        items_pos_list, items_ori_list = calculate_reorder.calculate_block()
        items_pos_list = items_pos_list + total_offset

        ################################### change order!!!!!!!!!!!!!!!!!!!!!!!!!########################3
        distance = np.linalg.norm(items_pos_list[:, :2] - origin_point, axis=1)
        order = np.argsort(distance)
        items_pos_list = items_pos_list[order]
        items_ori_list = items_ori_list[order]
        new_item_lw = new_item_lw[order]
        new_item_pos_before = new_item_pos_before[order]
        new_item_ori_before = new_item_ori_before[order]
        items_ori_list_arm = np.copy(items_ori_list)

        data_after = np.concatenate((items_pos_list[:, :2], new_item_lw[:, :2], items_ori_list[:, 2].reshape(-1, 1)), axis=1)

        # np.savetxt(sim_root + 'labels_after_knolling/%012d.txt' % int(i + 16000), data_after)

        save_urdf_path = sim_root + 'sim_urdf/'
        os.makedirs(save_urdf_path, exist_ok=True)
        # image = label2image(data_after, img_index=i, save_urdf_path=save_urdf_path)
        # cv2.imwrite(sim_root + 'images_after_knolling/%012d.png' % int(i + 16000), image)


if __name__ == '__main__':

    area_num = 2
    ratio_num = 1
    lego_num = None
    real_time_flag = True
    # evaluations = 77

    gap_item = 0.015
    gap_block = 0.015

    item_odd_prevent = True
    block_odd_prevent = True
    upper_left_max = True
    forced_rotate_box = False
    configuration = None

    num_collect_img_start = 0
    num_collect_img_end = 100
    num_box_one_img = 1
    total_offset = [0.016, -0.17 + 0.016, 0]

    origin_point = np.array([0, -0.2])
    predict()