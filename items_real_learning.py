import numpy as np
from stl import mesh
from cam_obs_learning import *
import pyrealsense2 as rs
import math
from urdfpy import URDF

class Sort_objects():
    
    def __init__(self):

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

        self.error_rate = 0.05

    def get_data_virtual(self, area_num, ratio_num, lego_num, boxes_index):

        boxes = []
        xyz_list = []
        # for i in range(lego_num):
        #     boxes.append(URDF.load('../urdf/box_generator/box_%d.urdf' % i))
        #     xyz_list.append(boxes[i].links[0].visuals[0].geometry.box.size)
        # print(boxes_index)
        for i in range(len(boxes_index)):
            # print(boxes_index[i])
            boxes.append(URDF.load('./urdf/box_generator/box_%d.urdf' % boxes_index[i]))
            xyz_list.append(boxes[i].links[0].visuals[0].geometry.box.size)

        pos_list = []
        ori_list = []
        xyz_list = np.asarray(xyz_list, dtype=np.float32)
        # print(xyz_list)

        return self.judge(xyz_list, pos_list, ori_list, area_num, ratio_num, boxes_index)

    def get_data_real(self, area_num, ratio_num, lego_num):
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

        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
        # Start streaming
        pipeline.start(config)

        frames = None
        for _ in range(100):
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            # depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())

        color_colormap_dim = color_image.shape
        resized_color_image = color_image
        img_path = 'Test_images/image_real'
        cv2.imwrite(img_path + '.png', resized_color_image)
        cv2.waitKey(1)

        # cv2.waitKey(1)

        # img = cv2.imread("read_real_cam.png")

        # structure of results: x, y, length, width, ori
        results = yolov8_predict(img_path=img_path,
                                 real_flag=True,
                                 target=None)

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
                        if ratio_range[j] >= lw_ratio[m] >= ratio_range[j + 1]:
                            transform_flag.append(0)
                            # print(f'boxes{m} matches in area{i}, ratio{j}!')
                            kind_index.append(index)
                            new_item_lw.append(item_lw[m])
                            new_item_pos.append(item_pos[m])
                            new_item_ori.append(item_ori[m])
                            new_urdf_index.append(m)
                            index += 1
                            rest_index = np.delete(rest_index, np.where(rest_index == m))
                        elif ratio_range[ratio_num * 2 - j] <= lw_ratio[m] <= ratio_range[ratio_num * 2 - j - 1]:
                            transform_flag.append(1)
                            # print(f'boxes{m} matches in area{i}, ratio{j}, remember to rotate the ori after knolling!')
                            item_lw[m, [0, 1]] = item_lw[m, [1, 0]]
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
        new_item_ori = np.concatenate((np.zeros((len(new_item_pos), 2)), new_item_ori.reshape(len(new_item_ori), 1)), axis=1)
        transform_flag = np.asarray(transform_flag)
        if len(rest_index) != 0:
            # we should implement the rest of boxes!
            rest_xyz = item_lw[rest_index]
            new_item_lw = np.concatenate((new_item_lw, rest_xyz), axis=0)
            all_index.append(list(np.arange(index, len(item_lw))))
            transform_flag = np.append(transform_flag, np.zeros(len(item_lw) - index))

        return new_item_lw, new_item_pos, new_item_ori, all_index, transform_flag, new_urdf_index

        # all_index = []
        # new_xyz_list = []
        # kind = []
        # new_results = []
        # z = 0.006
        # roll = 0
        # pitch = 0
        # num = 0
        # for i in range(len(self.correct)):
        #     kind_index = []
        #     for j in range(len(results)):
        #         # if np.linalg.norm(self.correct[i][:2] - results[j][3:5]) < 0.003:
        #         if np.abs(self.correct[i][0] - results[j][0]) < 0.002 and np.abs(self.correct[i][1] - results[j][1]) < 0.002:
        #             kind_index.append(num)
        #             new_xyz_list.append(self.correct[i])
        #             num += 1
        #             if i in kind:
        #                 pass
        #             else:
        #                 kind.append(i)
        #             new_results.append(results[j])
        #         else:
        #             pass
        #     if len(kind_index) != 0:
        #         all_index.append(kind_index)
        #
        # new_xyz_list = np.asarray(new_xyz_list)
        # new_results = np.asarray(new_results)
        # kind = np.asarray(kind)
        # print('this is kind', kind)
        # print('this is all index', all_index)
        # print(new_xyz_list)
        #
        # # 按照234重新将result排序
        # pos_before = np.concatenate((new_results[:, :2], np.zeros(len(new_results).reshape(-1, 1))), axis=1)
        # ori_before = np.concatenate((np.zeros((len(new_results)), 2), new_results[:, 4]), axis=1)
        #
        # return new_xyz_list, pos_before, ori_before, all_index, kind

    def judge(self, item_xyz, item_pos, item_ori, area_num, ratio_num, boxes_index):

        category_num = int(area_num * ratio_num + 1)
        s = item_xyz[:, 0] * item_xyz[:, 1]
        s_min, s_max = np.min(s), np.max(s)
        s_range = np.linspace(s_max, s_min, int(area_num + 1))
        lw_ratio = item_xyz[:, 0] / item_xyz[:, 1]
        ratio_min, ratio_max = np.min(lw_ratio), np.max(lw_ratio)
        ratio_range = np.linspace(ratio_max, ratio_min, int(ratio_num * 2 + 1))

        # ! initiate the number of items
        all_index = []
        new_item_xyz = []
        transform_flag = []
        new_urdf_index = []
        rest_index = np.arange(len(item_xyz))
        index = 0

        for i in range(area_num):
            for j in range(ratio_num):
                kind_index = []
                for m in range(len(item_xyz)):
                    if m not in rest_index:
                        continue
                    elif s_range[i] >= s[m] >= s_range[i + 1]:
                        if ratio_range[j] >= lw_ratio[m] >= ratio_range[j + 1]:
                            transform_flag.append(0)
                            # print(f'boxes{m} matches in area{i}, ratio{j}!')
                            kind_index.append(index)
                            new_item_xyz.append(item_xyz[m])
                            index += 1
                            rest_index = np.delete(rest_index, np.where(rest_index == m))
                            new_urdf_index.append(boxes_index[m])
                        elif ratio_range[ratio_num * 2 - j] <= lw_ratio[m] <= ratio_range[ratio_num * 2 - j - 1]:
                            transform_flag.append(1)
                            # print(f'boxes{m} matches in area{i}, ratio{j}, remember to rotate the ori after knolling!')
                            item_xyz[m, [0, 1]] = item_xyz[m, [1, 0]]
                            kind_index.append(index)
                            new_item_xyz.append(item_xyz[m])
                            index += 1
                            rest_index = np.delete(rest_index, np.where(rest_index == m))
                            new_urdf_index.append(boxes_index[m])
                if len(kind_index) != 0:
                    all_index.append(kind_index)

        new_item_xyz = np.asarray(new_item_xyz).reshape(-1, 3)
        transform_flag = np.asarray(transform_flag)
        if len(rest_index) != 0:
            # we should implement the rest of boxes!
            rest_xyz = item_xyz[rest_index]
            new_item_xyz = np.concatenate((new_item_xyz, rest_xyz), axis=0)
            all_index.append(list(np.arange(index, len(item_xyz))))
            transform_flag = np.append(transform_flag, np.zeros(len(item_xyz) - index))

        return new_item_xyz, item_pos, item_ori, all_index, transform_flag, new_urdf_index

if __name__ == '__main__':

    lego_num = np.array([1, 3, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1])
    order_kinds = np.arange(len(lego_num))
    index = np.where(lego_num == 0)
    order_kinds = np.delete(order_kinds, index)
    Sort_objects1 = Sort_objects()
    xyz_list, _, _, all_index = Sort_objects1.get_data_virtual(order_kinds, lego_num)
    print('this is xyz list\n', xyz_list)
    print(all_index)