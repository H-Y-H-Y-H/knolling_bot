import numpy as np
# import pyrealsense2 as rs
import math
from urdfpy import URDF

class Sort_objects():
    
    def __init__(self):
        pass
    def get_data_virtual(self, area_num, ratio_num, box_num, boxes_index):

        # boxes = []
        # xyz_list = []
        # for i in range(len(boxes_index)):
        #     boxes.append(URDF.load('../urdf/box_generator/box_%d.urdf' % boxes_index[i]))
        #     xyz_list.append(boxes[i].links[0].visuals[0].geometry.box.size)
        # pos_list = []
        # ori_list = []
        # xyz_list = np.asarray(xyz_list, dtype=np.float32)

        length_range = np.round(np.random.uniform(0.016, 0.048, size=(box_num, 1)), decimals=3)
        width_range = np.round(np.random.uniform(0.016, np.minimum(length_range, 0.036), size=(box_num, 1)), decimals=3)
        height_range = np.round(np.random.uniform(0.010, 0.020, size=(box_num, 1)), decimals=3)
        pos_list = []
        ori_list = []
        xyz_list = np.concatenate((length_range, width_range, height_range), axis=1)

        return self.judge(xyz_list, pos_list, ori_list, area_num, ratio_num)
    
    def judge(self, item_xyz, item_pos, item_ori, area_num, ratio_num):

        category_num = int(area_num * ratio_num + 1)
        s = item_xyz[:, 0] * item_xyz[:, 1]
        s_min, s_max = np.min(s), np.max(s)
        s_range = np.linspace(s_max, s_min, int(area_num + 1))
        lw_ratio = item_xyz[:, 0] / item_xyz[:, 1]
        ratio_min, ratio_max = np.min(lw_ratio), np.max(lw_ratio)
        ratio_range = np.linspace(ratio_max, ratio_min, int(ratio_num + 1))
        # ratio_range_high = np.linspace(ratio_max, 1, int(ratio_num + 1))
        # ratio_range_low = np.linspace(1 / ratio_max, 1, int(ratio_num + 1))

        #! initiate the number of items
        all_index = []
        new_item_xyz = []
        transform_flag = []
        rest_index = np.arange(len(item_xyz))
        index = 0


        for i in range(area_num):
            for j in range(ratio_num):
                kind_index = []
                for m in range(len(item_xyz)):
                    if m not in rest_index:
                        continue
                    else:
                        if s_range[i] >= s[m] >= s_range[i + 1]:
                            if ratio_range[j] >= lw_ratio[m] >= ratio_range[j + 1]:
                                transform_flag.append(0)
                                # print(f'boxes{m} matches in area{i}, ratio{j}!')
                                kind_index.append(index)
                                new_item_xyz.append(item_xyz[m])
                                index += 1
                                rest_index = np.delete(rest_index, np.where(rest_index == m))
                        # if ratio_range[j] >= lw_ratio[m] >= ratio_range[j + 1]:
                        #     transform_flag.append(0)
                        #     print(f'boxes{m} matches in area{i}, ratio{j}!')
                        #     kind_index.append(index)
                        #     new_item_xyz.append(item_xyz[m])
                        #     index += 1
                        #     rest_index = np.delete(rest_index, np.where(rest_index == m))
                        # elif ratio_range_low[j] <= lw_ratio[m] <= ratio_range_low[j + 1]:
                        #     transform_flag.append(1)
                        #     print(f'boxes{m} matches in area{i}, ratio{j}, remember to rotate the ori after knolling!')
                        #     item_xyz[m, [0, 1]] = item_xyz[m, [1, 0]]
                        #     kind_index.append(index)
                        #     new_item_xyz.append(item_xyz[m])
                        #     index += 1
                        #     rest_index = np.delete(rest_index, np.where(rest_index == m))

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

        return new_item_xyz, item_pos, item_ori, all_index, transform_flag

if __name__ == '__main__':

    lego_num = np.array([1, 3, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1])
    order_kinds = np.arange(len(lego_num))
    index = np.where(lego_num == 0)
    order_kinds = np.delete(order_kinds, index)
    Sort_objects1 = Sort_objects()
    xyz_list, _, _, all_index = Sort_objects1.get_data_virtual(order_kinds, lego_num)
    print('this is xyz list\n', xyz_list)
    print(all_index)