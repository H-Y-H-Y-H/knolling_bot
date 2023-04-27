import numpy as np
from stl import mesh
import pyrealsense2 as rs
import math

class Sort_objects():
    
    def __init__(self):

        # self.correct = []
        # self.cube_2x2 = [0.016, 0.016, 0.012]
        # self.cube_2x3 = [0.024, 0.016, 0.012]
        # self.cube_2x4 = [0.032, 0.016, 0.012]
        # # self.pencil = [0.01518, 0.09144, 0.01524]
        #
        # self.correct.append(self.cube_2x2)
        # self.correct.append(self.cube_2x3)
        # self.correct.append(self.cube_2x4)
        # # self.correct.append(self.pencil)
        # self.correct = np.asarray(self.correct).reshape(-1, 3)

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

        self.error_rate = 0.001

    def get_data_virtual(self, order_kinds, lego_num):
        
        #! don't motify
        names = globals()
        xyz_list = []
        pos_list = []
        ori_list = []
        # print(lego_num)
        for i in range(len(lego_num)):
            for j in range(lego_num[i]):
                xyz_list.append(self.correct[i])

        # for i in range(self.num_2x2):
        #     names[f'cube_{i}_dimension'] = mesh.Mesh.from_file(urdf_path + 'item_0/2x2.STL')
        #     xyz_list.append(names['cube_%d_dimension' % i].max_ - names['cube_%d_dimension' % i].min_)
        # for i in range(self.num_2x3):
        #     names[f'cube_{i}_dimension'] = mesh.Mesh.from_file(urdf_path + 'item_1/2x3.STL')
        #     xyz_list.append(names['cube_%d_dimension' % i].max_ - names['cube_%d_dimension' % i].min_)
        # for i in range(self.num_2x4):
        #     names[f'cube_{i}_dimension'] = mesh.Mesh.from_file(urdf_path + 'item_2/2x4.STL')
        #     xyz_list.append(names['cube_%d_dimension' % i].max_ - names['cube_%d_dimension' % i].min_)
        # for i in range(self.num_pencil):
        #     names[f'cube_{i}_dimension'] = mesh.Mesh.from_file(urdf_path + 'item_3/%d.STL' % i)
        #     xyz_list.append(names['cube_%d_dimension' % i].max_ - names['cube_%d_dimension' % i].min_)
        xyz_list = np.asarray(xyz_list, dtype=np.float32)
        # print(xyz_list)

        return self.judge(xyz_list, pos_list, ori_list, order_kinds)
    
    def judge(self, item_xyz, item_pos, item_ori, order_kinds):

        #! initiate the number of items
        all_index = []
        new_item_xyz = []

        items_names = globals()
        for i in range(len(order_kinds)):
            items_names[f'item_{order_kinds[i]}'] = []
            items_names[f'xyz_{order_kinds[i]}'] = []
            for j in range(item_xyz.shape[0]):
                if np.abs(item_xyz[j, 0] - self.correct[order_kinds[i], 0]) < np.sum(self.correct[order_kinds[i], 0]) * self.error_rate and \
                        np.abs(item_xyz[j, 1] - self.correct[order_kinds[i], 1]) < np.sum(self.correct[order_kinds[i], 1]) * self.error_rate:
                    items_names[f'item_{order_kinds[i]}'].append(len(new_item_xyz))
                    new_item_xyz.append(list(item_xyz[j]))
                    # items_names[f'xyz_{i}'].append(list(item_xyz[j]))
            all_index.append(items_names[f'item_{order_kinds[i]}'])

        new_item_xyz = np.asarray(new_item_xyz).reshape(-1, 3)
        return new_item_xyz, item_pos, item_ori, all_index

if __name__ == '__main__':

    lego_num = np.array([1, 3, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1])
    order_kinds = np.arange(len(lego_num))
    index = np.where(lego_num == 0)
    order_kinds = np.delete(order_kinds, index)
    Sort_objects1 = Sort_objects()
    xyz_list, _, _, all_index = Sort_objects1.get_data_virtual(order_kinds, lego_num)
    print('this is xyz list\n', xyz_list)
    print(all_index)