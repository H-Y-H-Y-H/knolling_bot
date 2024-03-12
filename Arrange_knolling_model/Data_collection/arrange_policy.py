import numpy as np
import cv2
import pybullet as p
import pybullet_data as pd
from utils import *
import copy

class configuration_zzz():

    def __init__(self, xyz_list, all_index, gap_item, gap_block, all_cls,
                 item_odd_prevent, block_odd_prevent, upper_left_max, forced_rotate_box, iteration_time):

        self.xyz_list = xyz_list
        self.all_index = all_index
        self.gap_item = gap_item
        self.gap_block = gap_block
        self.all_cls = all_cls
        # self.transform_flag = transform_flag
        self.item_odd_prevent = item_odd_prevent
        self.block_odd_prevent = block_odd_prevent
        self.upper_left_max = upper_left_max
        self.forced_rotate_box = forced_rotate_box
        self.iteration_time = iteration_time

    def calculate_items(self, item_num, item_xyz):

        min_xy = np.ones(2) * 100
        best_item_config = []
        best_item_sequence = []
        item_odd_flag = False
        all_item_x = 100
        all_item_y = 100
        best_iteration = 0
        xy_array = []
        iteration_array = []

        for iter in range(self.iteration_time):

            fac = []  # 定义一个列表存放因子
            for i in range(1, item_num + 1):
                if item_num % i == 0:
                    fac.append(i)
                    continue
            # fac = fac[::-1]

            if self.item_odd_prevent == True:
                if item_num % 2 != 0 and len(fac) == 2 and item_num >=5:  # its odd! we should generate the factor again!
                    item_num += 1
                    item_odd_flag = True
                    fac = []  # 定义一个列表存放因子
                    for i in range(1, item_num + 1):
                        if item_num % i == 0:
                            fac.append(i)
                            continue

            item_sequence = np.random.choice(len(item_xyz), len(item_xyz), replace=False)
            if item_odd_flag == True:
                item_sequence = np.append(item_sequence, item_sequence[-1])

            for j in range(len(fac)):
                # if item_num == 3:
                #     item_num_row = 1
                #     item_num_column = 3
                # else:
                item_num_row = int(fac[j])
                item_num_column = int(item_num / item_num_row)
                item_sequence = item_sequence.reshape(item_num_row, item_num_column)
                item_min_x = 0
                item_min_y = 0

                for r in range(item_num_row):
                    new_row = item_xyz[item_sequence[r, :]]
                    item_min_x = item_min_x + np.max(new_row, axis=0)[0]



                for c in range(item_num_column):
                    new_column = item_xyz[item_sequence[:, c]]
                    item_min_y = item_min_y + np.max(new_column, axis=0)[1]

                item_min_x = item_min_x + (item_num_row - 1) * self.gap_item
                item_min_y = item_min_y + (item_num_column - 1) * self.gap_item

                if item_min_x + item_min_y < all_item_x + all_item_y:
                    best_item_config = [item_num_row, item_num_column]
                    best_item_sequence = item_sequence
                    all_item_x = item_min_x
                    all_item_y = item_min_y
                    best_iteration = iter
                    min_xy = np.array([all_item_x, all_item_y])
                    xy_array.append(np.sum(min_xy))
                    iteration_array.append(iter)
                    # print(f'interation {self.best_iteration}, min_xy {np.sum(self.min_xy)}')
        # print(f'In iteration {best_iteration}, the min xy is {min_xy}, total is {np.sum(min_xy)}')
        # xy_array = 1 / np.asarray(xy_array)
        # iteration_array = np.asarray(iteration_array)
        # selected_index = np.where(xy_array > (1 / np.sum(min_xy) * 0.9))[0]
        # selected_iter = iteration_array[selected_index]
        # print(f'In iteration {selected_iter[0]}, the xy has met the condition, is {1 / xy_array[selected_index[0]]}')
        return min_xy, best_item_config, item_odd_flag, best_item_sequence

    def calculate_block(self):  # first: calculate, second: reorder!

        min_result = []
        best_config = []
        item_odd_list = []

        ################## zzz add sequence ###################
        item_sequence_list = []
        ################## zzz add sequence ###################

        for i in range(len(self.all_index)):
            item_index = self.all_index[i]
            item_xyz = self.xyz_list[item_index, :]
            item_num = len(item_index)
            xy, config, odd, item_sequence = self.calculate_items(item_num, item_xyz)
            # print(f'this is min xy {xy}')
            min_result.append(list(xy))
            # print(f'this is the best item config\n {config}')
            best_config.append(list(config))
            item_odd_list.append(odd)
            item_sequence_list.append(item_sequence)
        min_result = np.asarray(min_result).reshape(-1, 2)
        best_config = np.asarray(best_config).reshape(-1, 2)
        item_odd_list = np.asarray(item_odd_list)
        # print('this is item sequence list', item_sequence_list)
        # item_sequence_list = np.asarray(item_sequence_list, dtype=object)

        # print(best_config)

        if self.upper_left_max == True:
            # reorder the block based on the min_xy 哪个block面积大哪个在前
            s_block_sequence = np.argsort(min_result[:, 0] * min_result[:, 1])[::-1]
            new_all_index = []
            for i in s_block_sequence:
                new_all_index.append(self.all_index[i])
            self.all_index = new_all_index.copy()
            min_result = min_result[s_block_sequence]
            best_config = best_config[s_block_sequence]
            item_odd_list = item_odd_list[s_block_sequence]
            item_sequence_list = [item_sequence_list[i] for i in s_block_sequence]
            # item_sequence_list = item_sequence_list[s_block_sequence]
            # reorder the block based on the min_xy 哪个block面积大哪个在前

        # 安排总的摆放
        iteration = 100
        all_num = best_config.shape[0]
        all_x = 100
        all_y = 100
        odd_flag = False
        best_min_xy = np.inf

        fac = []  # 定义一个列表存放因子
        for i in range(1, all_num + 1):
            if all_num % i == 0:
                fac.append(i)
                continue
        # fac = fac[::-1]

        if self.block_odd_prevent == True:
            if all_num % 2 != 0 and len(fac) == 2:  # its odd! we should generate the factor again!
                all_num += 1
                odd_flag = True
                fac = []  # 定义一个列表存放因子
                for i in range(1, all_num + 1):
                    if all_num % i == 0:
                        fac.append(i)
                        continue

        for i in range(iteration):

            if self.upper_left_max == True:
                sequence = np.concatenate((np.array([0]), np.random.choice(best_config.shape[0] - 1, size=len(self.all_index) - 1, replace=False) + 1))
            else:
                sequence = np.random.choice(best_config.shape[0], size=len(self.all_index), replace=False)
            # sequence = np.arange(len(self.all_index))

            if odd_flag == True:
                sequence = np.append(sequence, sequence[-1])
            else:
                pass
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

                # the zero or 90 should permanently be 0
                for r in range(num_row):
                    for c in range(num_column):
                        new_row = min_xy[sequence[r][c]]
                        if self.forced_rotate_box == True:
                            if new_row[0] > new_row[1]:
                                zero_or_90 = 90
                        else:
                            zero_or_90 = np.random.choice(np.array([0, 90]))
                        if zero_or_90 == 90:
                            rotate_flag[r][c] = True
                            temp = new_row[0]
                            new_row[0] = new_row[1]
                            new_row[1] = temp

                    # insert 'whether to rotate' here
                for r in range(num_row):
                    new_row = min_xy[sequence[r, :]]
                    min_x = min_x + np.max(new_row, axis=0)[0]

                for c in range(num_column):
                    new_column = min_xy[sequence[:, c]]
                    min_y = min_y + np.max(new_column, axis=0)[1]

                if min_x + min_y < all_x + all_y:
                    best_all_config = sequence
                    all_x = min_x
                    all_y = min_y
                    best_rotate_flag = rotate_flag
                    best_min_xy = np.copy(min_xy)

        # print(f'in iteration{i}, the min xy is {best_min_xy}')
        # print('this is best all sequence', best_all_config)

        return self.reorder_block(best_config, best_all_config, best_rotate_flag, best_min_xy, odd_flag, item_odd_list, item_sequence_list)

    def reorder_item(self, best_config, start_pos, index_block, item_index, item_xyz, rotate_flag, item_odd_list, item_sequence):

        # initiate the pos and ori
        # we don't analysis these imported oris
        # we directly define the ori is 0 or 90 degree, depending on the algorithm.

        # item_row = best_config[index_block][0]
        # item_column = best_config[index_block][1]
        # print(item_sequence)
        item_row = item_sequence.shape[0]
        item_column = item_sequence.shape[1]
        item_odd_flag = item_odd_list[index_block]
        if item_odd_flag == True:
            item_pos = np.zeros([len(item_index) + 1, 3])
            item_ori = np.zeros([len(item_index) + 1, 3])
            item_xyz = np.append(item_xyz, item_xyz[-1]).reshape(-1, 3)
            # index_temp = np.arange(item_pos.shape[0] - 1)
            # index_temp = np.append(index_temp, index_temp[-1]).reshape(item_row, item_column)
        else:
            item_pos = np.zeros([len(item_index), 3])
            item_ori = np.zeros([len(item_index), 3])
            # index_temp = np.arange(item_pos.shape[0]).reshape(item_row, item_column)

        # the initial position of the first items

        if rotate_flag == True:

            temp = np.copy(item_xyz[:, 0])
            item_xyz[:, 0] = item_xyz[:, 1]
            item_xyz[:, 1] = temp
            item_ori[:, 2] = 0
            # print(item_ori)
            temp = item_row
            item_row = item_column
            item_column = temp
            # index_temp = index_temp.transpose()
            item_sequence = item_sequence.transpose()
        else:
            item_ori[:, 2] = 0

        # start_pos[0] = start_pos[0] + np.max(item_xyz, axis=0)[0] / 2
        # start_pos[1] = start_pos[1] + np.max(item_xyz, axis=0)[1] / 2
        #
        #
        # for j in range(item_row):
        #     for k in range(item_column):
        #         ################### check whether to transform for each item in each block!################
        #         if self.transform_flag[item_index[index_temp[j][k]]] == 1:
        #             print(f'the index {item_index[index_temp[j][k]]} should be rotated because of transformation')
        #             item_ori[index_temp[j][k], 2] -= np.pi / 2
        #         ################### check whether to transform for each item in each block!################
        #         x_2x2 = start_pos[0] + (item_xyz[index_temp[j][k]][0]) * j + self.gap_item * j
        #         y_2x2 = start_pos[1] + (item_xyz[index_temp[j][k]][1]) * k + self.gap_item * k
        #         item_pos[index_temp[j][k]][0] = x_2x2
        #         item_pos[index_temp[j][k]][1] = y_2x2

        start_item_x = np.array([start_pos[0]])
        start_item_y = np.array([start_pos[1]])
        previous_start_item_x = start_item_x
        previous_start_item_y = start_item_y

        for m in range(item_row):
            new_row = item_xyz[item_sequence[m, :]]
            start_item_x = np.append(start_item_x,
                                     (previous_start_item_x + np.max(new_row, axis=0)[0] + self.gap_item))
            previous_start_item_x = (previous_start_item_x + np.max(new_row, axis=0)[0] + self.gap_item)
        start_item_x = np.delete(start_item_x, -1)

        for n in range(item_column):
            new_column = item_xyz[item_sequence[:, n]]
            start_item_y = np.append(start_item_y,
                                     (previous_start_item_y + np.max(new_column, axis=0)[1] + self.gap_item))
            previous_start_item_y = (previous_start_item_y + np.max(new_column, axis=0)[1] + self.gap_item)
        start_item_y = np.delete(start_item_y, -1)

        x_pos, y_pos = np.copy(start_pos)[0], np.copy(start_pos)[1]

        for j in range(item_row):
            for k in range(item_column):
                if item_odd_flag == True and j == item_row - 1 and k == item_column - 1:
                    break
                # ################### check whether to transform for each item in each block!################
                # if self.transform_flag[item_index[item_sequence[j][k]]] == 1:
                #     # print(f'the index {item_index[index_temp[j][k]]} should be rotated because of transformation')
                #     item_ori[item_sequence[j][k], 2] -= np.pi / 2
                # ################### check whether to transform for each item in each block!################
                x_pos = start_item_x[j] + (item_xyz[item_sequence[j][k]][0]) / 2
                y_pos = start_item_y[k] + (item_xyz[item_sequence[j][k]][1]) / 2
                item_pos[item_sequence[j][k]][0] = x_pos
                item_pos[item_sequence[j][k]][1] = y_pos
        if item_odd_flag == True:
            item_pos = np.delete(item_pos, -1, axis=0)
            item_ori = np.delete(item_ori, -1, axis=0)
        else:
            pass
        # print('this is the shape of item pos', item_pos.shape)
        return item_ori, item_pos

    def reorder_block(self, best_config, best_all_config, best_rotate_flag, min_xy, odd_flag, item_odd_list, item_sequence_list):

        # print(f'the best configuration of all items is\n {best_all_config}')
        # print(f'the best configuration of each kind of items is\n {best_config}')
        # print(f'the rotate of each block of items is\n {best_rotate_flag}')
        # print(f'this is the min_xy of each kind of items after rotation\n {min_xy}')

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
                if odd_flag == True and m == num_all_row - 1 and n == num_all_column - 1:
                    break  # this is the redundancy block
                item_index = self.all_index[best_all_config[m][n]]  # determine the index of blocks

                # print('try', item_index)
                item_xyz = self.xyz_list[item_index, :]
                # print('try', item_xyz)
                start_pos = np.asarray([start_x[m], start_y[n]])
                index_block = best_all_config[m][n]
                item_sequence = item_sequence_list[index_block]
                rotate_flag = best_rotate_flag[m][n]

                ori, pos = self.reorder_item(best_config, start_pos, index_block, item_index, item_xyz, rotate_flag,
                                        item_odd_list, item_sequence)
                if rotate_flag == True:
                    temp = self.xyz_list[item_index, 0]
                    self.xyz_list[item_index, 0] = self.xyz_list[item_index, 1]
                    self.xyz_list[item_index, 1] = temp
                # print('tryori', ori)
                # print('trypos', pos)
                item_pos[item_index] = pos
                item_ori[item_index] = ori

        return item_pos, item_ori  # pos_list, ori_list

class arrangement():

    def __init__(self, arrange_policy,
                 item_even=True, block_even=True, upper_left_max=False, forced_rotate_box=False):

        self.arrange_policy = arrange_policy

    def calculate_items(self, item_num, item_xyz):

        min_xy = np.ones(2) * 100
        best_item_config = []
        best_item_sequence = []
        item_odd_flag = False
        all_item_x = 100
        all_item_y = 100
        best_iteration = 0
        xy_array = []
        iteration_array = []

        for iter in range(self.arrange_policy['iteration_time']):

            fac = []  # 定义一个列表存放因子
            for i in range(1, item_num + 1):
                if item_num % i == 0:
                    fac.append(i)
                    continue
            # fac = fac[::-1]

            if self.arrange_policy['object_even'] == True:
                if item_num % 2 != 0 and len(fac) == 2 and item_num >=5:  # its odd! we should generate the factor again!
                    item_num += 1
                    item_odd_flag = True
                    fac = []  # 定义一个列表存放因子
                    for i in range(1, item_num + 1):
                        if item_num % i == 0:
                            fac.append(i)
                            continue

            item_sequence = np.random.choice(len(item_xyz), len(item_xyz), replace=False)
            if item_odd_flag == True:
                item_sequence = np.append(item_sequence, item_sequence[-1])

            for j in range(len(fac)):
                # if item_num == 3:
                #     item_num_row = 1
                #     item_num_column = 3
                # else:
                item_num_row = int(fac[j])
                item_num_column = int(item_num / item_num_row)
                item_sequence = item_sequence.reshape(item_num_row, item_num_column)
                item_min_x = 0
                item_min_y = 0

                for r in range(item_num_row):
                    new_row = item_xyz[item_sequence[r, :]]
                    item_min_x = item_min_x + np.max(new_row, axis=0)[0]



                for c in range(item_num_column):
                    new_column = item_xyz[item_sequence[:, c]]
                    item_min_y = item_min_y + np.max(new_column, axis=0)[1]

                item_min_x = item_min_x + (item_num_row - 1) * self.arrange_policy['gap_item']
                item_min_y = item_min_y + (item_num_column - 1) * self.arrange_policy['gap_item']

                if item_min_x + item_min_y < all_item_x + all_item_y:
                    best_item_config = [item_num_row, item_num_column]
                    best_item_sequence = item_sequence
                    all_item_x = item_min_x
                    all_item_y = item_min_y
                    best_iteration = iter
                    min_xy = np.array([all_item_x, all_item_y])
                    xy_array.append(np.sum(min_xy))
                    iteration_array.append(iter)
                    # print(f'interation {self.best_iteration}, min_xy {np.sum(self.min_xy)}')
        # print(f'In iteration {best_iteration}, the min xy is {min_xy}, total is {np.sum(min_xy)}')
        # xy_array = 1 / np.asarray(xy_array)
        # iteration_array = np.asarray(iteration_array)
        # selected_index = np.where(xy_array > (1 / np.sum(min_xy) * 0.9))[0]
        # selected_iter = iteration_array[selected_index]
        # print(f'In iteration {selected_iter[0]}, the xy has met the condition, is {1 / xy_array[selected_index[0]]}')
        return min_xy, best_item_config, item_odd_flag, best_item_sequence

    def calculate_block(self, data, all_index):  # first: calculate, second: reorder!

        self.lwh_list = data[:, :3]
        self.all_index = copy.deepcopy(all_index)

        min_result = []
        best_config = []
        item_odd_list = []

        ################## zzz add sequence ###################
        item_sequence_list = []
        ################## zzz add sequence ###################

        for i in range(len(self.all_index)):
            item_index = self.all_index[i]
            item_xyz = self.lwh_list[item_index, :]
            item_num = len(item_index)
            xy, config, odd, item_sequence = self.calculate_items(item_num, item_xyz)
            # print(f'this is min xy {xy}')
            min_result.append(list(xy))
            # print(f'this is the best item config\n {config}')
            best_config.append(list(config))
            item_odd_list.append(odd)
            item_sequence_list.append(item_sequence)
        min_result = np.asarray(min_result).reshape(-1, 2)
        best_config = np.asarray(best_config).reshape(-1, 2)
        item_odd_list = np.asarray(item_odd_list)
        # print('this is item sequence list', item_sequence_list)
        # item_sequence_list = np.asarray(item_sequence_list, dtype=object)

        # print(best_config)

        if self.arrange_policy['upper_left_max'] == True:
            # reorder the block based on the min_xy 哪个block面积大哪个在前
            s_block_sequence = np.argsort(min_result[:, 0] * min_result[:, 1])[::-1]
            new_all_index = []
            for i in s_block_sequence:
                new_all_index.append(self.all_index[i])
            self.all_index = new_all_index.copy()
            min_result = min_result[s_block_sequence]
            best_config = best_config[s_block_sequence]
            item_odd_list = item_odd_list[s_block_sequence]
            item_sequence_list = [item_sequence_list[i] for i in s_block_sequence]
            # item_sequence_list = item_sequence_list[s_block_sequence]
            # reorder the block based on the min_xy 哪个block面积大哪个在前

        # 安排总的摆放
        iteration = 100
        all_num = best_config.shape[0]
        all_x = 100
        all_y = 100
        odd_flag = False
        best_min_xy = np.inf

        fac = []  # 定义一个列表存放因子
        for i in range(1, all_num + 1):
            if all_num % i == 0:
                fac.append(i)
                continue
        # fac = fac[::-1]

        if self.arrange_policy['block_even'] == True:
            if all_num % 2 != 0 and len(fac) == 2:  # its odd! we should generate the factor again!
                all_num += 1
                odd_flag = True
                fac = []  # 定义一个列表存放因子
                for i in range(1, all_num + 1):
                    if all_num % i == 0:
                        fac.append(i)
                        continue

        for i in range(self.arrange_policy['iteration_time']):

            if self.arrange_policy['upper_left_max'] == True:
                sequence = np.concatenate((np.array([0]), np.random.choice(best_config.shape[0] - 1, size=len(self.all_index) - 1, replace=False) + 1))
            else:
                sequence = np.random.choice(best_config.shape[0], size=len(self.all_index), replace=False)
            # sequence = np.arange(len(self.all_index))

            if odd_flag == True:
                sequence = np.append(sequence, sequence[-1])
            else:
                pass
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

                # the zero or 90 should permanently be 0
                for r in range(num_row):
                    for c in range(num_column):
                        new_row = min_xy[sequence[r][c]]
                        if self.arrange_policy['forced_rotate_box'] == True:
                            if new_row[0] > new_row[1]:
                                zero_or_90 = 90
                        else:
                            zero_or_90 = np.random.choice(np.array([0, 90]))
                        if zero_or_90 == 90:
                            rotate_flag[r][c] = True
                            temp = new_row[0]
                            new_row[0] = new_row[1]
                            new_row[1] = temp

                    # insert 'whether to rotate' here
                for r in range(num_row):
                    new_row = min_xy[sequence[r, :]]
                    min_x = min_x + np.max(new_row, axis=0)[0]

                for c in range(num_column):
                    new_column = min_xy[sequence[:, c]]
                    min_y = min_y + np.max(new_column, axis=0)[1]

                if min_x + min_y < all_x + all_y:
                    best_all_config = sequence
                    all_x = min_x
                    all_y = min_y
                    best_rotate_flag = rotate_flag
                    best_min_xy = np.copy(min_xy)

        return self.reorder_block(best_config, best_all_config, best_rotate_flag, best_min_xy, odd_flag, item_odd_list, item_sequence_list)

    def reorder_item(self, best_config, start_pos, index_block, item_index, item_xyz, rotate_flag, item_odd_list, item_sequence):

        # initiate the pos and ori
        # we don't analysis these imported oris
        # we directly define the ori is 0 or 90 degree, depending on the algorithm.
        item_row = item_sequence.shape[0]
        item_column = item_sequence.shape[1]
        item_odd_flag = item_odd_list[index_block]
        if item_odd_flag == True:
            item_pos = np.zeros([len(item_index) + 1, 3])
            item_ori = np.zeros([len(item_index) + 1, 3])
            item_xyz = np.append(item_xyz, item_xyz[-1]).reshape(-1, 3)
            # index_temp = np.arange(item_pos.shape[0] - 1)
            # index_temp = np.append(index_temp, index_temp[-1]).reshape(item_row, item_column)
        else:
            item_pos = np.zeros([len(item_index), 3])
            item_ori = np.zeros([len(item_index), 3])
            # index_temp = np.arange(item_pos.shape[0]).reshape(item_row, item_column)

        # the initial position of the first items

        if rotate_flag == True:

            temp = np.copy(item_xyz[:, 0])
            item_xyz[:, 0] = item_xyz[:, 1]
            item_xyz[:, 1] = temp
            item_ori[:, 2] = 0
            # print(item_ori)
            temp = item_row
            item_row = item_column
            item_column = temp
            # index_temp = index_temp.transpose()
            item_sequence = item_sequence.transpose()
        else:
            item_ori[:, 2] = 0

        start_item_x = np.array([start_pos[0]])
        start_item_y = np.array([start_pos[1]])
        previous_start_item_x = start_item_x
        previous_start_item_y = start_item_y

        for m in range(item_row):
            new_row = item_xyz[item_sequence[m, :]]
            start_item_x = np.append(start_item_x,
                                     (previous_start_item_x + np.max(new_row, axis=0)[0] + self.arrange_policy['gap_item']))
            previous_start_item_x = (previous_start_item_x + np.max(new_row, axis=0)[0] + self.arrange_policy['gap_item'])
        start_item_x = np.delete(start_item_x, -1)

        for n in range(item_column):
            new_column = item_xyz[item_sequence[:, n]]
            start_item_y = np.append(start_item_y,
                                     (previous_start_item_y + np.max(new_column, axis=0)[1] + self.arrange_policy['gap_item']))
            previous_start_item_y = (previous_start_item_y + np.max(new_column, axis=0)[1] + self.arrange_policy['gap_item'])
        start_item_y = np.delete(start_item_y, -1)

        x_pos, y_pos = np.copy(start_pos)[0], np.copy(start_pos)[1]

        for j in range(item_row):
            for k in range(item_column):
                if item_odd_flag == True and j == item_row - 1 and k == item_column - 1:
                    break
                # ################### check whether to transform for each item in each block!################
                # if self.transform_flag[item_index[item_sequence[j][k]]] == 1:
                #     # print(f'the index {item_index[index_temp[j][k]]} should be rotated because of transformation')
                #     item_ori[item_sequence[j][k], 2] -= np.pi / 2
                # ################### check whether to transform for each item in each block!################
                x_pos = start_item_x[j] + (item_xyz[item_sequence[j][k]][0]) / 2
                y_pos = start_item_y[k] + (item_xyz[item_sequence[j][k]][1]) / 2
                item_pos[item_sequence[j][k]][0] = x_pos
                item_pos[item_sequence[j][k]][1] = y_pos
        if item_odd_flag == True:
            item_pos = np.delete(item_pos, -1, axis=0)
            item_ori = np.delete(item_ori, -1, axis=0)
        else:
            pass
        # print('this is the shape of item pos', item_pos.shape)
        return item_ori, item_pos

    def reorder_block(self, best_config, best_all_config, best_rotate_flag, min_xy, odd_flag, item_odd_list, item_sequence_list):

        # print(f'the best configuration of all items is\n {best_all_config}')
        # print(f'the best configuration of each kind of items is\n {best_config}')
        # print(f'the rotate of each block of items is\n {best_rotate_flag}')
        # print(f'this is the min_xy of each kind of items after rotation\n {min_xy}')

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
            start_x.append((previous_start_x + np.max(new_row, axis=0)[0] + self.arrange_policy['gap_block']))
            previous_start_x = (previous_start_x + np.max(new_row, axis=0)[0] + self.arrange_policy['gap_block'])
        start_x = np.delete(start_x, -1)
        # print(f'this is start_x {start_x}')

        for n in range(num_all_column):
            new_column = min_xy[best_all_config[:, n]]
            # print(new_column)
            # print(np.max(new_column, axis=0)[1])
            start_y.append((previous_start_y + np.max(new_column, axis=0)[1] + self.arrange_policy['gap_block']))
            previous_start_y = (previous_start_y + np.max(new_column, axis=0)[1] + self.arrange_policy['gap_block'])
        start_y = np.delete(start_y, -1)
        # print(f'this is start_y {start_y}')d

        # determine the start position per item
        item_pos = np.zeros([len(self.lwh_list), 3])
        item_ori = np.zeros([len(self.lwh_list), 3])
        # print(self.xyz_list[self.all_index[0]])
        # print(self.all_index)
        for m in range(num_all_row):
            for n in range(num_all_column):
                if odd_flag == True and m == num_all_row - 1 and n == num_all_column - 1:
                    break  # this is the redundancy block
                item_index = self.all_index[best_all_config[m][n]]  # determine the index of blocks

                # print('try', item_index)
                item_xyz = self.lwh_list[item_index, :]
                # print('try', item_xyz)
                start_pos = np.asarray([start_x[m], start_y[n]])
                index_block = best_all_config[m][n]
                item_sequence = item_sequence_list[index_block]
                rotate_flag = best_rotate_flag[m][n]

                ori, pos = self.reorder_item(best_config, start_pos, index_block, item_index, item_xyz, rotate_flag,
                                        item_odd_list, item_sequence)
                if rotate_flag == True:
                    temp = self.lwh_list[item_index, 0]
                    self.lwh_list[item_index, 0] = self.lwh_list[item_index, 1]
                    self.lwh_list[item_index, 1] = temp
                # print('tryori', ori)
                # print('trypos', pos)
                item_pos[item_index] = pos
                item_ori[item_index] = ori

        return item_pos, item_ori  # pos_list, ori_list

    def sort(self, data, data_name):

        object_lwh = data[:, :3]
        objects_class = data[:, 3]
        objects_color = data[:, 4]

        # generate the class, color, area, ratio range to classify objects
        if self.arrange_policy['area_classify_flag'] == False:
            self.arrange_policy['area_num'] = 1
        else:
            self.arrange_policy['area_num'] = 2
        if self.arrange_policy['ratio_classify_flag'] == False:
            self.arrange_policy['ratio_num'] = 1
        else:
            self.arrange_policy['ratio_num'] = 2
        class_index = np.unique(objects_class)
        color_index = np.unique(objects_color)

        # generate the area and ratio range to classify objects
        s = object_lwh[:, 0] * object_lwh[:, 1]
        s_min, s_max = np.min(s), np.max(s)
        s_range = np.linspace(s_max, s_min, int(self.arrange_policy['area_num'] + 1))

        item_lwh_temp = np.copy(object_lwh)
        convert_index = np.where(item_lwh_temp[:, 0] < item_lwh_temp[:, 1])[0]
        temp = item_lwh_temp[convert_index, 0]
        item_lwh_temp[convert_index, 0] = item_lwh_temp[convert_index, 1]
        item_lwh_temp[convert_index, 1] = temp

        lw_ratio = item_lwh_temp[:, 0] / item_lwh_temp[:, 1]
        ratio_min, ratio_max = np.min(lw_ratio), np.max(lw_ratio)
        ratio_range = np.linspace(ratio_max, ratio_min, int(self.arrange_policy['ratio_num'] + 1))

        # ! initiate the number of items
        all_index = []
        new_object_lwh = []
        new_object_class = []
        new_object_color = []
        new_object_name = []
        rest_index = np.arange(len(object_lwh))
        index = 0

        # for cls in range(len(class_index)):
        #     for clr in range(len(color_index)):
        #         for i in range(self.arrange_policy['area_num']):
        #             for j in range(self.arrange_policy['ratio_num']):
        #                 kind_index = []
        #                 for m in range(len(object_lwh)):
        #                     if m not in rest_index:
        #                         continue
        #                     else:
        #                         if (s_range[i] >= s[m] >= s_range[i + 1]) and (ratio_range[j] >= lw_ratio[m] >= ratio_range[j + 1]) \
        #                             and (objects_class[m] == class_index[cls]) and (objects_color[m] == color_index[clr]):
        #                             # print(f'boxes{m} matches in area{i}, ratio{j}!')
        #                             kind_index.append(index)
        #                             new_object_lwh.append(object_lwh[m])
        #                             new_object_class.append(class_index[cls])
        #                             new_object_color.append(color_index[clr])
        #                             new_object_name.append(data_name[m])
        #                             index += 1
        #                             rest_index = np.delete(rest_index, np.where(rest_index == m))
        #                 if len(kind_index) != 0:
        #                     all_index.append(kind_index)
        if self.arrange_policy['type_classify_flag'] == True:
            for cls in range(len(class_index)):
                kind_index = []
                for m in range(len(object_lwh)):
                    if m not in rest_index:
                        continue
                    else:
                        if (objects_class[m] == class_index[cls]):
                            # print(f'boxes{m} matches in area{i}, ratio{j}!')
                            kind_index.append(index)
                            new_object_lwh.append(object_lwh[m])
                            new_object_class.append(class_index[cls])
                            new_object_color.append(objects_color[m])
                            new_object_name.append(data_name[m])
                            index += 1
                            rest_index = np.delete(rest_index, np.where(rest_index == m))
                if len(kind_index) != 0:
                    all_index.append(kind_index)
        if self.arrange_policy['color_classify_flag'] == True:
            for clr in range(len(color_index)):
                kind_index = []
                for m in range(len(object_lwh)):
                    if m not in rest_index:
                        continue
                    else:
                        if (objects_color[m] == color_index[clr]):
                            # print(f'boxes{m} matches in area{i}, ratio{j}!')
                            kind_index.append(index)
                            new_object_lwh.append(object_lwh[m])
                            new_object_class.append(objects_class[m])
                            new_object_color.append(color_index[clr])
                            new_object_name.append(data_name[m])
                            index += 1
                            rest_index = np.delete(rest_index, np.where(rest_index == m))
                if len(kind_index) != 0:
                    all_index.append(kind_index)
        if self.arrange_policy['area_classify_flag'] == True or self.arrange_policy['ratio_classify_flag'] == True:
            for i in range(self.arrange_policy['area_num']):
                for j in range(self.arrange_policy['ratio_num']):
                    kind_index = []
                    for m in range(len(object_lwh)):
                        if m not in rest_index:
                            continue
                        else:
                            if (s_range[i] >= s[m] >= s_range[i + 1]) and (ratio_range[j] >= lw_ratio[m] >= ratio_range[j + 1]):
                                # print(f'boxes{m} matches in area{i}, ratio{j}!')
                                kind_index.append(index)
                                new_object_lwh.append(object_lwh[m])
                                new_object_class.append(objects_class[m])
                                new_object_color.append(objects_color[m])
                                new_object_name.append(data_name[m])
                                index += 1
                                rest_index = np.delete(rest_index, np.where(rest_index == m))
                    if len(kind_index) != 0:
                        all_index.append(kind_index)

        new_object_lwh = np.asarray(new_object_lwh).reshape(-1, 3)
        new_object_class = np.asarray(new_object_class).reshape(-1, 1)
        new_object_color = np.asarray(new_object_color).reshape(-1, 1)
        new_object_name = np.asarray(new_object_name)
        if len(rest_index) != 0:
            # we should implement the rest of boxes!
            print('we should implement the rest of boxes!')
            rest_xyz = object_lwh[rest_index]
            new_object_lwh = np.concatenate((new_object_lwh, rest_xyz), axis=0)
            all_index.append(list(np.arange(index, len(object_lwh))))

        new_data = np.concatenate((new_object_lwh, new_object_class, new_object_color), axis=1)

        return new_data, new_object_name, all_index

    def generate_arrangement(self, data, data_name) ->'n*4 numpy array, length, width, class, color':

        data_before, data_name_before, all_index = self.sort(data=data, data_name=data_name)
        pos_after, ori_after = self.calculate_block(data=data_before, all_index=all_index)
        # print('here')
        data_after = np.concatenate((pos_after, ori_after, data_before), axis=1)

        return data_after, data_name_before

class visualize_env():

    def __init__(self, para_dict, knolling_para=None, lstm_dict=None, arrange_dict=None):

        self.para_dict = para_dict
        self.knolling_para = knolling_para

        self.kImageSize = {'width': 480, 'height': 480}
        self.init_pos_range = para_dict['init_pos_range']
        self.init_ori_range = para_dict['init_ori_range']
        self.init_offset_range = para_dict['init_offset_range']
        self.urdf_path = para_dict['urdf_path']
        self.pybullet_path = pd.getDataPath()
        self.is_render = para_dict['is_render']
        self.save_img_flag = para_dict['save_img_flag']
        self.objects_num = para_dict['objects_num']

        self.x_low_obs = 0.03
        self.x_high_obs = 0.27
        self.y_low_obs = -0.14
        self.y_high_obs = 0.14
        self.z_low_obs = 0.0
        self.z_high_obs = 0.05
        x_grasp_accuracy = 0.2
        y_grasp_accuracy = 0.2
        z_grasp_accuracy = 0.2
        self.x_grasp_interval = (self.x_high_obs - self.x_low_obs) * x_grasp_accuracy
        self.y_grasp_interval = (self.y_high_obs - self.y_low_obs) * y_grasp_accuracy
        self.z_grasp_interval = (self.z_high_obs - self.z_low_obs) * z_grasp_accuracy
        self.table_boundary = 0.03
        self.table_center = np.array([(self.x_low_obs + self.x_high_obs) / 2, (self.y_low_obs + self.y_high_obs) / 2])

        self.gripper_interval = 0.01

        if self.is_render:
            p.connect(p.GUI, options="--width=1280 --height=720")
        else:
            p.connect(p.DIRECT)

        self.camera_parameters = {
            'width': 640.,
            'height': 480,
            'fov': 42,
            'near': 0.1,
            'far': 100.,
            'camera_up_vector':
                [1, 0, 0],  # I really do not know the parameter's effect.
            'light_direction': [
                0.5, 0, 1
            ],  # the direction is from the light source position to the origin of the world frame.
        }
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.150, 0, 0], #0.175
                                                               distance=0.4,
                                                               yaw=90,
                                                               pitch = -90,
                                                               roll=0,
                                                               upAxisIndex=2)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov=self.camera_parameters['fov'],
                                                              aspect=self.camera_parameters['width'] /
                                                                     self.camera_parameters['height'],
                                                              nearVal=self.camera_parameters['near'],
                                                              farVal=self.camera_parameters['far'])
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                     cameraYaw=45,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[0.1, 0, 0])
        p.setAdditionalSearchPath(pd.getDataPath())
        # p.setPhysicsEngineParameter(numSolverIterations=10)
        p.setTimeStep(1. / 120.)

        self.grid_size = 5
        self.x_range = (-0.001, 0.001)
        self.y_range = (-0.001, 0.001)
        x_center = 0
        y_center = 0
        x_offset_values = np.linspace(self.x_range[0], self.x_range[1], self.grid_size)
        y_offset_values = np.linspace(self.y_range[0], self.y_range[1], self.grid_size)
        xx, yy = np.meshgrid(x_offset_values, y_offset_values)
        sigma = 0.01
        kernel = np.exp(-((xx - x_center) ** 2 + (yy - y_center) ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
        self.kernel = kernel / np.sum(kernel)

        self.main_demo_epoch = 0

        self.create_entry_num = 0

    def create_scene(self):

        if np.random.uniform(0, 1) > 0.5:
            p.configureDebugVisualizer(lightPosition=[np.random.randint(1, 2), np.random.uniform(0, 1.5), 2],
                                       shadowMapResolution=8192, shadowMapIntensity=np.random.randint(0, 1) / 10)
        else:
            p.configureDebugVisualizer(lightPosition=[np.random.randint(1, 2), np.random.uniform(-1.5, 0), 2],
                                       shadowMapResolution=8192, shadowMapIntensity=np.random.randint(0, 1) / 10)
        self.baseid = p.loadURDF(self.urdf_path + "plane_zzz.urdf", useMaximalCoordinates=True)

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

        background = np.random.randint(1, 5)
        textureId = p.loadTexture(self.urdf_path + f"img_{background}.png")
        p.changeVisualShape(self.baseid, -1, textureUniqueId=textureId, specularColor=[0, 0, 0])

        p.setGravity(0, 0, -10)

    def get_data_virtual(self):

        length_range = np.round(np.random.uniform(self.para_dict['box_range'][0][0],
                                                  self.para_dict['box_range'][0][1],
                                                  size=(self.para_dict['objects_num'], 1)), decimals=3)
        width_range = np.round(np.random.uniform(self.para_dict['box_range'][1][0],
                                                 np.minimum(length_range, 0.036),
                                                 size=(self.para_dict['objects_num'], 1)), decimals=3)
        height_range = np.round(np.random.uniform(self.para_dict['box_range'][2][0],
                                                  self.para_dict['box_range'][2][1],
                                                  size=(self.para_dict['objects_num'], 1)), decimals=3)
        lwh_list = np.concatenate((length_range, width_range, height_range), axis=1)
        return lwh_list

    def create_objects(self, pos_data=None, ori_data=None, lwh_data=None):

        self.lwh_list = self.get_data_virtual()
        rdm_ori_roll  = np.random.uniform(self.init_ori_range[0][0], self.init_ori_range[0][1], size=(self.objects_num, 1))
        rdm_ori_pitch = np.random.uniform(self.init_ori_range[1][0], self.init_ori_range[1][1], size=(self.objects_num, 1))
        rdm_ori_yaw   = np.random.uniform(self.init_ori_range[2][0], self.init_ori_range[2][1], size=(self.objects_num, 1))
        rdm_ori = np.concatenate((rdm_ori_roll, rdm_ori_pitch, rdm_ori_yaw), axis=1)
        rdm_pos_x = np.random.uniform(self.init_pos_range[0][0], self.init_pos_range[0][1], size=(self.objects_num, 1))
        rdm_pos_y = np.random.uniform(self.init_pos_range[1][0], self.init_pos_range[1][1], size=(self.objects_num, 1))
        rdm_pos_z = np.random.uniform(self.init_pos_range[2][0], self.init_pos_range[2][1], size=(self.objects_num, 1))
        x_offset = np.random.uniform(self.init_offset_range[0][0], self.init_offset_range[0][1])
        y_offset = np.random.uniform(self.init_offset_range[1][0], self.init_offset_range[1][1])
        print('this is offset: %.04f, %.04f' % (x_offset, y_offset))
        rdm_pos = np.concatenate((rdm_pos_x + x_offset, rdm_pos_y + y_offset, rdm_pos_z), axis=1)

        self.objects_index = []
        for i in range(self.objects_num):
            obj_name = f'object_{i}'
            create_box(obj_name, rdm_pos[i], p.getQuaternionFromEuler(rdm_ori[i]), size=self.lwh_list[i])
            self.objects_index.append(int(i + 2))
            r = np.random.uniform(0, 0.9)
            g = np.random.uniform(0, 0.9)
            b = np.random.uniform(0, 0.9)
            p.changeVisualShape(self.objects_index[i], -1, rgbaColor=(r, g, b, 1))

        for _ in range(int(100)):
            p.stepSimulation()
            if self.is_render == True:
                pass
                # time.sleep(1/96)

        p.changeDynamics(self.baseid, -1, lateralFriction=self.para_dict['base_lateral_friction'],
                         contactDamping=self.para_dict['base_contact_damping'],
                         contactStiffness=self.para_dict['base_contact_stiffness'])

    def create_arm(self):
        self.arm_id = p.loadURDF(self.urdf_path + "robot_arm928/robot_arm.urdf",
                                 basePosition=[-0.08, 0, 0.02], useFixedBase=True,
                                 flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        p.changeDynamics(self.arm_id, 7, lateralFriction=self.para_dict['gripper_lateral_friction'],
                         contactDamping=self.para_dict['gripper_contact_damping'],
                         contactStiffness=self.para_dict['gripper_contact_stiffness'])
        p.changeDynamics(self.arm_id, 8, lateralFriction=self.para_dict['gripper_lateral_friction'],
                         contactDamping=self.para_dict['gripper_contact_damping'],
                         contactStiffness=self.para_dict['gripper_contact_stiffness'])

    def get_images(self):
        (width, length, image, image_depth, seg_mask) = p.getCameraImage(width=640,
                                                                         height=480,
                                                                         viewMatrix=self.view_matrix,
                                                                         projectionMatrix=self.projection_matrix,
                                                                         renderer=p.ER_BULLET_HARDWARE_OPENGL)
        far_range = self.camera_parameters['far']
        near_range = self.camera_parameters['near']
        depth_data = far_range * near_range / (far_range - (far_range - near_range) * image_depth)
        top_height = 0.4 - depth_data
        my_im = image[:, :, :3]
        temp = np.copy(my_im[:, :, 0])  # change rgb image to bgr for opencv to save
        my_im[:, :, 0] = my_im[:, :, 2]
        my_im[:, :, 2] = temp
        img = np.copy(my_im)

        cv2.namedWindow('zzz', 0)
        cv2.resizeWindow('zzz', 1280, 960)
        cv2.imshow('zzz', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def setup(self):

        p.resetSimulation()
        self.create_scene()
        self.create_arm()
        self.create_objects()
        self.get_images()

        while True:
            p.stepSimulation()

if __name__ == '__main__':

    arrange_policy = {'length_range': [0.036, 0.06], 'width_range': [0.016, 0.036], 'class_range': [0, 9], 'color_range': [0, 4],
                      'objects_num': 5, 'gap_item': 0.016, 'gap_block': 0.016, 'area_num': 1, 'ratio_num': 1,
                      'preference': 'zzz',
                      'object_even': True, 'block_even': True}

    length_data = np.random.uniform(low=arrange_policy['length_range'][0], high=arrange_policy['length_range'][1], size=(arrange_policy['objects_num'], 1))
    width_data = np.random.uniform(low=arrange_policy['width_range'][0], high=arrange_policy['width_range'][1], size=(arrange_policy['objects_num'], 1))
    height_data = np.ones((arrange_policy['objects_num'], 1)) * 0.016
    class_data = np.random.randint(low=arrange_policy['class_range'][0], high=arrange_policy['class_range'][1], size=(arrange_policy['objects_num'], 1))
    color_data = np.random.randint(low=arrange_policy['color_range'][0], high=arrange_policy['color_range'][1], size=(arrange_policy['objects_num'], 1))
    color_data = np.ones((arrange_policy['objects_num'], 1))
    data = np.concatenate((length_data, width_data, height_data, class_data, color_data), axis=1).round(decimals=3)

    arrange_policy = arrangement(arrange_policy)
    arrange_policy.generate_arrangement(data)
