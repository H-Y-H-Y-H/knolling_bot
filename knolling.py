import numpy as np
import pyrealsense2 as rs
import pybullet_data as pd
import math
from func import *
import socket
import cv2
# from cam_obs_learning import *
import torch
from urdfpy import URDF
from env import Env


class knolling_main:

    def __init__(self):

        self.kImageSize = {'width': 480, 'height': 480}
        self.urdf_path = './urdf/'
        self.pybullet_path = pd.getDataPath()
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

        # p.setAdditionalSearchPath(pd.getDataPath())

    def get_parameters(self, is_render=None, evaluations=None, real_operate=False, obs_order='1',
                       check_detection_loss=None, env_parameters=None, use_knolling_model=None):

        # self.lego_num = lego_num
        self.is_render = is_render
        self.real_operate = real_operate
        self.obs_order = obs_order
        self.check_detection_loss = check_detection_loss
        self.evaluations = evaluations
        self.env_parameters = env_parameters
        self.use_knolling_model = use_knolling_model

    def planning(self, order, conn, real_height, sim_height):

        def get_start_end():  # generating all trajectories of all items in normal condition
            arm_z = 0
            roll = 0
            pitch = 0
            if self.obs_order == 'sim_image_obj_evaluate':
                manipulator_before, new_xyz_list, error = self.knolling_env.get_obs(self.obs_order)
                return error
            else:
                manipulator_before, new_xyz_list = self.knolling_env.get_obs(self.obs_order)

            # sequence pos_before, ori_before, pos_after, ori_after
            start_end = np.concatenate((manipulator_before, self.manipulator_after), axis=1)
            print('get start and end')

            return start_end, new_xyz_list

        def move(cur_pos, cur_ori, tar_pos, tar_ori):

            # add the offset manually
            if self.real_operate == True:
                # # automatically add z and x bias
                d = np.array([0, 0.3])
                d_y = np.array((0, 0.17, 0.21, 0.30))
                d_y = d
                z_bias = np.array([-0.003, 0.004])
                x_bias = np.array([-0.002, 0.000])# yolo error is +2mm along x axis!
                y_bias = np.array([0, -0.004, -0.001, 0.004])
                y_bias = np.array([0.001, 0.005])
                # z_parameters = np.polyfit(d, z_bias, 3)
                z_parameters = np.polyfit(d, z_bias, 1)
                x_parameters = np.polyfit(d, x_bias, 1)
                y_parameters = np.polyfit(d_y, y_bias, 1)
                new_z_formula = np.poly1d(z_parameters)
                new_x_formula = np.poly1d(x_parameters)
                new_y_formula = np.poly1d(y_parameters)

                distance = tar_pos[0]
                distance_y = tar_pos[0]
                tar_pos[2] = tar_pos[2] + new_z_formula(distance)
                print('this is z add', new_z_formula(distance))
                tar_pos[0] = tar_pos[0] + new_x_formula(distance)
                print('this is x add', new_x_formula(distance))
                if tar_pos[1] > 0:
                    tar_pos[1] += new_y_formula(distance_y) * np.clip((6 * (tar_pos[1] + 0.01)), 0, 1) + 0.0025 # 0.003 is manual!
                else:
                    tar_pos[1] -= new_y_formula(distance_y) * np.clip((6 * (tar_pos[1] - 0.01)), 0, 1) - 0.0025 # 0.003 is manual!
                print('this is tar pos after manual', tar_pos)

            if tar_ori[2] > 3.1416 / 2:
                tar_ori[2] = tar_ori[2] - np.pi
                print('tar ori is too large')
            elif tar_ori[2] < -3.1416 / 2:
                tar_ori[2] = tar_ori[2] + np.pi
                print('tar ori is too small')
            # print('this is tar ori', tar_ori)

            #################### use feedback control ###################
            if abs(cur_pos[0] - tar_pos[0]) < 0.001 and abs(cur_pos[1] - tar_pos[1]) < 0.001:
                # vertical, choose a small slice
                move_slice = 0.004
            else:
                # horizontal, choose a large slice
                if self.real_operate == True:
                    move_slice = 0.008
                else:
                    move_slice = 0.004

            # ###### zzz set time sleep ######
            # if cur_pos[2] - tar_pos[2] > 0.02:
            #     print(cur_pos)
            #     print(tar_pos)
            #     print('this is time sleep')
            #     time.sleep(1)

            if self.real_operate == True:
                tar_pos = tar_pos + np.array([0, 0, real_height])
                target_pos = np.copy(tar_pos)
                target_ori = np.copy(tar_ori)
                # target_pos[2] = Cartesian_offset_nn(np.array([tar_pos])).reshape(-1, )[2] # remove nn offset temporary

                if np.abs(target_pos[2] - cur_pos[2]) > 0.01 \
                        and np.abs(target_pos[0] - cur_pos[0]) < 0.01 \
                        and np.abs(target_pos[1] - cur_pos[1]) < 0.01:
                    print('we dont need feedback control')
                    mark_ratio = 0.8
                    seg_time = 0
                else:
                    mark_ratio = 0.99
                    seg_time = 0

                while True:
                    plot_cmd = []
                    # plot_real = []
                    sim_xyz = []
                    sim_ori = []
                    real_xyz = []

                    # divide the whole trajectory into several segment
                    seg_time += 1
                    seg_pos = mark_ratio * (target_pos - cur_pos) + cur_pos
                    seg_ori = mark_ratio * (target_ori - cur_ori) + cur_ori
                    distance = np.linalg.norm(seg_pos - cur_pos)
                    num_step = np.ceil(distance / move_slice)
                    step_pos = (seg_pos - cur_pos) / num_step
                    step_ori = (seg_ori - cur_ori) / num_step

                    while True:
                        tar_pos = cur_pos + step_pos
                        tar_ori = cur_ori + step_ori
                        sim_xyz.append(tar_pos)
                        sim_ori.append(tar_ori)
                        # print(tar_ori)

                        ik_angles_sim = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=tar_pos,
                                                                     maxNumIterations=200,
                                                                     targetOrientation=p.getQuaternionFromEuler(
                                                                         tar_ori))

                        for motor_index in range(self.num_motor):
                            p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                                    targetPosition=ik_angles_sim[motor_index], maxVelocity=25)
                        for i in range(30):
                            p.stepSimulation()

                        angle_sim = np.asarray(real_cmd2tarpos(rad2cmd(ik_angles_sim[0:5])), dtype=np.float32)
                        plot_cmd.append(angle_sim)

                        break_flag = abs(seg_pos[0] - tar_pos[0]) < 0.001 and abs(
                            seg_pos[1] - tar_pos[1]) < 0.001 and abs(seg_pos[2] - tar_pos[2]) < 0.001 and \
                                     abs(seg_ori[0] - tar_ori[0]) < 0.001 and abs(
                            seg_ori[1] - tar_ori[1]) < 0.001 and abs(seg_ori[2] - tar_ori[2]) < 0.001
                        if break_flag:
                            break

                        # update cur_pos and cur_ori in several step of each segment
                        cur_pos = tar_pos
                        cur_ori = tar_ori

                    sim_xyz = np.asarray(sim_xyz)

                    plot_step = np.arange(num_step)
                    plot_cmd = np.asarray(plot_cmd)
                    # print('this is the shape of cmd', plot_cmd.shape)
                    # print('this is the shape of xyz', sim_xyz.shape)
                    # print('this is the motor pos sent', plot_cmd[-1])
                    conn.sendall(plot_cmd.tobytes())
                    # print('waiting the manipulator')
                    angles_real = conn.recv(8192)
                    # print('received')
                    angles_real = np.frombuffer(angles_real, dtype=np.float32)
                    angles_real = angles_real.reshape(-1, 6)

                    if seg_time > 0:
                        seg_flag = False
                        print('segment fail, try to tune!')
                        ik_angles_sim = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=target_pos,
                                                                     maxNumIterations=200,
                                                                     targetOrientation=p.getQuaternionFromEuler(
                                                                         target_ori))

                        for motor_index in range(self.num_motor):
                            p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                                    targetPosition=ik_angles_sim[motor_index], maxVelocity=7.5)
                        for i in range(30):
                            p.stepSimulation()

                        angle_sim = np.asarray(real_cmd2tarpos(rad2cmd(ik_angles_sim[0:5])), dtype=np.float32)
                        final_cmd = np.append(angle_sim, 0).reshape(1, -1)
                        final_cmd = np.asarray(final_cmd, dtype=np.float32)
                        conn.sendall(final_cmd.tobytes())

                        # get the pos after tune!
                        final_angles_real = conn.recv(4096)
                        # print('received')
                        final_angles_real = np.frombuffer(final_angles_real, dtype=np.float32).reshape(-1, 6)
                        print('this is final after moving', final_angles_real)

                        ik_angles_real = np.asarray(cmd2rad(real_tarpos2cmd(final_angles_real)), dtype=np.float32)
                        for motor_index in range(self.num_motor):
                            p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                                    targetPosition=ik_angles_real[motor_index], maxVelocity=25)
                        for i in range(30):
                            p.stepSimulation()
                        real_xyz = np.append(real_xyz, np.asarray(p.getLinkState(self.arm_id, 9)[0])).reshape(-1, 3)
                        cur_pos = real_xyz[-1]
                        # print(real_xyz)
                        break
                    else:
                        print('this is the shape of angles real', angles_real.shape)
                        for i in range(len(angles_real)):
                            ik_angles_real = np.asarray(cmd2rad(real_tarpos2cmd(angles_real[i])), dtype=np.float32)
                            for motor_index in range(self.num_motor):
                                p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                                        targetPosition=ik_angles_real[motor_index], maxVelocity=25)
                            for i in range(30):
                                p.stepSimulation()
                            real_xyz = np.append(real_xyz, np.asarray(p.getLinkState(self.arm_id, 9)[0])).reshape(-1, 3)
                        cur_pos = real_xyz[-1]
                        break

            else:
                tar_pos = tar_pos + np.array([0, 0, sim_height])
                target_pos = np.copy(tar_pos)
                target_ori = np.copy(tar_ori)

                distance = np.linalg.norm(tar_pos - cur_pos)
                num_step = np.ceil(distance / move_slice)
                step_pos = (target_pos - cur_pos) / num_step
                step_ori = (target_ori - cur_ori) / num_step

                print('this is sim tar pos', tar_pos)
                while True:
                    tar_pos = cur_pos + step_pos
                    tar_ori = cur_ori + step_ori
                    ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=tar_pos,
                                                              maxNumIterations=200,
                                                              targetOrientation=p.getQuaternionFromEuler(tar_ori))
                    for motor_index in range(self.num_motor):
                        p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                                targetPosition=ik_angles0[motor_index], maxVelocity=25)
                    for i in range(30):
                        p.stepSimulation()
                        # time.sleep(1 / 960)
                    if abs(target_pos[0] - tar_pos[0]) < 0.001 and abs(target_pos[1] - tar_pos[1]) < 0.001 and abs(
                            target_pos[2] - tar_pos[2]) < 0.001 and \
                            abs(target_ori[0] - tar_ori[0]) < 0.001 and abs(target_ori[1] - tar_ori[1]) < 0.001 and abs(
                        target_ori[2] - tar_ori[2]) < 0.001:
                        break
                    cur_pos = tar_pos
                    cur_ori = tar_ori

            return cur_pos

        def gripper(gap, obj_width):
            obj_width += 0.006
            if self.real_operate == True:
                obj_width_range = np.array([0.021, 0.026, 0.032, 0.039, 0.045, 0.052, 0.057])
                motor_pos_range = np.array([2050, 2150, 2250, 2350, 2450, 2550, 2650])
                formula_parameters = np.polyfit(obj_width_range, motor_pos_range, 3)
                motor_pos = np.poly1d(formula_parameters)
            else:
                close_open_gap = 0.053
                obj_width_range = np.array([0.022, 0.057])
                motor_pos_range = np.array([0.022, 0.010]) # 0.0273
                formula_parameters = np.polyfit(obj_width_range, motor_pos_range, 1)
                motor_pos = np.poly1d(formula_parameters)

            if self.real_operate == True:
                if gap > 0.0265:  # close
                    pos_real = np.asarray([[gap, 1600]], dtype=np.float32)
                elif gap <= 0.0265:  # open
                    pos_real = np.asarray([[gap, motor_pos(obj_width)]], dtype=np.float32)
                print('gripper', pos_real)
                conn.sendall(pos_real.tobytes())
                # print(f'this is the cmd pos {pos_real}')
                p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL, targetPosition=gap, force=10)
                p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL, targetPosition=gap, force=10)

                real_pos = conn.recv(4096)
                # test_real_pos = np.frombuffer(real_pos, dtype=np.float32)
                real_pos = np.frombuffer(real_pos, dtype=np.float32)
                # print('this is test float from buffer', test_real_pos)

            else:
                if gap > 0.0265: # close
                    p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL, targetPosition=motor_pos(obj_width) + close_open_gap, force=10)
                    p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL, targetPosition=motor_pos(obj_width) + close_open_gap, force=10)
                else: # open
                    p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL, targetPosition=motor_pos(obj_width), force=10)
                    p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL, targetPosition=motor_pos(obj_width), force=10)
            for i in range(30):
                p.stepSimulation()
                time.sleep(1 / 120)

        def clean_desk():

            if self.real_operate == False:
                gripper_width = 0.024
                gripper_height = 0.034
            else:
                gripper_width = 0.018
                gripper_height = 0.04
            restrict_gripper_diagonal = np.sqrt(gripper_width ** 2 + gripper_height ** 2)
            barricade_pos = []
            barricade_index = []
            manipulator_before, new_xyz_list = self.knolling_env.get_obs(self.obs_order)
            print('this is test obs xyz', new_xyz_list)
            for i in range(len(manipulator_before)):
                for j in range(len(self.manipulator_after)):
                    restrict_item_i = np.sqrt((new_xyz_list[i][0]) ** 2 + (new_xyz_list[i][1]) ** 2)
                    restrict_item_j = np.sqrt((new_xyz_list[j][0]) ** 2 + (new_xyz_list[j][1]) ** 2)
                    if np.linalg.norm(self.manipulator_after[j][:3] - manipulator_before[i][:3]) < restrict_item_i / 2 + restrict_item_j / 2:
                        if i not in barricade_index:
                            print('We need to sweep the desktop to provide an enough space')
                            barricade_pos.append(manipulator_before[i][:3])
                            barricade_index.append(i)
            barricade_pos = np.asarray(barricade_pos)

            x_high = np.max(self.manipulator_after[:, 0])
            x_low = np.min(self.manipulator_after[:, 0])
            y_high = np.max(self.manipulator_after[:, 1])
            y_low = np.min(self.manipulator_after[:, 1])
            p.addUserDebugLine(lineFromXYZ=[x_low, y_low, 0], lineToXYZ=[x_high, y_low, 0])
            p.addUserDebugLine(lineFromXYZ=[x_low, y_low, 0], lineToXYZ=[x_low, y_high, 0])
            p.addUserDebugLine(lineFromXYZ=[x_high, y_high, 0], lineToXYZ=[x_high, y_low, 0])
            p.addUserDebugLine(lineFromXYZ=[x_high, y_high, 0], lineToXYZ=[x_low, y_high, 0])

            while len(barricade_index) > 0:

                # pos
                offset_low = np.array([0, 0, 0.005])
                offset_high = np.array([0, 0, 0.035])
                # ori
                rest_ori = np.array([0, 1.57, 0])
                # axis and direction
                if y_high - y_low > x_high - x_low:
                    offset_rectangle = np.array([0, 0, math.pi / 2])
                    axis = 'x_axis'
                    if (x_high + x_low) / 2 > (self.x_high_obs + self.x_low_obs) / 2:
                        direction = 'negative'
                        offset_horizontal = np.array([np.max(self.xyz_list) - 0.001, 0, 0])
                    else:
                        direction = 'positive'
                        offset_horizontal = np.array([-(np.max(self.xyz_list) - 0.001), 0, 0])
                else:
                    offset_rectangle = np.array([0, 0, 0])
                    axis = 'y_axis'
                    if (y_high + y_low) / 2 > (self.y_high_obs + self.y_low_obs) / 2:
                        direction = 'negative'
                        offset_horizontal = np.array([0, np.max(self.xyz_list) - 0.001, 0])
                    else:
                        direction = 'positive'
                        offset_horizontal = np.array([0, -(np.max(self.xyz_list) - 0.001), 0])

                trajectory_pos_list = []
                trajectory_ori_list = []
                print(barricade_index)
                for i in range(len(barricade_index)):
                    diag = np.sqrt((new_xyz_list[barricade_index[i]][0]) ** 2 + (new_xyz_list[barricade_index[i]][1]) ** 2)
                    if axis == 'x_axis':
                        if direction == 'positive':
                            print('x,p')
                            offset_horizontal = np.array([-(diag / 2 + gripper_height / 2), 0, 0])
                            terminate = np.array([x_high, barricade_pos[i][1], barricade_pos[i][2]])
                        elif direction == 'negative':
                            print('x,n')
                            offset_horizontal = np.array([diag / 2 + gripper_height / 2, 0, 0])
                            terminate = np.array([x_low, barricade_pos[i][1], barricade_pos[i][2]])
                    elif axis == 'y_axis':
                        if direction == 'positive':
                            print('y,p')
                            offset_horizontal = np.array([0, -(diag / 2 + gripper_height / 2), 0])
                            terminate = np.array([barricade_pos[i][0], y_high, barricade_pos[i][2]])
                        elif direction == 'negative':
                            print('y,n')
                            offset_horizontal = np.array([0, diag / 2 + gripper_height / 2, 0])
                            terminate = np.array([barricade_pos[i][0], y_low, barricade_pos[i][2]])

                    trajectory_pos_list.append([0.03159, 0])
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
                trajectory_pos_list.append([0, 0, 0.08])
                trajectory_ori_list.append([0, math.pi / 2, 0])

                last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
                last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))
                for j in range(len(trajectory_pos_list)):

                    if len(trajectory_pos_list[j]) == 3:
                        last_pos = move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j])
                        last_ori = np.copy(trajectory_ori_list[j])
                    elif len(trajectory_pos_list[j]) == 2:
                        gripper(trajectory_pos_list[j][0], trajectory_pos_list[j][1])

                break_flag = False
                barricade_pos = []
                barricade_index = []
                manipulator_before, new_xyz_list = self.knolling_env.get_obs(self.obs_order)
                for i in range(len(manipulator_before)):
                    for j in range(len(self.manipulator_after)):
                        # 这里会因为漏检的bug而报错！
                        restrict_item_i = np.sqrt((new_xyz_list[i][0]) ** 2 + (new_xyz_list[i][1]) ** 2)
                        restrict_item_j = np.sqrt((new_xyz_list[j][0]) ** 2 + (new_xyz_list[j][1]) ** 2)
                        if np.linalg.norm(self.manipulator_after[j][:3] - manipulator_before[i][:3]) < restrict_item_i / 2 + restrict_item_j / 2:
                            if i not in barricade_index:
                                print('We still need to sweep the desktop to provide an enough space')
                                barricade_pos.append(manipulator_before[i][:3])
                                barricade_index.append(i)
                                break_flag = True
                                break
                    if break_flag == True:
                        break
                barricade_pos = np.asarray(barricade_pos)
            else:
                print('nothing to sweep')
                pass
            print('Sweeping desktop end')

            return manipulator_before, new_xyz_list

        def clean_item(manipulator_before, new_xyz_list):

            if self.real_operate == False:
                gripper_width = 0.024
                gripper_height = 0.034
            else:
                gripper_width = 0.018
                gripper_height = 0.04

            restrict_gripper_diagonal = np.sqrt(gripper_width ** 2 + gripper_height ** 2)
            gripper_lego_gap = 0.006
            crowded_pos = []
            crowded_ori = []
            crowded_index = []

            # these two variables have been defined in clean_desk function, we don't need to define them twice!!!!!
            # manipulator_before, new_xyz_list = self.get_obs(self.obs_order, _)
            # print(manipulator_before)

            # define the cube which is crowded
            for i in range(len(manipulator_before)):
                for j in range(len(manipulator_before)):
                    restrict_item_i = np.sqrt((new_xyz_list[i][0]) ** 2 + (new_xyz_list[i][1]) ** 2)
                    restrict_item_j = np.sqrt((new_xyz_list[j][0]) ** 2 + (new_xyz_list[j][1]) ** 2)
                    if 0.0001 < np.linalg.norm(manipulator_before[j][:3] - manipulator_before[i][:3]) < restrict_item_i / 2 + restrict_item_j / 2 + 0.001:
                        if i not in crowded_index and j not in crowded_index:
                            print('We need to separate the items surrounding it to provide an enough space')
                            crowded_pos.append(manipulator_before[i][:3])
                            crowded_ori.append(manipulator_before[i][3:6])
                            crowded_pos.append(manipulator_before[j][:3])
                            crowded_ori.append(manipulator_before[j][3:6])
                            crowded_index.append(i)
                            crowded_index.append(j)
                        if i in crowded_index and j not in crowded_index:
                            print('We need to separate the items surrounding it to provide an enough space')
                            crowded_pos.append(manipulator_before[j][:3])
                            crowded_ori.append(manipulator_before[j][3:6])
                            crowded_index.append(j)
            crowded_pos = np.asarray(crowded_pos)

            while len(crowded_index) > 0:
                # pos
                offset_low = np.array([0, 0, 0.005])
                offset_high = np.array([0, 0, 0.035])
                # ori
                rest_ori = np.array([0, 1.57, 0])

                trajectory_pos_list = []
                trajectory_ori_list = []
                for i in range(len(crowded_index)):
                    break_flag = False
                    once_flag = False

                    length_lego = new_xyz_list[crowded_index[i]][0]
                    width_lego = new_xyz_list[crowded_index[i]][1]
                    theta = manipulator_before[crowded_index[i]][5]

                    matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                       [np.sin(theta), np.cos(theta)]])
                    target_point = np.array([[(length_lego + gripper_height + gripper_lego_gap) / 2, (width_lego + gripper_width + gripper_lego_gap) / 2],
                                            [-(length_lego + gripper_height + gripper_lego_gap) / 2, (width_lego + gripper_width + gripper_lego_gap) / 2],
                                            [-(length_lego + gripper_height + gripper_lego_gap) / 2, -(width_lego + gripper_width + gripper_lego_gap) / 2],
                                            [(length_lego + gripper_height + gripper_lego_gap) / 2, -(width_lego + gripper_width + gripper_lego_gap) / 2]])
                    target_point_rotate = (matrix.dot(target_point.T)).T
                    print('this is target point rotate\n', target_point_rotate)
                    sequence_point = np.concatenate((target_point_rotate, np.zeros((4, 1))), axis=1)

                    t = 0
                    for j in range(len(sequence_point)):
                        vertex_break_flag = False
                        for k in range(len(manipulator_before)):
                            # exclude itself
                            if np.linalg.norm(crowded_pos[i] - manipulator_before[k][:3]) < 0.001:
                                continue
                            restrict_item_k = np.sqrt((new_xyz_list[k][0]) ** 2 + (new_xyz_list[k][1]) ** 2)
                            if 0.001 < np.linalg.norm(sequence_point[0] + crowded_pos[i] - manipulator_before[k][:3]) < restrict_item_k/2 + restrict_gripper_diagonal/2 + 0.001:
                                print(np.linalg.norm(sequence_point[0] + crowded_pos[i] - manipulator_before[k][:3]))
                                p.addUserDebugPoints([sequence_point[0] + crowded_pos[i]], [[0.1, 0, 0]], pointSize=5)
                                p.addUserDebugPoints([manipulator_before[k][:3]], [[0, 1, 0]], pointSize=5)
                                print("this vertex doesn't work")
                                vertex_break_flag = True
                                break
                        if vertex_break_flag == False:
                            print("this vertex is ok")
                            print(break_flag)
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
                        if i == len(crowded_index) - 1:
                            print('cannot find any proper vertices to insert, we should unpack the heap!!!')
                            x_high = np.max(self.manipulator_after[:, 0])
                            x_low = np.min(self.manipulator_after[:, 0])
                            y_high = np.max(self.manipulator_after[:, 1])
                            y_low = np.min(self.manipulator_after[:, 1])
                            crowded_x_high = np.max(crowded_pos[:, 0])
                            crowded_x_low = np.min(crowded_pos[:, 0])
                            crowded_y_high = np.max(crowded_pos[:, 1])
                            crowded_y_low = np.min(crowded_pos[:, 1])

                            trajectory_pos_list.append([0.03159, 0])
                            trajectory_pos_list.append([(x_high + x_low) / 2, (y_high + y_low) / 2, offset_high[2]])
                            trajectory_pos_list.append([(x_high + x_low) / 2, (y_high + y_low) / 2, offset_low[2]])
                            trajectory_pos_list.append([(crowded_x_high + crowded_x_low) / 2, (crowded_y_high + crowded_y_low) / 2, offset_low[2]])
                            trajectory_pos_list.append([(crowded_x_high + crowded_x_low) / 2, (crowded_y_high + crowded_y_low) / 2, offset_high[2]])
                            trajectory_pos_list.append([0, 0, 0.08])

                            trajectory_ori_list.append(rest_ori)
                            trajectory_ori_list.append(rest_ori)
                            trajectory_ori_list.append(rest_ori)
                            trajectory_ori_list.append(rest_ori)
                            trajectory_ori_list.append(rest_ori)
                            trajectory_ori_list.append(rest_ori)
                        else:
                            pass
                    else:
                        trajectory_pos_list.append([0.03159, 0])
                        print('this is crowded pos', crowded_pos[i])
                        print('this is sequence point', sequence_point)
                        trajectory_pos_list.append(crowded_pos[i] + offset_high + sequence_point[0])
                        trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[0])
                        trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[1])
                        trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[2])
                        trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[3])
                        trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[0])
                        trajectory_pos_list.append(crowded_pos[i] + offset_high + sequence_point[0])
                        # reset the manipulator to read the image
                        trajectory_pos_list.append([0, 0, 0.06])

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
                trajectory_pos_list = np.asarray(trajectory_pos_list)
                trajectory_ori_list = np.asarray(trajectory_ori_list)

                ######################### add the debug lines for visualization ####################
                line_id = []
                four_points = trajectory_pos_list[2:6]
                line_id.append(p.addUserDebugLine(lineFromXYZ=four_points[0], lineToXYZ=four_points[1]))
                line_id.append(p.addUserDebugLine(lineFromXYZ=four_points[1], lineToXYZ=four_points[2]))
                line_id.append(p.addUserDebugLine(lineFromXYZ=four_points[2], lineToXYZ=four_points[3]))
                line_id.append(p.addUserDebugLine(lineFromXYZ=four_points[3], lineToXYZ=four_points[0]))
                ######################### add the debug line for visualization ####################

                for j in range(len(trajectory_pos_list)):
                    if len(trajectory_pos_list[j]) == 3:
                        last_pos = move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j])
                        last_ori = np.copy(trajectory_ori_list[j])
                    elif len(trajectory_pos_list[j]) == 2:
                        gripper(trajectory_pos_list[j][0], trajectory_pos_list[j][1])

                ######################### remove the debug lines after moving ######################
                for i in line_id:
                    p.removeUserDebugItem(i)
                ######################### remove the debug lines after moving ######################

                # check the environment again and update the pos and ori of cubes
                crowded_pos = []
                crowded_ori = []
                crowded_index = []
                manipulator_before, new_xyz_list = self.knolling_env.get_obs(self.obs_order)
                for i in range(len(manipulator_before)):
                    for j in range(len(manipulator_before)):
                        restrict_item_i = np.sqrt((new_xyz_list[i][0]) ** 2 + (new_xyz_list[i][1]) ** 2)
                        restrict_item_j = np.sqrt((new_xyz_list[j][0]) ** 2 + (new_xyz_list[j][1]) ** 2)
                        if 0.0001 < np.linalg.norm(manipulator_before[j][:3] - manipulator_before[i][:3]) < restrict_item_i / 2 + restrict_item_j / 2:
                            if i not in crowded_index and j not in crowded_index:
                                print('We need to separate the items surrounding it to provide an enough space')
                                crowded_pos.append(manipulator_before[i][:3])
                                crowded_ori.append(manipulator_before[i][3:6])
                                crowded_pos.append(manipulator_before[j][:3])
                                crowded_ori.append(manipulator_before[j][3:6])
                                crowded_index.append(i)
                                crowded_index.append(j)
                            if i in crowded_index and j not in crowded_index:
                                print('We need to separate the items surrounding it to provide an enough space')
                                crowded_pos.append(manipulator_before[j][:3])
                                crowded_ori.append(manipulator_before[j][3:6])
                                crowded_index.append(j)
                crowded_pos = np.asarray(crowded_pos)
            else:
                print('nothing around the item')
                pass
            print('separating end')

        def knolling():

            if self.obs_order == 'sim_image_obj_evaluate':
                env_loss = get_start_end()
                return env_loss
            else:
                start_end, new_xyz_list_knolling = get_start_end()


            rest_pos = np.array([0, 0, 0.05])
            rest_ori = np.array([0, 1.57, 0])
            offset_low = np.array([0, 0, 0.0])
            offset_low_place = np.array([0, 0, 0.002])
            offset_high = np.array([0, 0, 0.035])
            offset_highest = np.array([0, 0, 0.05])

            grasp_width = np.min(new_xyz_list_knolling[:, :2], axis=1)

            for i in range(len(self.manipulator_after)):

                trajectory_pos_list = [[0.02, grasp_width[i]], # open!
                                       offset_high + start_end[i][:3],
                                       offset_low + start_end[i][:3],
                                       [0.0273, grasp_width[i]], # close
                                       offset_high + start_end[i][:3],
                                       offset_high + start_end[i][6:9],
                                       offset_low_place + start_end[i][6:9],
                                       [0.02, grasp_width[i]],
                                       offset_high + start_end[i][6:9]]

                trajectory_ori_list = [rest_ori + start_end[i][3:6],
                                       rest_ori + start_end[i][3:6],
                                       rest_ori + start_end[i][3:6],
                                       [0.0273, grasp_width[i]],
                                       rest_ori + start_end[i][3:6],
                                       rest_ori + start_end[i][9:12],
                                       rest_ori + start_end[i][9:12],
                                       [0.02, grasp_width[i]],
                                       rest_ori + start_end[i][9:12]]
                if i == 0:
                    last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
                    last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))
                else:
                    pass

                for j in range(len(trajectory_pos_list)):

                    if len(trajectory_pos_list[j]) == 3:
                        print('ready to move', trajectory_pos_list[j])
                        # print('ready to move cur ori', last_ori)
                        last_pos = move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j])
                        last_ori = np.copy(trajectory_ori_list[j])
                        # print('this is last ori after moving', last_ori)

                    elif len(trajectory_pos_list[j]) == 2:
                        gripper(trajectory_pos_list[j][0], trajectory_pos_list[j][1])

            # back to the reset pos and ori
            last_pos = move(last_pos, last_ori, rest_pos, rest_ori)
            last_ori = np.copy(rest_ori)
            ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=rest_pos,
                                                      maxNumIterations=200,
                                                      targetOrientation=p.getQuaternionFromEuler(rest_ori))
            for motor_index in range(self.num_motor):
                p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                        targetPosition=ik_angles0[motor_index], maxVelocity=7)
            for i in range(30):
                p.stepSimulation()
                # self.images = self.get_image()
                # time.sleep(1 / 48)

        def check_accuracy_sim(): # need improvement


            manipulator_before, new_xyz_list = self.knolling_env.get_obs(self.obs_order) # the sequence along 2,3,4
            manipulator_knolling = manipulator_before[:, :2]
            xyz_knolling = new_xyz_list
            # don't change the order of xyz in sim!!!!!!!!!!!!!

            order_knolling = np.lexsort((manipulator_before[:, 1], manipulator_before[:, 0]))
            manipulator_knolling_test = np.copy(manipulator_knolling[order_knolling, :])
            for i in range(len(order_knolling) - 1):
                if np.abs(manipulator_knolling_test[i, 0] - manipulator_knolling_test[i + 1, 0]) < 0.003:
                    if manipulator_knolling_test[i, 1] < manipulator_knolling_test[i + 1, 1]:
                        order_knolling[i], order_knolling[i + 1] = order_knolling[i + 1], \
                                                                           order_knolling[i]
                        print('knolling change the order!')
                    else:
                        pass
            print('this is the ground truth order', order_knolling)
            # print('this is the ground truth before changing the order\n', manipulator_knolling)
            manipulator_knolling = manipulator_knolling[order_knolling, :]


            new_pos_before, new_ori_before = [], []
            for i in range(len(self.lego_idx)):
                new_pos_before.append(p.getBasePositionAndOrientation(self.lego_idx[i])[0][:2])
            new_pos_before = np.asarray(new_pos_before)
            manipulator_ground_truth = new_pos_before
            xyz_ground_truth = self.xyz_list
            # don't change the order of xyz in sim!!!!!!!!!!!!!

            order_ground_truth = np.lexsort((manipulator_ground_truth[:, 1], manipulator_ground_truth[:, 0]))
            manipulator_ground_truth_test = np.copy(manipulator_ground_truth[order_ground_truth, :])
            for i in range(len(order_ground_truth) - 1):
                if np.abs(manipulator_ground_truth_test[i, 0] - manipulator_ground_truth_test[i + 1, 0]) < 0.003:
                    if manipulator_ground_truth_test[i, 1] < manipulator_ground_truth_test[i + 1, 1]:
                        order_ground_truth[i], order_ground_truth[i + 1] = order_ground_truth[i + 1], \
                                                                   order_ground_truth[i]
                        print('truth change the order!')
                    else:
                        pass
            print('this is the ground truth order', order_knolling)
            # print('this is the ground truth before changing the order\n', manipulator_knolling)
            manipulator_ground_truth = manipulator_ground_truth[order_ground_truth, :]

            print('this is manipulator ground truth while checking \n', manipulator_ground_truth)
            print('this is manipulator after knolling while checking \n', manipulator_knolling)
            print('this is xyz ground truth while checking \n', xyz_ground_truth)
            print('this is xyz after knolling while checking \n', xyz_knolling)
            for i in range(len(manipulator_ground_truth)):
                if np.linalg.norm(manipulator_ground_truth[i] - manipulator_knolling[i]) < 0.005 and \
                    np.linalg.norm(xyz_ground_truth[i] - xyz_knolling[i]) < 0.005:
                    print('find it!')
                else:
                    print('error!')

        def check_accuracy_real():
            manipulator_before, new_xyz_list = self.knolling_env.get_obs(self.obs_order)
            manipulator_knolling = manipulator_before[:, :2]
            xyz_knolling = new_xyz_list
            # don't change the order of xyz in sim!!!!!!!!!!!!!

            order_knolling = np.lexsort((manipulator_before[:, 1], manipulator_before[:, 0]))
            manipulator_knolling_test = np.copy(manipulator_knolling[order_knolling, :])
            for i in range(len(order_knolling) - 1):
                if np.abs(manipulator_knolling_test[i, 0] - manipulator_knolling_test[i + 1, 0]) < 0.003:
                    if manipulator_knolling_test[i, 1] < manipulator_knolling_test[i + 1, 1]:
                        order_knolling[i], order_knolling[i + 1] = order_knolling[i + 1], \
                                                                   order_knolling[i]
                        print('knolling change the order!')
                    else:
                        pass
            print('this is the ground truth order', order_knolling)
            # print('this is the ground truth before changing the order\n', manipulator_knolling)
            manipulator_knolling = manipulator_knolling[order_knolling, :]

            new_pos_before, new_ori_before = [], []
            for i in range(len(self.lego_idx)):
                new_pos_before.append(p.getBasePositionAndOrientation(self.lego_idx[i])[0][:2])
            new_pos_before = np.asarray(new_pos_before)
            manipulator_ground_truth = new_pos_before
            xyz_ground_truth = self.xyz_list
            # don't change the order of xyz in sim!!!!!!!!!!!!!

            order_ground_truth = np.lexsort((manipulator_ground_truth[:, 1], manipulator_ground_truth[:, 0]))
            manipulator_ground_truth_test = np.copy(manipulator_ground_truth[order_ground_truth, :])
            for i in range(len(order_ground_truth) - 1):
                if np.abs(manipulator_ground_truth_test[i, 0] - manipulator_ground_truth_test[i + 1, 0]) < 0.003:
                    if manipulator_ground_truth_test[i, 1] < manipulator_ground_truth_test[i + 1, 1]:
                        order_ground_truth[i], order_ground_truth[i + 1] = order_ground_truth[i + 1], \
                                                                           order_ground_truth[i]
                        print('truth change the order!')
                    else:
                        pass
            print('this is the ground truth order', order_knolling)
            # print('this is the ground truth before changing the order\n', manipulator_knolling)
            manipulator_ground_truth = manipulator_ground_truth[order_ground_truth, :]

            print('this is manipulator ground truth while checking \n', manipulator_ground_truth)
            print('this is manipulator after knolling while checking \n', manipulator_knolling)
            print('this is xyz ground truth while checking \n', xyz_ground_truth)
            print('this is xyz after knolling while checking \n', xyz_knolling)
            for i in range(len(manipulator_ground_truth)):
                if np.linalg.norm(manipulator_ground_truth[i] - manipulator_knolling[i]) < 0.005 and \
                        np.linalg.norm(xyz_ground_truth[i] - xyz_knolling[i]) < 0.005:
                    print('find it!')
                else:
                    print('error!')

            print('this is all distance between messy and neat in real world')

        if order == 1:
            manipulator_before_desk, new_xyz_list_desk = clean_desk()
            clean_item(manipulator_before_desk, new_xyz_list_desk)
        elif order == 3:
            if self.obs_order == 'sim_image_obj_evaluate':
                error = knolling()
                return error
            else:
                knolling()
        elif order == 4:
            if self.real_operate == True:
                check_accuracy_real()
            else:
                check_accuracy_sim()
        elif order == 5:
            error = get_start_end()
            return error

    def step(self):

        self.knolling_env = Env(is_render=self.is_render)
        self.knolling_env.get_env_parameters(manual_knolling_parameters=self.env_parameters,
                                             use_knolling_model=self.use_knolling_model, real_operate=self.real_operate,
                                             evaluations=self.evaluations, obs_order=self.obs_order)
        # env_parameters = {
        #     'total_offset': [0.016, -0.17 + 0.016, 0], 'gap_item': 0.015, 'gap_block': 0.015, 'random_offset': False,
        #     'area_num': 2, 'ratio_num': 1, 'boxes_num': 10,
        #     'item_odd_prevent': True, 'block_odd_prevent': True, 'upper_left_max': True, 'forced_rotate_box': False,
        # }
        image_trim, self.manipulator_after = self.knolling_env.manual_knolling()
        cv2.imwrite('./image_sim/528_sim_tar.png', image_trim)
        _, self.arm_id = self.knolling_env.reset()

        if self.real_operate == True:

            with open(file="Cartisian_data/cmd.txt", mode="w", encoding="utf-8") as f:
                f.truncate(0)
            with open(file="Cartisian_data/real.txt", mode="w", encoding="utf-8") as f:
                f.truncate(0)
            with open(file="Cartisian_data/step.txt", mode="w", encoding="utf-8") as f:
                f.truncate(0)

            HOST = "192.168.0.186"  # Standard loopback interface address (localhost)
            PORT = 8881  # Port to listen on (non-privileged ports are > 1023)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((HOST, PORT))
            # It should be an integer from 1 to 65535, as 0 is reserved. Some systems may require superuser privileges if the port number is less than 8192.
            # associate the socket with a specific network interface
            s.listen()
            print(f"Waiting for connection...\n")
            conn, addr = s.accept()
            print(conn)
            print(f"Connected by {addr}")
            table_surface_height = 0.032
            sim_table_surface_height = -0.01
            num_motor = 5
            # ! reset the pos in both real and sim
            reset_pos = [0.005, 0, 0.1]
            ik_angles = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=reset_pos,
                                                     maxNumIterations=300,
                                                     targetOrientation=p.getQuaternionFromEuler([0, 1.57, 0]))
            reset_real = np.asarray(real_cmd2tarpos(rad2cmd(ik_angles[0:5])), dtype=np.float32)
            conn.sendall(reset_real.tobytes())

            for i in range(num_motor):
                p.setJointMotorControl2(self.arm_id, i, p.POSITION_CONTROL, targetPosition=ik_angles[i],
                                        maxVelocity=3)
            for _ in range(30):
                p.stepSimulation()
                # time.sleep(1 / 24)
        else:
            conn = None
            table_surface_height = 0.032
            sim_table_surface_height = -0.01

        #######################################################################################
        # 1: clean_desk + clean_item, 3: knolling, 4: check_accuracy of knolling, 5: get_camera
        # self.planning(1, conn, table_surface_height, sim_table_surface_height, evaluation)
        # error = self.planning(5, conn, table_surface_height, sim_table_surface_height, evaluation)
        error = self.planning(3, conn, table_surface_height, sim_table_surface_height)
        # self.planning(4, conn, table_surface_height, sim_table_surface_height, evaluation)
        #######################################################################################

        if self.obs_order == 'sim_image_obj_evaluate':
            return error

        if self.real_operate == True:
            end = np.array([0], dtype=np.float32)
            conn.sendall(end.tobytes())
        print(f'evaluation 0 over!!!!!')

if __name__ == '__main__':


    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("Device:", device)


    real_operate = False
    obs_order = 'sim_image_obj'
    check_detection_loss = False
    is_render = True
    use_knolling_model = False

    torch.manual_seed(42)
    # np.random.seed(202)
    # random.seed(202)

    env = knolling_main()
    evaluations = 3

    env_parameters = {
        'total_offset': [0.035, -0.17 + 0.016, 0], 'gap_item': 0.015, 'gap_block': 0.015, 'random_offset': False,
        'area_num': 2, 'ratio_num': 1, 'boxes_num': 10,
        'item_odd_prevent': True, 'block_odd_prevent': True, 'upper_left_max': True, 'forced_rotate_box': False,
    }

    env.get_parameters(evaluations=evaluations, is_render=is_render,
                       real_operate=real_operate, obs_order=obs_order,
                       check_detection_loss=check_detection_loss,
                       env_parameters=env_parameters, use_knolling_model=use_knolling_model
    )


    # for i in range(evaluations):
    env.step()