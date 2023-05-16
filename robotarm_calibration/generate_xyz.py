import sys
sys.path.append('../')

import numpy as np
import pyrealsense2 as rs
import pybullet_data as pd
import math
from turdf import *
import socket
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from shapely.geometry import Polygon

torch.manual_seed(42)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(3, 12)
        self.fc2 = nn.Linear(12, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 12)
        self.fc6 = nn.Linear(12, 3)

    def forward(self, x):
        # define forward pass
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

    def loss(self, pred, target):
        value = (pred - target) ** 2
        return torch.mean(value)

class Arm:

    def __init__(self, is_render=True):

        self.kImageSize = {'width': 480, 'height': 480}
        self.urdf_path = '../urdf'
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

        self.correct = np.array([[0.016, 0.016, 0.012],
                                [0.024, 0.016, 0.012],
                                [0.032, 0.016, 0.012],
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

        if random.uniform(0,1) > 0.5:
            p.configureDebugVisualizer(lightPosition=[random.randint(3,5), random.randint(3,5), 5])
        else:
            p.configureDebugVisualizer(lightPosition=[random.randint(3,5), random.randint(-5, -3), 5])
        # p.configureDebugVisualizer(lightPosition=[5, 0, 5], shadowMapIntensity=0.9)
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                     cameraYaw=45,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[0.1, 0, 0])
        p.setAdditionalSearchPath(pd.getDataPath())

    def get_parameters(self, num_2x2=0, num_2x3=0, num_2x4=0, num_pencil=0,
                       total_offset=[0.1, 0, 0.006], grasp_order=[1, 0, 2],
                       gap_item=0.03, gap_block=0.02,
                       real_operate=False, obs_order='1',
                       random_offset = False, check_obs_error=None):

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
        self.check_obs_error = check_obs_error
        if self.check_obs_error == True:
            total_loss = check_dataset() # run here to check the error of the dataset
            np.savetxt('check_obs_error', total_loss)
        else:
            pass
        self.test_error_motion = True


    def get_obs(self, order, evaluation):

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
                cv2.imwrite("test_without_marker.png", resized_color_image)

                cv2.waitKey(1)

            img = cv2.imread("test_without_marker.png")

            # structure: x,y,length,width,yaw
            results = np.asarray(
                detect(img, evaluation=evaluation, real_operate=self.real_operate, all_truth=None, order_truth=None))
            results = np.asarray(results[:, :5]).astype(np.float32)
            print('this is the result of yolo+resnet', results)
            pred_cos = np.cos(2 * results[:, 4].reshape((-1, 1)))
            pred_sin = np.sin(2 * results[:, 4].reshape((-1, 1)))
            pred_compare = np.concatenate((results[:, 2:4], pred_cos, pred_sin), axis=1)
            pred_compare_2 = np.concatenate((results[:, 2:4], -pred_cos, -pred_sin), axis=1)

            all_index = []
            new_xyz_list = []
            kind = []
            new_results = []
            ori_index = []
            z = 0
            roll = 0
            pitch = 0
            num = 0
            for i in range(len(self.correct)):
                kind_index = []
                for j in range(len(results)):
                    # if np.linalg.norm(self.correct[i][:2] - results[j][3:5]) < 0.003:
                    if np.linalg.norm(self.correct[i][0] - results[j][2]) < 0.003:
                        kind_index.append(num)
                        new_xyz_list.append(self.correct[i])
                        num += 1
                        if i in kind:
                            pass
                        else:
                            kind.append(i)
                        ori_index.append(j)
                        new_results.append(results[j])
                    else:
                        pass
                        print('detect failed!!!')
                if len(kind_index) != 0:
                    all_index.append(kind_index)
            new_xyz_list = np.asarray(new_xyz_list)
            ori_index = np.asarray(ori_index)
            new_results = np.asarray(new_results)
            print(new_results)
            print(all_index)

            manipulator_before = []
            for i in range(len(all_index)):
                for j in range(len(all_index[i])):
                    manipulator_before.append(
                        [new_results[all_index[i][j]][0], new_results[all_index[i][j]][1], z, roll, pitch,
                         new_results[all_index[i][j], 4]])
            manipulator_before = np.asarray(manipulator_before)

            print('this is the result of dectection before changing the sequence\n', results)
            print('this is manipulator before after the detection \n', manipulator_before)

            return manipulator_before, new_xyz_list

        if order == 'real_image_obj':
            manipulator_before, new_xyz_list = get_real_image_obs()
            return manipulator_before, new_xyz_list

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

        baseid = p.loadURDF(os.path.join(self.urdf_path, "plane_zzz.urdf"), basePosition=[-10.11, -3.0165, 0], useFixedBase=1,
                            flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        self.arm_id = p.loadURDF(os.path.join(self.urdf_path, "robot_arm928/robot_arm1.urdf"),
                                 basePosition=[-0.08, 0, 0.02], useFixedBase=True,
                                 flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        textureId = p.loadTexture(os.path.join(self.urdf_path, "img_1.png"))
        p.changeDynamics(baseid, -1, lateralFriction=1, spinningFriction=1, rollingFriction=0.002, linearDamping=0.5, angularDamping=0.5)
        p.changeDynamics(self.arm_id, 7, lateralFriction=1, spinningFriction=1, rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.arm_id, 8, lateralFriction=1, spinningFriction=1, rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeVisualShape(baseid, -1, textureUniqueId=textureId,rgbaColor=[np.random.uniform(0.9,1), np.random.uniform(0.9,1),np.random.uniform(0.9,1), 1])

        # set the initial pos of the arm
        ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=[0.015, 0, 0.1],
                                                  maxNumIterations=200,
                                                  targetOrientation=p.getQuaternionFromEuler([0, math.pi / 2, 0]))
        for motor_index in range(self.num_motor):
            p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                    targetPosition=ik_angles0[motor_index], maxVelocity=7)
        for i in range(60):
            p.stepSimulation()
        return self.get_obs('images', None)

    def planning(self, order, conn, real_height, sim_height, evaluation):

        def get_start_end():  # generating all trajectories of all items in normal condition
            z = 0
            roll = 0
            pitch = 0
            if self.obs_order == 'sim_image_obj_evaluate':
                manipulator_before, new_xyz_list, error = self.get_obs(self.obs_order, evaluation)
                return error
            else:
                manipulator_before, new_xyz_list = self.get_obs(self.obs_order, evaluation)

            # sequence pos_before, ori_before, pos_after, ori_after
            start_end = []
            for i in range(len(self.obj_idx)):
                start_end.append([manipulator_before[i][0], manipulator_before[i][1], z, roll, pitch, manipulator_before[i][5],
                                  self.manipulator_after[i][0], self.manipulator_after[i][1], z, roll, pitch, self.manipulator_after[i][5]])
            start_end = np.asarray((start_end))
            print('get start and end')

            return start_end

        def Cartesian_offset_nn(xyz_input):

            # input:(n, 3), output: (n, 3)

            input_sc = [[-0.01, -0.201, -0.01],
                        [0.30, 0.201, 0.0601]]
            output_sc = [[-0.01, -0.201, -0.01],
                         [0.30, 0.201, 0.0601]]
            input_sc = np.load('nn_data_xyz/all_distance_free_new/real_scale.npy')
            output_sc = np.load('nn_data_xyz/all_distance_free_new/cmd_scale.npy')

            scaler_output = MinMaxScaler()
            scaler_input = MinMaxScaler()
            scaler_output.fit(output_sc)
            scaler_input.fit(input_sc)

            model = Net().to(device)
            model.load_state_dict(torch.load("model_pt_xyz/all_distance_free_new.pt"))
            # print(model)
            model.eval()
            with torch.no_grad():
                xyz_input_scaled = scaler_input.transform(xyz_input).astype(np.float32)
                xyz_input_scaled = torch.from_numpy(xyz_input_scaled)
                xyz_input_scaled = xyz_input_scaled.to(device)
                pred_xyz = model.forward(xyz_input_scaled)
                # print(pred_angle)
                pred_xyz = pred_xyz.cpu().data.numpy()
                xyz_output = scaler_output.inverse_transform(pred_xyz)

            return xyz_output

        def move(cur_pos, cur_ori, tar_pos, tar_ori):

            if tar_ori[2] > 1.58:
                tar_ori[2] = tar_ori[2] - np.pi
            elif tar_ori[2] < -1.58:
                tar_ori[2] = tar_ori[2] + np.pi

            d = np.array([0, 0.3])
            d_y = np.array((0, 0.17, 0.21, 0.30))
            d_y = d
            z_bias = np.array([-0.005, 0.004])
            x_bias = np.array([-0.002, 0.00])  # yolo error is +2mm along x axis!
            y_bias = np.array([0, -0.004, -0.001, 0.004])
            y_bias = np.array([0.002, 0.006])
            # z_parameters = np.polyfit(d, z_bias, 3)
            z_parameters = np.polyfit(d, z_bias, 1)
            x_parameters = np.polyfit(d, x_bias, 1)
            y_parameters = np.polyfit(d_y, y_bias, 1)
            new_z_formula = np.poly1d(z_parameters)
            new_x_formula = np.poly1d(x_parameters)
            new_y_formula = np.poly1d(y_parameters)

            # # automatically add z and x bias
            # # d = np.array([0, 0.10, 0.185, 0.225, 0.27])
            # d = np.array([0, 0.3])
            # d_y = np.array((0, 0.17, 0.30))
            # z_bias = np.array([-0.003, 0.008])
            # x_bias = np.array([-0.005, -0.001])
            # y_bias = np.array([0, -0.004, 0.004])
            # # y_bias = np.array([])
            # # z_parameters = np.polyfit(d, z_bias, 3)
            # z_parameters = np.polyfit(d, z_bias, 1)
            # x_parameters = np.polyfit(d, x_bias, 1)
            # y_parameters = np.polyfit(d_y, y_bias, 4)
            # new_z_formula = np.poly1d(z_parameters)
            # new_x_formula = np.poly1d(x_parameters)
            # new_y_formula = np.poly1d(y_parameters)

            distance = tar_pos[0]
            distance_y = tar_pos[0]
            tar_pos[2] = tar_pos[2] + new_z_formula(distance)
            print('this is z', new_z_formula(distance))
            tar_pos[0] = tar_pos[0] + new_x_formula(distance)
            print('this is x', new_x_formula(distance))
            if tar_pos[1] > 0:
                tar_pos[1] += new_y_formula(distance_y) * np.clip((6 * (tar_pos[1] + 0.01)), 0, 1)
            else:
                tar_pos[1] -= new_y_formula(distance_y) * np.clip((6 * (tar_pos[1] - 0.01)), 0, 1)
            print('this is tar pos after manual', tar_pos)

            # distance = tar_pos[0]
            # tar_pos[2] = tar_pos[2] + new_z_formula(distance)
            # print('this is z', new_z_formula(distance))
            # tar_pos[0] = tar_pos[0] + new_x_formula(distance)
            # print('this is x', new_x_formula(distance))
            # # distance_y = np.linalg.norm(tar_pos[:2])
            # if tar_pos[1] > 0:
            #     distance_y = np.linalg.norm(tar_pos[:2])
            #     print('this is y', new_y_formula(distance_y))
            #     tar_pos[1] += new_y_formula(distance_y)
            # else:
            #     distance_y = np.linalg.norm(tar_pos[:2])
            #     print('this is y', new_y_formula(distance_y))
            #     tar_pos[1] -= new_y_formula(distance_y)
            # print('this is tar pos after manual', tar_pos)


            if abs(cur_pos[0] - tar_pos[0]) < 0.001 and abs(cur_pos[1] - tar_pos[1]) < 0.001:
                # vertical, choose a small slice
                move_slice = 0.004
            else:
                # horizontal, choose a large slice
                move_slice = 0.008

            if self.real_operate == True:
                tar_pos = tar_pos + np.array([0, 0, real_height])
                target_pos = np.copy(tar_pos)
                target_ori = np.copy(tar_ori)
                # target_pos[2] = Cartesian_offset_nn(np.array([tar_pos])).reshape(-1, )[2] # remove nn offset temporary

                vertical_flag = False
                print('this is tar pos', target_pos)
                print('this is cur pos', cur_pos)
                if np.abs(target_pos[2] - cur_pos[2]) > 0.01 \
                        and np.abs(target_pos[0] - cur_pos[0]) < 0.01\
                        and np.abs(target_pos[1] - cur_pos[1]) < 0.01:
                    print('we dont need feedback control')
                    mark_ratio = 0.8
                    vertical_flag = True
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

                    print('this is cur pos', cur_pos)
                    print('this is seg pos', seg_pos)

                    while True:
                        tar_pos = cur_pos + step_pos
                        # print(tar_pos)
                        tar_ori = cur_ori + step_ori
                        sim_xyz.append(tar_pos)
                        sim_ori.append(tar_ori)

                        ik_angles_sim = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=tar_pos,
                                                                  maxNumIterations=200,
                                                                  targetOrientation=p.getQuaternionFromEuler(tar_ori))

                        for motor_index in range(self.num_motor):
                            p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                                    targetPosition=ik_angles_sim[motor_index], maxVelocity=2.5)
                        for i in range(20):
                            p.stepSimulation()

                        angle_sim = np.asarray(real_cmd2tarpos(rad2cmd(ik_angles_sim[0:5])), dtype=np.float32)
                        plot_cmd.append(angle_sim)

                        break_flag = abs(seg_pos[0] - tar_pos[0]) < 0.001 and abs(
                            seg_pos[1] - tar_pos[1]) < 0.001 and abs(seg_pos[2] - tar_pos[2]) < 0.001 and \
                                     abs(seg_ori[0] - tar_ori[0]) < 0.001 and abs(
                            seg_ori[1] - tar_ori[1]) < 0.001 and abs(seg_ori[2] - tar_ori[2]) < 0.001
                        if break_flag:
                            break

                        cur_pos = tar_pos
                        cur_ori = tar_ori

                    sim_xyz = np.asarray(sim_xyz)

                    plot_step = np.arange(num_step)
                    plot_cmd = np.asarray(plot_cmd)
                    print('this is the shape of cmd', plot_cmd.shape)
                    print('this is the shape of xyz', sim_xyz.shape)
                    # print('this is the motor pos sent', plot_cmd[-1])
                    conn.sendall(plot_cmd.tobytes())
                    # if vertical_flag == True:
                    #     print('sleep')
                    #     time.sleep(5)
                    # print('waiting the manipulator')
                    angles_real = conn.recv(4096)
                    # print('received')
                    angles_real = np.frombuffer(angles_real, dtype=np.float32)
                    angles_real = angles_real.reshape(-1, 6)
                    # ik_angles_real = []
                    # for i in range(len(angles_real)):
                    #     ik_angles_real = np.asarray(cmd2rad(real_tarpos2cmd(angles_real[i])), dtype=np.float32)
                    #     for motor_index in range(self.num_motor):
                    #         p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                    #                                 targetPosition=ik_angles_real[motor_index], maxVelocity=25)
                    #     for i in range(40):
                    #         p.stepSimulation()
                    #     real_xyz.append(p.getLinkState(self.arm_id, 9)[0])
                    # real_xyz = np.asarray(real_xyz)
                    #
                    # # update cur_pos based on real pos of the arm
                    # cur_pos = real_xyz[-1]
                    # if abs(seg_pos[0] - cur_pos[0]) < 0.001 and abs(seg_pos[1] - cur_pos[1]) < 0.001 and abs(
                    #         seg_pos[2] - cur_pos[2]) < 0.001:
                    #     seg_flag = True
                    #     break
                    # print('this is seg_time', seg_time)
                    if seg_time > 0:
                        seg_flag = False
                        print('segment fail, try to tune!')
                        ik_angles_sim = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=target_pos,
                                                                     maxNumIterations=200,
                                                                     targetOrientation=p.getQuaternionFromEuler(
                                                                         target_ori))

                        for motor_index in range(self.num_motor):
                            p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                                    targetPosition=ik_angles_sim[motor_index], maxVelocity=2.5)
                        for i in range(20):
                            p.stepSimulation()

                        angle_sim = np.asarray(real_cmd2tarpos(rad2cmd(ik_angles_sim[0:5])), dtype=np.float32)
                        final_cmd = np.append(angle_sim, 0).reshape(1, -1)
                        final_cmd = np.asarray(final_cmd, dtype=np.float32)
                        print(final_cmd.shape)
                        print(final_cmd)
                        conn.sendall(final_cmd.tobytes())

                        # get the pos after tune!
                        final_angles_real = conn.recv(4096)
                        # print('received')
                        final_angles_real = np.frombuffer(final_angles_real, dtype=np.float32).reshape(-1, 6)
                        print('this is the shape of final angles real', final_angles_real.shape)

                        ik_angles_real = np.asarray(cmd2rad(real_tarpos2cmd(final_angles_real)), dtype=np.float32)
                        for motor_index in range(self.num_motor):
                            p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                                    targetPosition=ik_angles_real[motor_index], maxVelocity=25)
                        for i in range(40):
                            p.stepSimulation()
                        real_xyz = np.append(real_xyz, np.asarray(p.getLinkState(self.arm_id, 9)[0])).reshape(-1, 3)
                        cur_pos = real_xyz[-1]
                        print('this is cur pos after pid', cur_pos)
                        break
                    else:
                        print('this is the shape of angles real', angles_real.shape)
                        for i in range(len(angles_real)):
                            ik_angles_real = np.asarray(cmd2rad(real_tarpos2cmd(angles_real[i])), dtype=np.float32)
                            for motor_index in range(self.num_motor):
                                p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                                        targetPosition=ik_angles_real[motor_index], maxVelocity=25)
                            for i in range(40):
                                p.stepSimulation()
                            real_xyz = np.append(real_xyz, np.asarray(p.getLinkState(self.arm_id, 9)[0])).reshape(-1, 3)
                        cur_pos = real_xyz[-1]
                        break

                if self.test_error_motion == True:
                    with open(file="nn_data_xyz/free_2/cmd_xyz_nn.csv", mode="a", encoding="utf-8") as f:
                        sim_xyz = sim_xyz.tolist()
                        for i in range(len(sim_xyz)):
                            list_zzz = [str(j) for j in sim_xyz[i]]
                            f.writelines(' '.join(list_zzz))
                            f.write('\n')
                    with open(file="nn_data_xyz/free_2/real_xyz_nn.csv", mode="a", encoding="utf-8") as f:
                        real_xyz = real_xyz.tolist()
                        for i in range(len(real_xyz)):
                            list_zzz = [str(j) for j in real_xyz[i]]
                            f.writelines(' '.join(list_zzz))
                            f.write('\n')
                    with open(file="nn_data_xyz/free_2/cmd_nn.csv", mode="a", encoding="utf-8") as f:
                        plot_cmd = plot_cmd.tolist()
                        for i in range(len(plot_cmd)):
                            list_zzz = [str(j) for j in plot_cmd[i]]
                            f.writelines(' '.join(list_zzz))
                            f.write('\n')
                    with open(file="nn_data_xyz/free_2/real_nn.csv", mode="a", encoding="utf-8") as f:
                        angles_real = angles_real.tolist()
                        for i in range(len(angles_real)):
                            list_zzz = [str(j) for j in angles_real[i]]
                            f.writelines(' '.join(list_zzz))
                            f.write('\n')
                    with open(file="step.txt", mode="a", encoding="utf-8") as f:
                        list_zzz = [str(j) for j in plot_step]
                        f.writelines(' '.join(list_zzz))
                        f.write('\n')

                return cur_pos # return cur pos to let the manipualtor remember the improved pos

        def gripper(gap, obj_width=None):
            obj_width += 0.006
            obj_width_range = np.array([0.021, 0.026, 0.032, 0.039, 0.045, 0.052, 0.057])
            motor_pos_range = np.array([2000, 2100, 2200, 2300, 2400, 2500, 2600])
            formula_parameters = np.polyfit(obj_width_range, motor_pos_range, 3)
            motor_pos = np.poly1d(formula_parameters)

            if self.real_operate == True:
                if gap > 0.0265: # close
                    pos_real = np.asarray([[gap, 1600]], dtype=np.float32)
                elif gap <= 0.0265: # open
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
                p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL, targetPosition=gap, force=10)
                p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL, targetPosition=gap, force=10)
            for i in range(30):
                p.stepSimulation()
                # time.sleep(1 / 120)

        def knolling():

            rest_pos = np.array([0, 0, 0.05])
            rest_ori = np.array([0, 1.57, 1.57])
            offset_low = np.array([0, 0, 0.002])
            offset_high = np.array([0, 0, 0.035])

            last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
            last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))

            if self.test_error_motion == True:
                trajectory_pos_list = np.array([[0.01, 0.016],
                                                [0.01, 0.024],
                                                [0.01, 0.032],
                                                [0.01, 0.040],
                                                [0.01, 0.048]])
                # trajectory_pos_list = np.array([[0.24, -0.17, 0.03],
                #                                 [0.24, -0.17, 0.005],
                #                                 [0.24, 0.17, 0.03],
                #                                 [0.24, 0.17, 0.005]])
                for j in range(len(trajectory_pos_list)):

                    if len(trajectory_pos_list[j]) == 3:
                        last_pos = move(last_pos, last_ori, trajectory_pos_list[j], rest_ori)
                        # if trajectory_pos_list[j][2] < 0.02:
                        #     time.sleep(2)
                        time.sleep(2)
                        last_ori = np.copy(rest_ori)

                    elif len(trajectory_pos_list[j]) == 2:
                        gripper(trajectory_pos_list[j][0], trajectory_pos_list[j][1])
                        time.sleep(5)
            else:
                times = 2
                for j in range(times):

                    trajectory_pos_list = np.array([np.random.uniform(0, 0.28), np.random.uniform(-0.16, 0.16), np.random.uniform(0.032, 0.08)])

                    if len(trajectory_pos_list) == 3:
                        print('ready to move', trajectory_pos_list)
                        last_pos = move(last_pos, last_ori, trajectory_pos_list, rest_ori)
                        # time.sleep(5)
                        # last_pos = np.copy(trajectory_pos_list)
                        last_ori = np.copy(rest_ori)

                    elif len(trajectory_pos_list[j]) == 2:
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

        if order == 3:
            if self.obs_order == 'sim_image_obj_evaluate':
                error = knolling()
                return error
            else:
                knolling()

    def step(self, evaluation):

        if self.real_operate == True:

            with open(file="nn_data_xyz/free_2/cmd_xyz_nn.csv", mode="w", encoding="utf-8") as f:
                f.truncate(0)
            with open(file="nn_data_xyz/free_2/real_xyz_nn.csv", mode="w", encoding="utf-8") as f:
                f.truncate(0)
            with open(file="nn_data_xyz/free_2/cmd_nn.csv", mode="w", encoding="utf-8") as f:
                f.truncate(0)
            with open(file="nn_data_xyz/free_2/real_nn.csv", mode   ="w", encoding="utf-8") as f:
                f.truncate(0)
            with open(file="step.txt", mode="w", encoding="utf-8") as f:
                f.truncate(0)

            HOST = "192.168.0.186"  # Standard loopback interface address (localhost)
            PORT = 8880 # Port to listen on (non-privileged ports are > 1023)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((HOST, PORT))
            # It should be an integer from 1 to 65535, as 0 is reserved. Some systems may require superuser privileges if the port number is less than 8192.
            # associate the socket with a specific network interface
            s.listen()
            print('Test error motion:', self.test_error_motion)
            print(f"Waiting for connection...\n")
            conn, addr = s.accept()
            print(conn)
            print(f"Connected by {addr}")
            table_surface_height = 0.032
            sim_table_surface_height = 0
            num_motor = 5
            # ! reset the pos in both real and sim
            reset_pos = [0.015, 0, 0.1]
            ik_angles = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=reset_pos,
                                                     maxNumIterations=300,
                                                     targetOrientation=p.getQuaternionFromEuler(
                                                         [0, 1.57, 0]))
            reset_real = np.asarray(real_cmd2tarpos(rad2cmd(ik_angles[0:5])), dtype=np.float32)
            print('this is the reset motor pos', reset_real)
            conn.sendall(reset_real.tobytes())

            for i in range(num_motor):
                p.setJointMotorControl2(self.arm_id, i, p.POSITION_CONTROL, targetPosition=ik_angles[i],
                                        maxVelocity=3)
            for _ in range(200):
                p.stepSimulation()
                time.sleep(1 / 48)
        else:
            conn = None
            table_surface_height = 0.032
            sim_table_surface_height = 0

        #######################################################################################
        # self.planning(1, conn, table_surface_height, sim_table_surface_height, evaluation)
        # error = self.planning(5, conn, table_surface_height, sim_table_surface_height, evaluation)
        # self.planning(2, conn, table_surface_height, sim_table_surface_height, evaluation)
        error = self.planning(3, conn, table_surface_height, sim_table_surface_height, evaluation)
        # self.planning(4, conn, table_surface_height, sim_table_surface_height, evaluation)
        #######################################################################################

        if self.obs_order == 'sim_image_obj_evaluate':
            return error

        if self.real_operate == True:
            end = np.array([0], dtype=np.float32)
            conn.sendall(end.tobytes())
        print(f'evaluation {evaluation} over!!!!!')

if __name__ == '__main__':

    random_flag = True
    command = 'knolling'  # image virtual real

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("Device:", device)
    # model = Net().to(device)
    # model.load_state_dict(torch.load("Test_and_Calibration/nn_calibration/model_pt_xyz/combine_all_free_001005_far_free_001005_flank_free_001005_useful.pt"))
    # # print(model)
    # model.eval()

    if command == 'knolling':

        num_2x2 = 5
        num_2x3 = 5
        num_2x4 = 5
        total_offset = [0.15, 0, 0.006]
        grasp_order = [1, 0, 2]
        gap_item = 0.015
        gap_block = 0.02
        random_offset = True
        real_operate = True
        obs_order = 'real_image_obj'
        check_dataset_error = False


        env = Arm(is_render=True)
        env.get_parameters(num_2x2=num_2x2, num_2x3=num_2x3, num_2x4=num_2x4,
                           total_offset=total_offset, grasp_order=grasp_order,
                           gap_item=gap_item, gap_block=gap_block,
                           real_operate=real_operate, obs_order=obs_order,
                           random_offset=random_offset, check_obs_error=check_dataset_error)
        evaluations = 1

        for i in range(evaluations):
            # image_trim = env.change_config()
            _ = env.reset()
            env.step(i)