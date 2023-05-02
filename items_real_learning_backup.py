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

    def get_data_real(self):
        # 没法指定顺序了！只能234
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

        results = yolov8_predict(img_path=img_path,
                                 real_flag=True,
                                 target=None)
        # structure of results: x, y, length, width, ori


        all_index = []
        new_xyz_list = []
        kind = []
        new_results = []
        z = 0
        roll = 0
        pitch = 0
        num = 0
        for i in range(len(self.correct)):
            kind_index = []
            for j in range(len(results)):
                # if np.linalg.norm(self.correct[i][:2] - results[j][3:5]) < 0.003:
                if np.abs(self.correct[i][0] - results[j][0]) < 0.002 and np.abs(self.correct[i][1] - results[j][1]) < 0.002:
                    kind_index.append(num)
                    new_xyz_list.append(self.correct[i])
                    num += 1
                    if i in kind:
                        pass
                    else:
                        kind.append(i)
                    new_results.append(results[j])
                else:
                    pass
            if len(kind_index) != 0:
                all_index.append(kind_index)

        new_xyz_list = np.asarray(new_xyz_list)
        new_results = np.asarray(new_results)
        kind = np.asarray(kind)
        print('this is kind', kind)
        print('this is all index', all_index)
        print(new_xyz_list)

        # 按照234重新将result排序
        pos_before = np.concatenate((new_results[:, :2], np.zeros(len(new_results).reshape(-1, 1))), axis=1)
        ori_before = np.concatenate((np.zeros((len(new_results)), 2), new_results[:, 4]), axis=1)

        return new_xyz_list, pos_before, ori_before, all_index, kind
    
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