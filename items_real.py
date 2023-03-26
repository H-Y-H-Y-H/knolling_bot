import numpy as np
from stl import mesh
from cam_obs import *
import pyrealsense2 as rs
import math

class sort():
    
    def __init__(self):

        self.correct = []
        self.cube_2x2 = [0.016, 0.016, 0.012]
        self.cube_2x3 = [0.024, 0.016, 0.012]
        self.cube_2x4 = [0.032, 0.016, 0.012]
        # self.pencil = [0.01518, 0.09144, 0.01524]

        self.correct.append(self.cube_2x2)
        self.correct.append(self.cube_2x3)
        self.correct.append(self.cube_2x4)
        # self.correct.append(self.pencil)
        self.correct = np.asarray(self.correct).reshape(-1, 3)

        self.error_rate = 0.08

    def get_data_virtual(self, order_kinds, num_2x2, num_2x3, num_2x4, num_pencil):
        
        #! don't motify
        names = globals()
        xyz_list = []
        pos_list = []
        ori_list = []
        self.num_2x2 = num_2x2
        self.num_2x3 = num_2x3
        self.num_2x4 = num_2x4
        self.num_pencil = num_pencil

        for i in range(self.num_2x2):
            names[f'cube_{i}_dimension'] = mesh.Mesh.from_file('urdf/item_0/2x2.STL')
            xyz_list.append(names['cube_%d_dimension' % i].max_ - names['cube_%d_dimension' % i].min_)
        for i in range(self.num_2x3):
            names[f'cube_{i}_dimension'] = mesh.Mesh.from_file('urdf/item_1/2x3.STL')
            xyz_list.append(names['cube_%d_dimension' % i].max_ - names['cube_%d_dimension' % i].min_)
        for i in range(self.num_2x4):
            names[f'cube_{i}_dimension'] = mesh.Mesh.from_file('urdf/item_2/2x4.STL')
            xyz_list.append(names['cube_%d_dimension' % i].max_ - names['cube_%d_dimension' % i].min_)
        for i in range(self.num_pencil):
            names[f'cube_{i}_dimension'] = mesh.Mesh.from_file('urdf/item_3/%d.STL' % i)
            xyz_list.append(names['cube_%d_dimension' % i].max_ - names['cube_%d_dimension' % i].min_)
        xyz_list = np.asarray(xyz_list, dtype=np.float32)
        print(xyz_list)

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

            add = int((640 - 480) / 2)
            resized_color_image = cv2.copyMakeBorder(resized_color_image, add, add, 0, 0, cv2.BORDER_CONSTANT,
                                                     None, value=0)
            cv2.imwrite("Adjust_images/326_testpip_4.png", resized_color_image)

            cv2.waitKey(1)

        img = cv2.imread("Adjust_images/326_testpip_4.png")

        results = np.asarray(detect(img, real_operate=True, order_truth=None))
        results = np.asarray(results[:, :5]).astype(np.float32)
        # structure: x,y,yaw,length,width


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
                if np.linalg.norm(self.correct[i][0] - results[j][2]) < 0.003:
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
                    print('detect failed!!!')
            if len(kind_index) != 0:
                all_index.append(kind_index)

        # results = np.asarray(detect(img))
        # results = np.asarray(results[:, :5]).astype(np.float32)
        # # results[:, 2] = results[:, 2] * math.pi / 180
        # print('this is result', results)
        # print('this is correct', self.correct)
        # # structure: x,y,yaw,length,width
        #
        # all_index = []
        # new_xyz_list = []
        # kind = []
        # new_results = []
        # z = 0
        # roll = 0
        # pitch = 0
        # num = 0
        #
        # for i in range(len(self.correct)):
        #     kind_index = []
        #     for j in range(len(results)):
        #         if np.linalg.norm(self.correct[i][:2] - results[j][3:5]) < 0.003:
        #             kind_index.append(num)
        #             new_xyz_list.append(self.correct[i])
        #             num += 1
        #             if i in kind:
        #                 pass
        #             else:
        #                 kind.append(i)
        #             new_results.append(results[j])
        #         else:
        #             print('detect failed!!!')
        #     if len(kind_index) != 0:
        #         print(kind_index)
        #         all_index.append(kind_index)

        new_xyz_list = np.asarray(new_xyz_list)
        new_results = np.asarray(new_results)
        kind = np.asarray(kind)
        print('this is kind', kind)
        print('this is all index', all_index)
        print(new_xyz_list)

        # 按照234重新将result排序

        pos_before = []
        ori_before = []
        for i in range(len(all_index)):
            for j in range(len(all_index[i])):
                print(new_results[all_index[i][j]][0])
                print(new_results[all_index[i][j]][1])
                pos_before.append([new_results[all_index[i][j]][0], new_results[all_index[i][j]][1], z])
        pos_before = np.asarray(pos_before)
        ori_before = np.asarray(ori_before)
        print(pos_before)

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
                if abs(np.sum(item_xyz[j, :] - self.correct[order_kinds[i]])) < np.sum(self.correct[order_kinds[i]]) * self.error_rate:
                    items_names[f'item_{order_kinds[i]}'].append(len(new_item_xyz))
                    new_item_xyz.append(list(item_xyz[j]))
                    # items_names[f'xyz_{i}'].append(list(item_xyz[j]))
            all_index.append(items_names[f'item_{order_kinds[i]}'])

        new_item_xyz = np.asarray(new_item_xyz).reshape(-1, 3)
        return new_item_xyz, item_pos, item_ori, all_index

if __name__ == '__main__':

    order_kinds = [1, 2, 0]
    SORT = sort()
    xyz_list, _, _, all_index = SORT.get_data_virtual(order_kinds, num_2x2=2, num_2x3=1, num_2x4=4, num_pencil=0)
    print(xyz_list)
    print(all_index)