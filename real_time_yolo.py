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

def real_time_yolo(num_box_one_img, evaluations=None):

    model = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_bot/ultralytics/yolo_runs/train_standard_518_3/weights/best.pt'

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

    mean_floor = (160, 160 ,160)

    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        color_colormap_dim = color_image.shape
        resized_color_image = color_image
        # img_path = 'Test_images/image_real'

        # resized_color_image[np.concatenate((np.arange(10), np.arange(470, 480))), :, ] = mean_floor
        # resized_color_image[:, np.concatenate((np.arange(60), np.arange(580, 640))), ] = mean_floor


        img_path = './real_world_data_demo/cfg_4_520/images_before/%d/image_%d' % (num_box_one_img, evaluations)

        # img = adjust_img(img)

        cv2.imwrite(img_path + '.png', resized_color_image)
        img_path_input = img_path + '.png'
        args = dict(model=model, source=img_path_input, conf=0.4, iou=0.2)
        use_python = True
        if use_python:
            from ultralytics import YOLO
            images = YOLO(model)(**args)
        else:
            predictor = PosePredictor(overrides=args)
            predictor.predict_cli()
        device = 'cuda:0'

        origin_img = cv2.imread(img_path_input)

        use_xylw = False  # use lw or keypoints to export length and width

        one_img = images[0]

        pred_result = []
        pred_xylws = one_img.boxes.xywhn.cpu().detach().numpy()
        pred_keypoints = one_img.keypoints.cpu().detach().numpy()
        pred_keypoints[:, :, :2] = pred_keypoints[:, :, :2] / np.array([640, 480])
        pred_keypoints = pred_keypoints.reshape(len(pred_xylws), -1)

        pred = np.concatenate((np.zeros((len(pred_xylws), 1)), pred_xylws, pred_keypoints), axis=1)
        pred_order = np.lexsort((pred[:, 2], pred[:, 1]))

        pred_test = np.copy(pred[pred_order])
        for i in range(len(pred) - 1):
            if np.abs(pred_test[i, 1] - pred_test[i + 1, 1]) < 0.01:
                if pred_test[i, 2] < pred_test[i + 1, 2]:
                    # ground_truth_pose[order_ground_truth[i]], ground_truth_pose[order_ground_truth[i+1]] = ground_truth_pose[order_ground_truth[i+1]], ground_truth_pose[order_ground_truth[i]]
                    pred_order[i], pred_order[i + 1] = pred_order[i + 1], pred_order[i]
                    print('pred change the order!')
                else:
                    pass
        print('this is the pred order', pred_order)

        pred = pred[pred_order]
        pred_xylws = pred_xylws[pred_order]
        pred_keypoints = pred_keypoints[pred_order]


        for j in range(len(pred_xylws)):

            pred_keypoint = pred_keypoints[j].reshape(-1, 3)
            pred_xylw = pred_xylws[j]

            # pred_name = elements.names
            # pred_label = (f'{pred_name}')

            # plot pred
            print('this is pred xylw', pred_xylw)
            # print('this is pred cos sin', pred_cos_sin)
            origin_img, result = plot_and_transform(im=origin_img, box=pred_xylw, label='0:, predic', color=(0, 0, 0),
                                                    txt_color=(255, 255, 255), index=j,
                                                    scaled_xylw=pred_xylw, keypoints=pred_keypoint, use_xylw=use_xylw,
                                                    truth_flag=False)
            pred_result.append(result)
            # print('this is j', j)

        cv2.namedWindow('zzz', 0)
        cv2.imshow('zzz', origin_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            img_path_output = img_path + '_pred.png'
            cv2.imwrite(img_path_output, origin_img)
            break

    pred_result = np.asarray(pred_result)
    return pred_result

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

    num_collect_img = 1
    num_box_one_img = 10
    total_offset = [0.016, -0.17 + 0.016, 0]

    origin_point = np.array([0, -0.2])

    for i in range(num_collect_img):
        real_time_yolo_results = real_time_yolo(num_box_one_img, i)
        new_item_lw, new_item_pos_before, new_item_ori_before, all_index, transform_flag, new_urdf_index = label_sort(real_time_yolo_results)

        calculate_reorder = configuration_zzz(new_item_lw, all_index, gap_item, gap_block,
                                              transform_flag, configuration,
                                              item_odd_prevent, block_odd_prevent, upper_left_max,
                                              forced_rotate_box)

        # determine the center of the tidy configuration
        items_pos_list, items_ori_list = calculate_reorder.calculate_block()
        items_pos_list = items_pos_list + total_offset

        distance = np.linalg.norm(items_pos_list[:, :2] - origin_point, axis=1)
        order = np.argsort(distance)
        items_pos_list = items_pos_list[order]
        items_ori_list = items_ori_list[order]
        new_item_lw = new_item_lw[order]
        new_item_pos_before = new_item_pos_before[order]
        new_item_ori_before = new_item_ori_before[order]
        items_ori_list_arm = np.copy(items_ori_list)

        print('this is pos list\n', items_pos_list)
        print('this is ori list\n', items_ori_list)

        items_ori_list[:, 2] = 0
        data_after = np.concatenate((items_pos_list[:, :2], new_item_lw[:, :2], items_ori_list[:, 2].reshape(-1, 1)), axis=1)
        np.savetxt('./real_world_data_demo/cfg_4_520/labels_after/label_%d_%d.txt' % (num_box_one_img, i), data_after, fmt='%.03f')