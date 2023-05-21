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

manual_measure = False
movie = False

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

    model = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_bot/ultralytics/yolo_runs/train_standard_519/weights/best.pt'

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

    total_pred_result = []
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        color_colormap_dim = color_image.shape
        resized_color_image = color_image
        # print(resized_color_image)
        # img_path = 'Test_images/image_real'

        # resized_color_image[np.concatenate((np.arange(10), np.arange(470, 480))), :, ] = mean_floor
        # resized_color_image[:, np.concatenate((np.arange(60), np.arange(580, 640))), ] = mean_floor

        if manual_measure == False:
            img_path = './real_world_data_demo/cfg_4_520/images_before/%d/image_%d' % (num_box_one_img, evaluations)
        else:
            img_path = './real_world_data_demo/test_yolo_lw_loss/images_before/%d/image_%d' % (num_box_one_img, evaluations)
        if movie == True:
            img_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/yolo_pose4keypoints_tuning/origin_images/'
        # img = adjust_img(img)
        if movie == True:
            cv2.imwrite(img_path + '%012d.png' % evaluations, resized_color_image)
            img_path_input = img_path + '%012d.png' % evaluations
        else:
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
        if len(pred_xylws) == 0:
            continue
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
            # print('this is pred xylw', pred_xylw)
            # print('this is pred cos sin', pred_cos_sin)
            origin_img, result = plot_and_transform(im=origin_img, box=pred_xylw, label='0:, predic', color=(0, 0, 0),
                                                    txt_color=(255, 255, 255), index=j,
                                                    scaled_xylw=pred_xylw, keypoints=pred_keypoint, use_xylw=use_xylw,
                                                    truth_flag=False)
            pred_result.append(result)
            # print('this is j', j)

        pred_result = np.asarray(pred_result)
        # total_pred_result.append(pred_result)
        # print(pred_result)


        cv2.namedWindow('zzz', 0)
        cv2.resizeWindow('zzz', 1280, 960)
        cv2.imshow('zzz', origin_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            if movie == True:
                img_path_output = img_path + 'pred/%012d.png' % evaluations
                cv2.imwrite(img_path_output, origin_img)
            else:
                img_path_output = img_path + '_pred.png'
                cv2.imwrite(img_path_output, origin_img)
            break
    total_pred_result = np.asarray(total_pred_result)
    # print('this is total pred result', total_pred_result)
    # pred_result = np.concatenate((np.mean(total_pred_result, axis=0), np.max(total_pred_result, axis=0)), axis=1)
    print('this is pred result\n', pred_result)

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

    num_collect_img_start = 0
    num_collect_img_end = 10
    num_box_one_img = 10
    total_offset = [0.016, -0.17 + 0.016, 0]

    origin_point = np.array([0, -0.2])

    total_loss = []

    for i in range(num_collect_img_start, num_collect_img_end):
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

        if movie == False:
            if manual_measure == False:
                with open('real_after_pred.txt', 'r') as file:
                    data = file.read().replace(',', ' ')
                    data = list(data.split())
                    temp_data = np.array([float(d) for d in data]).reshape(-1, 5)[:10]
                    target = np.copy(temp_data)[:, 2:]
                    # print('demo target', target)
            else:
                data = input('please input the size (length and width):')
                # print(data)
                data = list(data.split())
                temp_data = np.array([float(d) for d in data]).reshape(num_box_one_img, -1)
                target = np.copy(temp_data)
                print(target)
                # print(target.shape)

            target_exist_list = []
            pred_exist_list = []
            temp_item_lw = []
            temp_item_pos = []
            temp_item_ori = []
            print('this is new item lw\n', new_item_lw)
            print('this is known label\n', target)

            if manual_measure == False:

                for j in range(len(new_item_lw)):
                    for m in range(len(target)):
                        if (np.abs(new_item_lw[j, 0] - target[m, 0]) < 0.002 and np.abs(new_item_lw[j, 1] - target[m, 1]) < 0.002):
                            if m not in target_exist_list:
                                print(f'new item lw {j} match temp data {m}!')
                                target_exist_list.append(m)
                                pred_exist_list.append(j)
                                target[m, :2] = new_item_lw[j, :2]
                                # target[m, 2:4] = new_item_lw[j, :2]
                                temp_item_pos.append(target[m, :2])
                                temp_item_ori.append(target[m, :4])
                                break
                        elif (np.abs(new_item_lw[j, 1] - target[m, 0]) < 0.002 and np.abs(new_item_lw[j, 0] - target[m, 1]) < 0.002):
                            if m not in target_exist_list:
                                print(f'new item lw {j} match temp data {m}!')
                                target_exist_list.append(m)
                                pred_exist_list.append(j)
                                temp = new_item_lw[j][0]
                                new_item_lw[j][0] = new_item_lw[j][1]
                                new_item_lw[j][1] = temp
                                target[m, :2] = new_item_lw[j, :2]
                                # target[m, 2:4] = new_item_lw[j, :2]
                                temp_item_pos.append(target[m, :2])
                                temp_item_ori.append(target[m, :4])
                                break

            #     # for j in range(len(new_item_lw)):
            #     #     for m in range(len(target)):
            #     #         if (np.abs(new_item_lw[j, 0] - target[m, 2]) < 0.002 and np.abs(new_item_lw[j, 1] - target[m, 3]) < 0.002):
            #     #             if m not in target_exist_list:
            #     #                 print(f'new item lw {j} match temp data {m}!')
            #     #                 target_exist_list.append(m)
            #     #                 pred_exist_list.append(j)
            #     #                 # target[m, :2] = new_item_lw[j, :2]
            #     #                 target[m, 2:4] = new_item_lw[j, :2]
            #     #                 temp_item_pos.append(target[m, :2])
            #     #                 temp_item_ori.append(target[m, :4])
            #     #                 break
            #     #         elif (np.abs(new_item_lw[j, 1] - target[m, 2]) < 0.002 and np.abs(new_item_lw[j, 0] - target[m, 3]) < 0.002):
            #     #             if m not in target_exist_list:
            #     #                 print(f'new item lw {j} match temp data {m}!')
            #     #                 target_exist_list.append(m)
            #     #                 pred_exist_list.append(j)
            #     #                 temp = new_item_lw[j][0]
            #     #                 new_item_lw[j][0] = new_item_lw[j][1]
            #     #                 new_item_lw[j][1] = temp
            #     #                 # target[m, :2] = new_item_lw[j, :2]
            #     #                 target[m, 2:4] = new_item_lw[j, :2]
            #     #                 temp_item_pos.append(target[m, :2])
            #     #                 temp_item_ori.append(target[m, :4])
            #     #                 break
            # else:
            #     for j in range(len(new_item_lw)):
            #         for m in range(len(target)):
            #             if (np.abs(new_item_lw[j, 0] - target[m, 0]) < 0.002 and np.abs(new_item_lw[j, 1] - target[m, 1]) < 0.002):
            #                 if m not in target_exist_list:
            #                     print(f'new item lw {j} match temp data {m}!')
            #                     target_exist_list.append(m)
            #                     pred_exist_list.append(j)
            #                     target[m, :2] = new_item_lw[j, :2]
            #                     # target[m, 2:4] = new_item_lw[j, :2]
            #                     temp_item_pos.append(target[m, :2])
            #                     temp_item_ori.append(target[m, :4])
            #                     break
            #             elif (np.abs(new_item_lw[j, 1] - target[m, 0]) < 0.002 and np.abs(new_item_lw[j, 0] - target[m, 1]) < 0.002):
            #                 if m not in target_exist_list:
            #                     print(f'new item lw {j} match temp data {m}!')
            #                     target_exist_list.append(m)
            #                     pred_exist_list.append(j)
            #                     temp = new_item_lw[j][0]
            #                     new_item_lw[j][0] = new_item_lw[j][1]
            #                     new_item_lw[j][1] = temp
            #                     target[m, :2] = new_item_lw[j, :2]
            #                     # target[m, 2:4] = new_item_lw[j, :2]
            #                     temp_item_pos.append(target[m, :2])
            #                     temp_item_ori.append(target[m, :4])
            #                     break

            if len(target_exist_list) != len(target):
                target_exist_list = np.asarray(target_exist_list)
                pred_exist_list = np.asarray(pred_exist_list)
                print(target_exist_list)
                if len(target_exist_list) == 0:
                    rest_target = target
                    rest_target_index = np.arange(num_box_one_img)
                    rest_pred = new_item_lw
                    rest_pred_backup = rest_pred[:, [1, 0, 2]]
                    rest_pred_index = np.arange(num_box_one_img)
                    rest_pred_index = np.tile(rest_pred_index, 2)
                else:
                    rest_target_index = np.delete(np.arange(num_box_one_img), target_exist_list)
                    rest_target = np.delete(target, target_exist_list, axis=0)
                    rest_pred = np.delete(new_item_lw, pred_exist_list, axis=0)
                    rest_pred_backup = rest_pred[:, [1, 0, 2]]
                    rest_pred = np.concatenate((rest_pred, rest_pred_backup), axis=0)
                    rest_pred_index = np.delete(np.arange(num_box_one_img), pred_exist_list)
                    rest_pred_index = np.tile(rest_pred_index, 2)

                print('rest_target', rest_target)
                print('rest_pred', rest_pred)
                if manual_measure == False:
                    for z in range(len(rest_target)):
                        add_index = np.argmin(np.linalg.norm(rest_pred - rest_target[z, :], axis=1))
                        print('this is add', add_index)
                        print(int(add_index - len(rest_pred) / 2))
                        print(int(add_index + len(rest_pred) / 2))
                        if add_index + 1 > int(len(rest_pred) / 2):
                            print(
                                f'target {target[rest_target_index[z], :]} matches pred{new_item_lw[rest_pred_index[add_index], [1, 0]]}, reverse')
                            target[rest_target_index[z], :2] = new_item_lw[rest_pred_index[add_index], [1, 0]]
                            rest_pred[add_index, :2].fill(-2)
                            rest_pred[int(add_index - len(rest_pred) / 2), :2].fill(-2)
                            rest_target[z, :].fill(0)
                        else:
                            print(
                                f'target {target[rest_target_index[z], :]} matches pred{new_item_lw[rest_pred_index[add_index], [0, 1]]}')
                            target[rest_target_index[z], :2] = new_item_lw[rest_pred_index[add_index], [0, 1]]
                            rest_pred[add_index, :2].fill(-2)
                            rest_pred[int(add_index + len(rest_pred) / 2), :2].fill(-2)
                            rest_target[z, :].fill(0)
                        print('this is reset target after', rest_target)
                        print('this is reset pred after', rest_pred)

                        # add_index = np.argmin(np.linalg.norm(rest_pred - rest_target[z, 2:], axis=1))
                        # if add_index >= len(rest_target):
                        #     print(f'target {target[rest_target_index[z], 2:4]} matches pred{new_item_lw[rest_pred_index[add_index], [1, 0]]}')
                        #     target[rest_target_index[z], 2:4] = new_item_lw[rest_pred_index[add_index], [1, 0]]
                        #     # rest_target[i, 2:] == np.array([0, 0, 0])
                        #     rest_target[z] == np.array([0, 0, 0])
                        # else:
                        #     print(f'target {target[rest_target_index[z], 2:4]} matches pred{new_item_lw[rest_pred_index[add_index], [0, 1]]}')
                        #     target[rest_target_index[z], 2:4] = new_item_lw[rest_pred_index[add_index], [0, 1]]
                        #     # rest_target[i, 2:] == np.array([0, 0, 0])
                        #     rest_target[z] == np.array([0, 0, 0])
                # else:
                #     for z in range(len(rest_target)):
                #         add_index = np.argmin(np.linalg.norm(rest_pred - rest_target[z, :], axis=1))
                #         print('this is add', add_index)
                #         print(int(add_index - len(rest_pred) / 2))
                #         print(int(add_index + len(rest_pred) / 2))
                #         if add_index + 1 > int(len(rest_pred) / 2):
                #             print(
                #                 f'target {target[rest_target_index[z], :]} matches pred{new_item_lw[rest_pred_index[add_index], [1, 0]]}, reverse')
                #             target[rest_target_index[z], :2] = new_item_lw[rest_pred_index[add_index], [1, 0]]
                #             rest_pred[add_index, :2].fill(-2)
                #             rest_pred[int(add_index - len(rest_pred) / 2), :2].fill(-2)
                #             rest_target[z, :].fill(0)
                #         else:
                #             print(
                #                 f'target {target[rest_target_index[z], :]} matches pred{new_item_lw[rest_pred_index[add_index], [0, 1]]}')
                #             target[rest_target_index[z], :2] = new_item_lw[rest_pred_index[add_index], [0, 1]]
                #             rest_pred[add_index, :2].fill(-2)
                #             rest_pred[int(add_index + len(rest_pred) / 2), :2].fill(-2)
                #             rest_target[z, :].fill(0)
                #         print('this is reset target after', rest_target)
                #         print('this is reset pred after', rest_pred)
                #
                #     # for z in range(len(rest_target)):
                #     #     add_index = np.argmin(np.linalg.norm(rest_pred - rest_target[z, :], axis=1))
                #     #     if add_index >= len(rest_target):
                #     #         print(f'target {target[rest_target_index[z], :]} matches pred{new_item_lw[rest_pred_index[add_index], [1, 0]]}')
                #     #         target[rest_target_index[z], :2] = new_item_lw[rest_pred_index[add_index], [1, 0]]
                #     #         # rest_target[i, 2:] == np.array([0, 0, 0])
                #     #         rest_target[z] == np.array([0, 0, 0])
                #     #     else:
                #     #         print(f'target {target[rest_target_index[z], :]} matches pred{new_item_lw[rest_pred_index[add_index], [1, 0]]}')
                #     #         target[rest_target_index[z], :2] = new_item_lw[rest_pred_index[add_index], [0, 1]]
                #     #         # rest_target[i, 2:] == np.array([0, 0, 0])
                #     #         rest_target[z] == np.array([0, 0, 0])

            loss = np.mean((target[:, :2] - temp_data[:, 2:4]), axis=0)
            print('this is temp data\n', temp_data)
            print('this is target\n', target)
            print('this is loss', loss)
            total_loss.append(loss)

            if manual_measure == False:
                target = np.concatenate((temp_data[:, :2], target), axis=1)
                np.savetxt('./real_world_data_demo/cfg_4_520/labels_after/label_%d_%d.txt' % (num_box_one_img, i),
                           target, fmt='%.04f')
            else:
                np.savetxt('./real_world_data_demo/test_yolo_lw_loss/labels_after/label_%d.txt' % (i), target,
                           fmt='%.04f')
        else:
            items_ori_list[:, 2] = 0
            # data_after = np.concatenate((items_pos_list[:, :2], new_item_lw[:, :2], items_ori_list[:, 2].reshape(-1, 1)), axis=1)
            data_before = np.concatenate(
                (np.ones(len(new_item_pos_before)).reshape(-1, 1), new_item_pos_before[:, :2], new_item_lw[:, :2], new_item_ori_before[:, 2].reshape(-1, 1)), axis=1)
            print('this is data before\n', data_before)
            np.savetxt('/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/yolo_pose4keypoints_tuning/origin_labels/origin_labels_pred/%012d.txt' % i, data_before)
            # print('this is pos list\n', items_pos_list)
            # print('this is ori list\n', items_ori_list)

    if movie == False:
        total_loss.append(np.mean(total_loss, axis=0))
        total_loss = np.asarray(total_loss)
        if manual_measure == False:
            np.savetxt('./real_world_data_demo/cfg_4_520/labels_after/label_%d_loss.txt' % (num_box_one_img), total_loss)
        else:
            np.savetxt('./real_world_data_demo/test_yolo_lw_loss/labels_after/%d/label_loss_%d.txt' % (num_box_one_img, num_collect_img_end), total_loss)
    else:
        pass

############## -0.0021 -0.0013