import numpy as np
import pyrealsense2 as rs
import pybullet_data as pd
import math
from func import *
import socket
import cv2
from urdfpy import URDF
import pyrealsense2 as rs
import sys
sys.path.append('/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_bot/ultralytics')
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

import torch

torch.manual_seed(42)

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


class Yolo_predict():

    def find_keypoints(self, xpos, ypos, l, w, ori, mm2px):

        gamma = ori
        rot_z = [[np.cos(gamma), -np.sin(gamma)],
                 [np.sin(gamma), np.cos(gamma)]]
        # rot_z = [[1, 0],
        #          [0, 1]]
        rot_z = np.asarray(rot_z)

        kp1 = np.asarray([l / 2, 0])
        kp2 = np.asarray([0, w / 2])
        kp3 = np.asarray([-l / 2, 0])
        kp4 = np.asarray([0, -w / 2])

        keypoint1 = np.dot(rot_z, kp1)
        keypoint2 = np.dot(rot_z, kp2)
        keypoint3 = np.dot(rot_z, kp3)
        keypoint4 = np.dot(rot_z, kp4)

        keypoint1 = np.array([((keypoint1[1] + ypos) * mm2px + 320) / 640, ((keypoint1[0] + xpos) * mm2px + 6) / 480, 1])
        keypoint2 = np.array([((keypoint2[1] + ypos) * mm2px + 320) / 640, ((keypoint2[0] + xpos) * mm2px + 6) / 480, 1])
        keypoint3 = np.array([((keypoint3[1] + ypos) * mm2px + 320) / 640, ((keypoint3[0] + xpos) * mm2px + 6) / 480, 1])
        keypoint4 = np.array([((keypoint4[1] + ypos) * mm2px + 320) / 640, ((keypoint4[0] + xpos) * mm2px + 6) / 480, 1])
        keypoints = np.concatenate((keypoint1, keypoint2, keypoint3, keypoint4), axis=0).reshape(-1, 3)

        return keypoints

    def data_preprocess(self, xy, lw, ori):

        mm2px = 530 / 0.34  # (1558)
        total_num = len(xy)
        num_item = 15
        label = []
        for j in range(total_num):
            # print(real_world_data[j])
            print('this is index if legos', j)
            xpos1, ypos1 = xy[j, 0], xy[j, 1]
            l, w = lw[j, 0], lw[j, 1]
            yawori = ori[j]

            # ensure the yolo sequence!
            label_y = (xpos1 * mm2px + 6) / 480
            label_x = (ypos1 * mm2px + 320) / 640
            length = l * 3
            width = w * 3
            # ensure the yolo sequence!
            keypoints = self.find_keypoints(xpos1, ypos1, l, w, yawori, mm2px)
            keypoints_order = np.lexsort((keypoints[:, 0], keypoints[:, 1]))
            keypoints = keypoints[keypoints_order]

            element = np.concatenate(([0], [label_x, label_y], [length, width], keypoints.reshape(-1)))
            label.append(element)

        label = np.asarray(label)

        return label

    def plot_and_transform(self, im, box, label='', color=(0, 0, 0), txt_color=(255, 255, 255), index=None, scaled_xylw=None, keypoints=None, use_xylw=True, truth_flag=None):
        # Add one xyxy box to image with label

        ############### zzz plot parameters ###############
        zzz_lw = 1
        tf = 1 # font thickness
        mm2px = 530 / 0.34
        # x_mm_center = scaled_xylw[1] * 0.3
        # y_mm_center = scaled_xylw[0] * 0.4 - 0.2
        # x_px_center = x_mm_center * mm2px + 6
        # y_px_center = y_mm_center * mm2px + 320
        x_px_center = scaled_xylw[1] * 480
        y_px_center = scaled_xylw[0] * 640

        # this is the knolling sequence, not opencv!!!!
        keypoints_x = ((keypoints[:, 1] * 480 - 6) / mm2px).reshape(-1, 1)
        keypoints_y = ((keypoints[:, 0] * 640 - 320) / mm2px).reshape(-1, 1)
        keypoints_mm = np.concatenate((keypoints_x, keypoints_y), axis=1)
        keypoints_center = np.average(keypoints_mm, axis=0)

        length = max(np.linalg.norm(keypoints_mm[0] - keypoints_mm[-1]),
                     np.linalg.norm(keypoints_mm[1] - keypoints_mm[2]))
        width = min(np.linalg.norm(keypoints_mm[0] - keypoints_mm[-1]),
                    np.linalg.norm(keypoints_mm[1] - keypoints_mm[2]))
        c1 = np.array([length / (2), width / (2)])
        c2 = np.array([length / (2), -width / (2)])
        c3 = np.array([-length / (2), width / (2)])
        c4 = np.array([-length / (2), -width / (2)])
        if use_xylw == True:
            # length = scaled_xylw[2] / 3
            # width = scaled_xylw[3] / 3
            # c1 = np.array([length / (2), width / (2)])
            # c2 = np.array([length / (2), -width / (2)])
            # c3 = np.array([-length / (2), width / (2)])
            # c4 = np.array([-length / (2), -width / (2)])
            box_center = np.array([scaled_xylw[0], scaled_xylw[1]])
        else:
            box_center = keypoints_center

        all_distance = np.linalg.norm((keypoints_mm - keypoints_center), axis=1)
        k = 2
        max_index = all_distance.argsort()[-k:]
        lucky_keypoint_index = np.argmax([keypoints_mm[max_index[0], 1], keypoints_mm[max_index[1], 1]])
        lucky_keypoint = keypoints_mm[max_index[lucky_keypoint_index]]
        # print('the ori keypoint is ', keypoints_mm[max_index[lucky_keypoint_index]])
        my_ori = np.arctan2(lucky_keypoint[1] - keypoints_center[1], lucky_keypoint[0] - keypoints_center[0])
        # In order to grasp, this ori is based on the longest side of the box, not the label ori!

        if length < width:
            if my_ori > np.pi / 2:
                my_ori_plot = my_ori - np.pi / 2
            else:
                my_ori_plot = my_ori + np.pi / 2
        else:
            my_ori_plot = my_ori

        rot_z = [[np.cos(my_ori_plot), -np.sin(my_ori_plot)],
                 [np.sin(my_ori_plot), np.cos(my_ori_plot)]]
        corn1 = (np.dot(rot_z, c1)) * mm2px
        corn2 = (np.dot(rot_z, c2)) * mm2px
        corn3 = (np.dot(rot_z, c3)) * mm2px
        corn4 = (np.dot(rot_z, c4)) * mm2px

        corn1 = [corn1[0] + x_px_center, corn1[1] + y_px_center]
        corn2 = [corn2[0] + x_px_center, corn2[1] + y_px_center]
        corn3 = [corn3[0] + x_px_center, corn3[1] + y_px_center]
        corn4 = [corn4[0] + x_px_center, corn4[1] + y_px_center]
        ############### zzz plot parameters ###############


        ############### zzz plot the box ###############
        if isinstance(box, torch.Tensor):
            box = box.cpu().detach().numpy()
        # print(box)
        p1 = np.array([int(box[0] * 640), int(box[1] * 480)])
        # print('this is p1 and p2', p1, p2)

        # cv2.rectangle(self.im, p1, p2, color, thickness=zzz_lw, lineType=cv2.LINE_AA)
        im = cv2.line(im, (int(corn1[1]), int(corn1[0])), (int(corn2[1]), int(corn2[0])), color, 1)
        im = cv2.line(im, (int(corn2[1]), int(corn2[0])), (int(corn4[1]), int(corn4[0])), color, 1)
        im = cv2.line(im, (int(corn4[1]), int(corn4[0])), (int(corn3[1]), int(corn3[0])), color, 1)
        im = cv2.line(im, (int(corn3[1]), int(corn3[0])), (int(corn1[1]), int(corn1[0])), color, 1)
        plot_x = np.copy((scaled_xylw[1] * 480 - 6) / mm2px)
        plot_y = np.copy((scaled_xylw[0] * 640 - 320) / mm2px)
        plot_l = np.copy(length)
        plot_w = np.copy(width)
        label1 = 'index: %d, x: %.4f, y: %.4f' % (index, plot_x, plot_y)
        label2 = 'l: %.4f, w: %.4f, ori: %.4f' % (plot_l, plot_w, my_ori)
        if label:
            w, h = cv2.getTextSize(label, 0, fontScale=zzz_lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h >= 3
            # cv2.rectangle(self.im, p1, p2, color, 0, cv2.LINE_AA)  # filled
            if truth_flag == True:
                txt_color = (0, 0, 255)
                im = cv2.putText(im, label1, (p1[0] - 50, p1[1] - 32 if outside else p1[1] + h + 2),
                                 0, zzz_lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
                im = cv2.putText(im, label2, (p1[0] - 50, p1[1] - 22 if outside else p1[1] + h + 12),
                                 0, zzz_lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
            else:
                im = cv2.putText(im, label1, (p1[0] - 50, p1[1] + 22 if outside else p1[1] + h + 2),
                                 0, zzz_lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
                im = cv2.putText(im, label2, (p1[0] - 50, p1[1] + 32 if outside else p1[1] + h + 12),
                                 0, zzz_lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
            # im = cv2.putText(im, label1, (c1[0] - 70, c1[1] - 35), 0, tl / 3, color, thickness=tf, lineType=cv2.LINE_AA)
        ############### zzz plot the box ###############

        ############### zzz plot the keypoints ###############
        shape = (640, 640)
        radius = 1
        for i, k in enumerate(keypoints):
            if truth_flag == False:
                if i == 0:
                    color_k = (255, 0, 0)
                else:
                    color_k = (0, 0, 0)
            elif truth_flag == True:
                if i == 0:
                    color_k = (0, 0, 255)
                elif i == 3:
                    color_k = (255, 255, 0)
            x_coord, y_coord = k[0] * 640, k[1] * 480
            # if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
            #     if len(k) == 3:
            #         conf = k[2]
            #         if conf < 0.5:
            #             continue
            im = cv2.circle(im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)
        ############### zzz plot the keypoints ###############

        result = np.concatenate((box_center, [round(length, 4)], [round(width, 4)], [my_ori]))

        return im, result

    def adjust_img(self, img):

        cv2.namedWindow('zzz', 0)
        cv2.imshow('zzz', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        alpha = 1
        beta = 20
        new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        cv2.namedWindow('zzz', 0)
        cv2.imshow('zzz', new_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_img

    def yolov8_predict(self, cfg=DEFAULT_CFG, use_python=False, img_path=None, img=None, real_flag=None, target=None):
        model = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_bot/ultralytics/yolo_runs/train_standard_521_tuning/weights/best.pt'
        model = '/home/zhizhuo/Creative Machines Lab/knolling_bot/ultralytics/yolo_runs/train_standard_521_tuning/weights/best.pt'

        origin_point = np.array([0, -0.20])
        if real_flag == True:

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

            mean_floor = (160, 160, 160)
            origin_point = np.array([0, -0.20])

            total_pred_result = []
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())
                color_colormap_dim = color_image.shape
                resized_color_image = color_image

                # img = adjust_img(img)

                cv2.imwrite(img_path + '.png', resized_color_image)
                img_path_input = img_path + '.png'
                args = dict(model=model, source=img_path_input, conf=0.5, iou=0.2)
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
                # pred_order = np.lexsort((pred[:, 2], pred[:, 1]))

                ######## order based on distance to draw it on the image!!!
                mm2px = 530 / 0.34
                x_px_center = pred_xylws[:, 1] * 480
                y_px_center = pred_xylws[:, 0] * 640
                mm_center = np.concatenate((((x_px_center - 6) / mm2px).reshape(-1, 1), ((y_px_center - 320) / mm2px).reshape(-1, 1)), axis=1)
                distance = np.linalg.norm(mm_center - origin_point, axis=1)
                pred_order = np.argsort(distance)

                # pred_test = np.copy(pred[pred_order])
                # for i in range(len(pred) - 1):
                #     if np.abs(pred_test[i, 1] - pred_test[i + 1, 1]) < 0.01:
                #         if pred_test[i, 2] < pred_test[i + 1, 2]:
                #             # ground_truth_pose[order_ground_truth[i]], ground_truth_pose[order_ground_truth[i+1]] = ground_truth_pose[order_ground_truth[i+1]], ground_truth_pose[order_ground_truth[i]]
                #             pred_order[i], pred_order[i + 1] = pred_order[i + 1], pred_order[i]
                #             print('pred change the order!')
                #         else:
                #             pass
                # # print('this is the pred order', pred_order)

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
                    origin_img, result = self.plot_and_transform(im=origin_img, box=pred_xylw, label='0:, predic', color=(0, 0, 0),
                                                            txt_color=(255, 255, 255), index=j,
                                                            scaled_xylw=pred_xylw, keypoints=pred_keypoint, use_xylw=use_xylw,
                                                            truth_flag=False)
                    pred_result.append(result)
                    # print('this is j', j)
                pred_result = np.asarray(pred_result)

                # distance = np.linalg.norm(pred_result[:, :2] - origin_point, axis=1)
                # order = np.argsort(distance)
                order = change_sequence(pred_result[:, :2])
                print('this is order', order)
                pred_result = pred_result[order]
                ############ fill the rest of result with zeros if the number of result is less than 10 #############
                if len(pred_result) != 0:
                    pred_result = np.concatenate((pred_result, np.zeros((1, pred_result.shape[1]))), axis=0)
                print('this is result\n', pred_result)
                ############ fill the rest of result with zeros if the number of result is less than 10 #############
                total_pred_result.append(pred_result)

                cv2.namedWindow('zzz', 0)
                cv2.resizeWindow('zzz', 1280, 960)
                cv2.imshow('zzz', origin_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    img_path_output = img_path + '_pred.png'
                    cv2.imwrite(img_path_output, origin_img)
                    break
            total_pred_result = np.asarray(total_pred_result)
            # pred_result = np.concatenate((np.mean(total_pred_result, axis=0), np.max(total_pred_result[:, :, 2:4], axis=0),
            #                               np.mean(total_pred_result[:, :, 4], axis=0).reshape(-1, 1)), axis=1)
            # pred_result = np.mean(total_pred_result, axis=0)
            # print(pred_result)

        else:
            # img = adjust_img(img)

            cv2.imwrite(img_path + '.png', img)
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

            use_xylw = False # use lw or keypoints to export length and width

            one_img = images[0]

            pred_result = []
            pred_xylws = one_img.boxes.xywhn.cpu().detach().numpy()
            pred_keypoints = one_img.keypoints.cpu().detach().numpy()
            pred_keypoints[:, :, :2] = pred_keypoints[:, :, :2] / np.array([640, 480])
            pred_keypoints = pred_keypoints.reshape(len(pred_xylws), -1)
            pred = np.concatenate((np.zeros((len(pred_xylws), 1)), pred_xylws, pred_keypoints), axis=1)

            ######## order based on distance to draw it on the image!!!
            mm2px = 530 / 0.34
            x_px_center = pred_xylws[:, 1] * 480
            y_px_center = pred_xylws[:, 0] * 640
            mm_center = np.concatenate(
                (((x_px_center - 6) / mm2px).reshape(-1, 1), ((y_px_center - 320) / mm2px).reshape(-1, 1)), axis=1)
            distance = np.linalg.norm(mm_center - origin_point, axis=1)
            pred_order = np.argsort(distance)

            pred = pred[pred_order]
            pred_xylws = pred_xylws[pred_order]
            pred_keypoints = pred_keypoints[pred_order]

            # pred_test = np.copy(pred[pred_order])
            # for i in range(len(pred) - 1):
            #     if np.abs(pred_test[i, 1] - pred_test[i + 1, 1]) < 0.01:
            #         if pred_test[i, 2] < pred_test[i + 1, 2]:
            #             # ground_truth_pose[order_ground_truth[i]], ground_truth_pose[order_ground_truth[i+1]] = ground_truth_pose[order_ground_truth[i+1]], ground_truth_pose[order_ground_truth[i]]
            #             pred_order[i], pred_order[i + 1] = pred_order[i + 1], pred_order[i]
            #             print('pred change the order!')
            #         else:
            #             pass
            print('this is the pred order', pred_order)

            for j in range(len(pred_xylws)):

                pred_keypoint = pred_keypoints[j].reshape(-1, 3)
                pred_xylw = pred_xylws[j]

                # plot pred
                print('this is pred xylw', pred_xylw)
                # print('this is pred cos sin', pred_cos_sin)
                origin_img, result = self.plot_and_transform(im=origin_img, box=pred_xylw, label='0:, predic', color=(0, 0, 0), txt_color=(255, 255, 255), index=j,
                                                scaled_xylw=pred_xylw, keypoints=pred_keypoint, use_xylw=use_xylw, truth_flag=False)
                pred_result.append(result)
                # print('this is j', j)

                if real_flag == False:
                    tar_xylw = np.copy(target[j, 1:5])
                    tar_keypoints = np.copy((target[j, 5:]).reshape(-1, 3)[:, :2])
                    # tar_keypoints = (target[j, 5:])
                    # tar_keypoints[:, 0] *= 640
                    # tar_keypoints[:, 1] *= 480
                    tar_label = '0: "target"'

                    # plot target
                    print('this is tar xylw', tar_xylw)
                    # print('this is tar cos sin', tar_keypoints)
                    origin_img, _ = self.plot_and_transform(im=origin_img, box=tar_xylw, label='0: target', color=(255, 255, 0), txt_color=(255, 255, 255), index=j,
                                                    scaled_xylw=tar_xylw, keypoints=tar_keypoints, use_xylw=use_xylw, truth_flag=True)

            cv2.namedWindow('zzz', 0)
            cv2.imshow('zzz', origin_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            img_path_output = img_path + '_pred.png'
            cv2.imwrite(img_path_output, origin_img)

        if real_flag == False:
            print('this is length of pred\n', pred[:, 1:5])
            print('this is length of target\n', target[:, 1:5])
            # loss_mean = np.mean((target - pred))
            # loss_std = np.std((target - pred), dtype=np.float64)
            loss = np.mean((target - pred) ** 2)
            # print('this is mean error', loss_mean)
            # print('this is std error', loss_std)
            pred_result = np.asarray(pred_result)
            return pred_result, loss

        else:
            return pred_result

class Sort_objects():

    def __init__(self, evaluations):
        self.evaluations = evaluations
        self.error_rate = 0.05

    def get_data_virtual(self, area_num, ratio_num, boxes_num):

        xyz_list = []
        pos_list = []
        ori_list = []
        length_range = np.round(np.random.uniform(0.016, 0.048, size=(boxes_num, 1)), decimals=3)
        width_range = np.round(np.random.uniform(0.016, np.minimum(length_range, 0.036), size=(boxes_num, 1)), decimals=3)
        height_range = np.round(np.random.uniform(0.012, 0.012, size=(boxes_num, 1)), decimals=3)

        for i in range(boxes_num):
            xyz_list.append(np.concatenate((length_range[i], width_range[i], height_range[i])))
        xyz_list = np.asarray(xyz_list, dtype=np.float32)
        print(xyz_list)

        return self.judge(xyz_list, pos_list, ori_list, area_num, ratio_num, boxes_num)

    def get_data_real(self, area_num, ratio_num, yolo_model):

        img_path = './Test_images/image_real'
        # structure of results: x, y, length, width, ori
        results = yolo_model.yolov8_predict(img_path=img_path, real_flag=True, target=None)

        item_pos = np.concatenate((results[:, :2], np.zeros(len(results)).reshape(-1, 1)), axis=1)
        item_lw = np.concatenate((results[:, 2:4], (np.ones(len(results)) * 0.012).reshape(-1, 1)), axis=1)
        item_ori = np.concatenate((np.zeros((len(results), 2)), results[:, 4].reshape(-1, 1)), axis=1)

        # ##################### generate customize boxes based on the result of yolo ######################
        # temp_box = URDF.load('./urdf/box_generator/template.urdf')
        # for i in range(len(results)):
        #     temp_box.links[0].collisions[0].origin[2, 3] = 0
        #     length = item_lw[i, 0]
        #     width = item_lw[i, 1]
        #     height = 0.012
        #     temp_box.links[0].visuals[0].geometry.box.size = [length, width, height]
        #     temp_box.links[0].collisions[0].geometry.box.size = [length, width, height]
        #     temp_box.save('./urdf/knolling_box/knolling_box_%d.urdf' % i)
        # ##################### generate customize boxes based on the result of yolo ######################

        category_num = int(area_num * ratio_num + 1)
        s = item_lw[:, 0] * item_lw[:, 1]
        s_min, s_max = np.min(s), np.max(s)
        s_range = np.linspace(s_max, s_min, int(area_num + 1))
        lw_ratio = item_lw[:, 0] / item_lw[:, 1]
        ratio_min, ratio_max = np.min(lw_ratio), np.max(lw_ratio)
        ratio_range = np.linspace(ratio_max, ratio_min, int(ratio_num + 1))

        # ! initiate the number of items
        all_index = []
        new_item_xyz = []
        new_item_pos = []
        new_item_ori = []
        transform_flag = []
        rest_index = np.arange(len(item_lw))
        index = 0

        for i in range(area_num):
            for j in range(ratio_num):
                kind_index = []
                for m in range(len(item_lw)):
                    if m not in rest_index:
                        continue
                    else:
                        if s_range[i] >= s[m] >= s_range[i + 1]:
                            if ratio_range[j] >= lw_ratio[m] >= ratio_range[j + 1]:
                                transform_flag.append(0)
                                # print(f'boxes{m} matches in area{i}, ratio{j}!')
                                kind_index.append(index)
                                new_item_xyz.append(item_lw[m])
                                new_item_pos.append(item_pos[m])
                                new_item_ori.append(item_ori[m])
                                index += 1
                                rest_index = np.delete(rest_index, np.where(rest_index == m))
                if len(kind_index) != 0:
                    all_index.append(kind_index)

        new_item_xyz = np.asarray(new_item_xyz).reshape(-1, 3)
        new_item_pos = np.asarray(new_item_pos)
        new_item_ori = np.asarray(new_item_ori)
        transform_flag = np.asarray(transform_flag)
        if len(rest_index) != 0:
            # we should implement the rest of boxes!
            rest_xyz = item_lw[rest_index]
            new_item_xyz = np.concatenate((new_item_xyz, rest_xyz), axis=0)
            all_index.append(list(np.arange(index, len(item_lw))))
            transform_flag = np.append(transform_flag, np.zeros(len(item_lw) - index))

        # the sequence of them are based on area and ratio!
        return new_item_xyz, new_item_pos, new_item_ori, all_index, transform_flag

    def judge(self, item_xyz, item_pos, item_ori, area_num, ratio_num, boxes_num):

        category_num = int(area_num * ratio_num + 1)
        s = item_xyz[:, 0] * item_xyz[:, 1]
        s_min, s_max = np.min(s), np.max(s)
        s_range = np.linspace(s_max, s_min, int(area_num + 1))
        lw_ratio = item_xyz[:, 0] / item_xyz[:, 1]
        ratio_min, ratio_max = np.min(lw_ratio), np.max(lw_ratio)
        ratio_range = np.linspace(ratio_max, ratio_min, int(ratio_num + 1))
        ratio_range_high = np.linspace(ratio_max, 1, int(ratio_num + 1))
        ratio_range_low = np.linspace(1 / ratio_max, 1, int(ratio_num + 1))

        # ! initiate the number of items
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

class Env:

    def __init__(self, is_render=True):

        self.kImageSize = {'width': 480, 'height': 480}
        self.urdf_path = './urdf/'
        self.pybullet_path = pd.getDataPath()
        self.is_render = is_render
        if self.is_render:
            p.connect(p.GUI, options="--width=1280 --height=720")
            # p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

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

        self.lateral_friction = 1
        self.spinning_friction = 1
        self.rolling_friction = 0

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
            p.configureDebugVisualizer(lightPosition=[random.randint(1,3), random.randint(1,2), 5])
        else:
            p.configureDebugVisualizer(lightPosition=[random.randint(1,3), random.randint(-2, -1), 5])
        p.configureDebugVisualizer(lightPosition=[random.randint(1, 3), random.randint(1, 2), 5],
                                   shadowMapResolution=8192, shadowMapIntensity=np.random.randint(5, 8) / 10)
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                     cameraYaw=45,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[0.1, 0, 0])
        p.setAdditionalSearchPath(pd.getDataPath())

    def get_env_parameters(self, manual_knolling_parameters=None, evaluations=None,
                           real_operate=False, obs_order='1', use_knolling_model=None):

        self.area_num = manual_knolling_parameters['area_num']
        self.ratio_num = manual_knolling_parameters['ratio_num']
        self.boxes_num = manual_knolling_parameters['boxes_num']
        self.random_offset = manual_knolling_parameters['random_offset']
        self.total_offset = manual_knolling_parameters['total_offset']
        self.manual_knolling_parameters = manual_knolling_parameters
        self.use_knolling_model = use_knolling_model
        self.yolo_model = Yolo_predict()

        self.real_operate = real_operate
        self.obs_order = obs_order
        self.evaluations = evaluations

    def get_obs(self, order):

        def get_images():

            (width, length, my_im, _, _) = p.getCameraImage(width=640,
                                                            height=480,
                                                            viewMatrix=self.view_matrix,
                                                            projectionMatrix=self.projection_matrix,
                                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)
            my_im = my_im[:, :, :3]
            temp = np.copy(my_im[:, :, 0])  # change rgb image to bgr for opencv to save
            my_im[:, :, 0] = my_im[:, :, 2]
            my_im[:, :, 2] = temp
            return my_im

        def get_sim_image_obs():

            (width, length, image, _, _) = p.getCameraImage(width=640,
                                                            height=480,
                                                            viewMatrix=self.view_matrix,
                                                            projectionMatrix=self.projection_matrix,
                                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

            img = image[:, :,:3]
            img_path = './image_sim/528_sim'
            # cv2.imwrite(img_path + '.png', img)

            ############### order the ground truth depend on x, y in the world coordinate system ###############
            new_xyz_list = self.lwh_list

            # collect the cur pos and cur ori in pybullet as the ground truth
            new_pos_before, new_ori_before = [], []
            for i in range(len(self.boxes_index)):
                new_pos_before.append(p.getBasePositionAndOrientation(self.boxes_index[i])[0][:2])
                new_ori_before.append(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.boxes_index[i])[1])[2])
            new_pos_before = np.asarray(new_pos_before)
            new_ori_before = np.asarray(new_ori_before)

            ground_truth_pose = self.yolo_model.data_preprocess(new_pos_before, new_xyz_list[:, :2], new_ori_before)
            # # structure of ground_truth_pose: yolo-pose label!!!!!!!
            # order_ground_truth = np.lexsort((ground_truth_pose[:, 2], ground_truth_pose[:, 1]))
            #
            # ground_truth_pose_test = np.copy(ground_truth_pose[order_ground_truth, :])
            # for i in range(len(order_ground_truth) - 1):
            #     if np.abs(ground_truth_pose_test[i, 1] - ground_truth_pose_test[i+1, 1]) < 0.01:
            #         if ground_truth_pose_test[i, 2] < ground_truth_pose_test[i+1, 2]:
            #             # ground_truth_pose[order_ground_truth[i]], ground_truth_pose[order_ground_truth[i+1]] = ground_truth_pose[order_ground_truth[i+1]], ground_truth_pose[order_ground_truth[i]]
            #             order_ground_truth[i], order_ground_truth[i+1] = order_ground_truth[i+1], order_ground_truth[i]
            #             print('truth change the order!')
            #         else:
            #             pass
            # print('this is the ground truth order', order_ground_truth)
            # # print('this is the ground truth before changing the order\n', ground_truth_pose)
            # new_xyz_list = new_xyz_list[order_ground_truth, :]
            # ground_truth_pose = ground_truth_pose[order_ground_truth, :]
            # ############### order the ground truth depend on x, y in the world coordinate system ###############

            ################### the results of object detection has changed the order!!!! ####################
            # structure of results: x, y, length, width, ori
            results, env_loss = self.yolo_model.yolov8_predict(img_path=img_path, img=img, real_flag=self.real_operate, target=ground_truth_pose)
            print('this is the result of yolo-pose\n', results)
            ################### the results of object detection has changed the order!!!! ####################

            z = 0
            roll = 0
            pitch = 0
            # # width_index = []
            # print('this is self.xyz、', self.lwh_list)
            # for i in range(len(self.lwh_list)):
            #     for j in range(len(results)):
            #         if (np.abs(self.lwh_list[i, 0] - results[j, 2]) < 0.002 and np.abs(self.lwh_list[i, 1] - results[j, 3]) < 0.002) or \
            #                 (np.abs(self.lwh_list[i, 1] - results[j, 2]) < 0.002 and np.abs(self.lwh_list[i, 0] - results[j, 3]) < 0.002):
            #             if j not in index:
            #                 index.append(j)
            #
            # manipulator_before = []
            # for i in index:
            #     manipulator_before.append([results[i][0], results[i][1], z, roll, pitch, results[i][4]])
            # manipulator_before = np.asarray(manipulator_before)
            manipulator_before = np.concatenate((results[:, :2], np.zeros((len(self.lwh_list), 3)), results[:, 4].reshape(-1, 1)), axis=1)
            manipulator_before[:, 2] += 0.006
            new_xyz_list = self.lwh_list
            print('this is manipulator before after the detection \n', manipulator_before)

            if self.obs_order == 'sim_image_obj_evaluate':
                return manipulator_before, new_xyz_list, env_loss
            else:
                return manipulator_before, new_xyz_list

        def get_lego_obs():

            # sequence: pos before, ori before
            new_pos_before, new_ori_before = [], []
            for i in range(self.boxes_num):
                new_pos_before.append(p.getBasePositionAndOrientation(i)[0])
                new_ori_before.append(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(i)[1]))

            manipulator_before = np.concatenate((new_pos_before, new_ori_before), axis=1)
            new_xyz_list = self.lwh_list

            return manipulator_before, new_xyz_list

        def get_real_image_obs():

            # # temp useless because of knolling demo
            # img_path = 'Test_images/image_real'
            # # structure: x,y,length,width,yaw
            # results = yolov8_predict(img_path=img_path, real_flag=self.real_operate, target=None)
            # print('this is the result of yolo-pose\n', results)
            #
            # z = 0
            # roll = 0
            # pitch = 0
            # index = []
            # print('this is self.xyz\n', self.xyz_list)
            # for i in range(len(self.xyz_list)):
            #     for j in range(len(results)):
            #         if (np.abs(self.xyz_list[i, 0] - results[j, 2]) <= 0.002 and np.abs(
            #                 self.xyz_list[i, 1] - results[j, 3]) <= 0.002) or \
            #                 (np.abs(self.xyz_list[i, 1] - results[j, 2]) <= 0.002 and np.abs(
            #                     self.xyz_list[i, 0] - results[j, 3]) <= 0.002):
            #             if j not in index:
            #                 print(f"find first xyz{i} in second xyz{j}")
            #                 index.append(j)
            #                 break
            #             else:
            #                 pass
            #
            # manipulator_before = []
            # for i in index:
            #     manipulator_before.append([results[i][0], results[i][1], z, roll, pitch, results[i][4]])
            # # for i in range(len(self.xyz_list)):
            # #     manipulator_before.append([self.pos_before[i][0], self.pos_before[i][1], z, roll, pitch, self.ori_before[i][2]])

            manipulator_before = np.concatenate((self.pos_before, self.ori_before), axis=1)
            manipulator_before = np.asarray(manipulator_before)
            new_xyz_list = self.lwh_list
            print('this is manipulator before after the detection \n', manipulator_before)

            return manipulator_before, new_xyz_list

        if order == 'sim_obj':
            manipulator_before, new_xyz_list = get_lego_obs()
            return manipulator_before, new_xyz_list
        elif order == 'sim_image_obj':
            manipulator_before, new_xyz_list = get_sim_image_obs()
            return manipulator_before, new_xyz_list
        elif order == 'images':
            image = get_images()
            return image
        elif order == 'real_image_obj':
            manipulator_before, new_xyz_list = get_real_image_obs()
            return manipulator_before, new_xyz_list
        elif order == 'sim_image_obj_evaluate':
            manipulator_before, new_xyz_list, error = get_sim_image_obs()
            return manipulator_before, new_xyz_list, error

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

        baseid = p.loadURDF(self.urdf_path + "plane_zzz.urdf", basePosition=[0, -0.2, 0], useFixedBase=1,
                            flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        self.arm_id = p.loadURDF(os.path.join(self.urdf_path, "robot_arm928/robot_arm1.urdf"),
                                 basePosition=[-0.08, 0, 0.02], useFixedBase=True,
                                 flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        textureId = p.loadTexture(self.urdf_path + "img_1.png")
        p.changeDynamics(baseid, -1, lateralFriction=1, spinningFriction=1, rollingFriction=0.002, linearDamping=0.5, angularDamping=0.5)
        p.changeDynamics(self.arm_id, 7, lateralFriction=1, spinningFriction=1, rollingFriction=0, linearDamping=1, angularDamping=1, jointDamping=1)
        p.changeDynamics(self.arm_id, 8, lateralFriction=1, spinningFriction=1, rollingFriction=0, linearDamping=1, angularDamping=1, jointDamping=1)
        # p.changeVisualShape(baseid, -1, textureUniqueId=textureId,rgbaColor=[np.random.uniform(0.9,1), np.random.uniform(0.9,1),np.random.uniform(0.9,1), 1])
        p.changeVisualShape(baseid, -1, textureUniqueId=textureId)

        # set the initial pos of the arm
        ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=[0, 0, 0.06],
                                                  maxNumIterations=200,
                                                  targetOrientation=p.getQuaternionFromEuler([0, math.pi / 2, 0]))
        for motor_index in range(self.num_motor):
            p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                    targetPosition=ik_angles0[motor_index], maxVelocity=7)
        for i in range(100):
            p.stepSimulation()

        # get the standard xyz and corresponding index from files in the computer
        items_sort = Sort_objects(self.evaluations)
        if self.real_operate == False:

            self.boxes_index = []
            for i in range(self.boxes_num):
                self.boxes_index.append(p.loadURDF(self.save_urdf_path_one_img + 'box_%d.urdf' % (i),
                                       basePosition=self.pos_before[i],
                                       baseOrientation=p.getQuaternionFromEuler(self.ori_before[i]), useFixedBase=False,
                                       flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
                r = np.random.uniform(0, 0.9)
                g = np.random.uniform(0, 0.9)
                b = np.random.uniform(0, 0.9)
                p.changeVisualShape(self.boxes_index[i], -1, rgbaColor=(r, g, b, 1))

            print('this is random ori when reset the environment\n', self.ori_before)
            print('this is random pos when reset the environment\n', self.pos_before)
        else:
            # these data has defined in function change_config, we don't need to define them twice!!!
            # self.xyz_list, pos_before, ori_before, self.all_index, self.kind = items_sort.get_data_real()
            num_lego = 0
            sim_pos = np.copy(self.pos_before)
            sim_pos[:, :2] += 0.006
            # for i in range(len(self.xyz_list)):
            #     self.lego_idx.append(
            #         p.loadURDF(self.save_urdf_path_one_img + 'box_%d.urdf' % (i),
            #                    basePosition=sim_pos[i],
            #                    baseOrientation=p.getQuaternionFromEuler(self.ori_before[i]), useFixedBase=False,
            #                    flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
            #     r = np.random.uniform(0, 0.9)
            #     g = np.random.uniform(0, 0.9)
            #     b = np.random.uniform(0, 0.9)
            #     p.changeVisualShape(self.lego_idx[num_lego], -1, rgbaColor=(r, g, b, 1))
            #     num_lego += 1

            for i in range(30):
                p.stepSimulation()

        data_before = np.concatenate((self.pos_before[:, :2], self.lwh_list[:, :2], self.ori_before[:, 2].reshape(-1, 1)), axis=1)
        # np.savetxt('./learning_data_demo/cfg_4_519/labels_before/label_8_%d.txt' % self.evaluations, data_before, fmt='%.03f')

        return self.get_obs('images'), self.arm_id
        # return self.pos_before, self.ori_before, self.xyz_list

    def manual_knolling(self):  # this is main function!!!!!!!!!

        p.resetSimulation()
        p.setGravity(0, 0, -10)

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

        baseid = p.loadURDF(self.urdf_path + "plane_zzz.urdf", basePosition=[0, -0.2, 0], useFixedBase=1,
                            flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        # self.arm_id = p.loadURDF(self.urdf_path + "robot_arm928/robot_arm1.urdf",
        #                          basePosition=[-0.08, 0, 0.02], useFixedBase=True,
        #                          flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        textureId = p.loadTexture(self.urdf_path + "img_1.png")
        p.changeDynamics(baseid, -1, lateralFriction=self.lateral_friction, frictionAnchor=True)
        # p.changeDynamics(self.arm_id, 7, lateralFriction=self.lateral_friction, frictionAnchor=True)
        # p.changeDynamics(self.arm_id, 8, lateralFriction=self.lateral_friction, frictionAnchor=True)
        p.changeVisualShape(baseid, -1, textureUniqueId=textureId)

        # get the standard xyz and corresponding index from files in the computer
        items_sort = Sort_objects(self.evaluations)
        if self.real_operate == False:
            self.lwh_list, _, _, self.all_index, self.transform_flag = items_sort.get_data_virtual(self.area_num, self.ratio_num, self.boxes_num)
        else:
            self.lwh_list, self.pos_before, self.ori_before, self.all_index, self.transform_flag = items_sort.get_data_real(self.area_num, self.ratio_num, self.yolo_model)

        print(f'this is standard trim xyz list\n {self.lwh_list}')
        print(f'this is standard trim index list\n {self.all_index}')

        if self.use_knolling_model == True:
            ######################## knolling demo ###############################

            ################## change order based on distance between boxes and upper left corner ##################
            order = change_sequence(self.pos_before)
            self.pos_before = self.pos_before[order]
            self.ori_before = self.ori_before[order]
            self.lwh_list = self.lwh_list[order]
            ################## change order based on distance between boxes and upper left corner ##################

            ################## input the demo data ##################
            knolling_demo_data = np.loadtxt('./num_10_after_demo_3.txt')[0].reshape(-1, 5)
            # with open('real_after_pred.txt', 'r') as file:
            #     data = file.read().replace(',', ' ')
            #     data = list(data.split())
            #     temp_data = np.array([float(d) for d in data]).reshape(-1, 5)[:10]
            #     knolling_demo_data = temp_data
            ################## input the demo data ##################

            index = []
            after_knolling = []
            after_knolling = np.asarray(after_knolling)

            self.pos_after = np.concatenate((knolling_demo_data[:, :2], np.zeros(len(knolling_demo_data)).reshape(-1, 1)), axis=1)
            self.ori_after = np.concatenate((np.zeros((len(knolling_demo_data), 2)), knolling_demo_data[:, 4].reshape(-1, 1)),
                                            axis=1)

            # self.items_pos_list = np.concatenate((after_knolling[:, :2], np.zeros(len(after_knolling)).reshape(-1, 1)), axis=1)
            # self.items_ori_list = np.concatenate((np.zeros((len(after_knolling), 2)), after_knolling[:, 4].reshape(-1, 1)), axis=1)
            # self.xyz_list = np.concatenate((after_knolling[:, 2:4], (np.ones(len(after_knolling)) * 0.012).reshape(-1, 1)), axis=1)
            ######################## knolling demo ###############################
        else:
            # determine the center of the tidy configuration
            calculate_reorder = configuration_zzz(self.lwh_list, self.all_index, self.transform_flag, self.manual_knolling_parameters)
            self.pos_after, self.ori_after = calculate_reorder.calculate_block()
            self.pos_after[:, 2] += 0.006
            # after this step the length and width of boxes in self.lwh_list my changed!!!!!!!!!!!

            ############## generate the random pos and ori for boxes after knolling ############
            restrict = np.max(self.lwh_list)
            gripper_height = 0.012
            last_pos = np.array([[0, 0, 1]])
            self.pos_before = []
            self.ori_before = []
            for i in range(self.boxes_num):
                rdm_pos = np.array([random.uniform(self.x_low_obs, self.x_high_obs),
                                    random.uniform(self.y_low_obs, self.y_high_obs), 0.006])
                ori = [0, 0, random.uniform(0, math.pi)]
                # ori = [0, 0, np.pi / 2]

                ################### after generate the neat configuration, we should recover the lw based on that in the urdf files!
                if self.transform_flag[i] == 1:
                    self.lwh_list[i, [0, 1]] = self.lwh_list[i, [1, 0]]
                    # if ori[2] > np.pi:
                    #     ori[2] -= np.pi / 2
                    # else:
                    #     ori[2] += np.pi / 2
                    # we dont' need to change the ori here, because the ori is definitely random
                    # the real ori provided to arm is genereatd by yolo
                ################### after generate the neat configuration, we should recover the lw based on that in the urdf files!

                check_list = np.zeros(last_pos.shape[0])
                while 0 in check_list:
                    rdm_pos = [random.uniform(self.x_low_obs, self.x_high_obs),
                               random.uniform(self.y_low_obs, self.y_high_obs), 0.006]
                    for z in range(last_pos.shape[0]):
                        if np.linalg.norm(last_pos[z] - rdm_pos) < restrict + gripper_height:
                            check_list[z] = 0
                        else:
                            check_list[z] = 1
                self.pos_before.append(rdm_pos)
                self.ori_before.append(ori)
                last_pos = np.append(last_pos, [rdm_pos], axis=0)
            self.pos_before = np.asarray(self.pos_before)
            self.ori_before = np.asarray(self.ori_before)
            ############## generate the random pos and ori for boxes after knolling ############

            ################## change order based on distance between boxes and upper left corner ##################
            origin_point = np.array([0, -0.2])
            distance = np.linalg.norm(self.pos_before[:, :2] - origin_point, axis=1)
            order = np.argsort(distance)
            self.pos_before = self.pos_before[order]
            self.ori_before = self.ori_before[order]
            self.lwh_list = self.lwh_list[order]
            self.pos_after = self.pos_after[order]
            self.ori_after = self.ori_after[order]
            ################## change order based on distance between boxes and upper left corner ##################

            x_low = np.min(self.pos_after, axis=0)[0]
            x_high = np.max(self.pos_after, axis=0)[0]
            y_low = np.min(self.pos_after, axis=0)[1]
            y_high = np.max(self.pos_after, axis=0)[1]
            center = np.array([(x_low + x_high) / 2, (y_low + y_high) / 2, 0])
            x_length = abs(x_high - x_low)
            y_length = abs(y_high - y_low)
            print(x_low, x_high, y_low, y_high)
            if self.random_offset == True:
                self.total_offset = np.array([random.uniform(self.x_low_obs + x_length / 2, self.x_high_obs - x_length / 2),
                                              random.uniform(self.y_low_obs + y_length / 2, self.y_high_obs - y_length / 2), 0.0])
            else:
                pass
            self.pos_after = self.pos_after + self.total_offset

        # after generate the neat configuratiion, pay attention to the difference of urdf ori and manipulator after ori!
        items_ori_list_arm = np.copy(self.ori_after)
        # for i in range(len(self.transform_flag)):
        #     if self.transform_flag[i] == 1:
        #         print(f'ori changed in index{i}!')
        #         if self.ori_after[i, 2] > np.pi:
        #             items_ori_list_arm[i, 2] = self.ori_after[i, 2] - np.pi / 2
        #         else:
        #             items_ori_list_arm[i, 2] = self.ori_after[i, 2] + np.pi / 2
        for i in range(len(self.lwh_list)):
            if self.lwh_list[i, 0] <= self.lwh_list[i, 1]:
                items_ori_list_arm[i, 2] += np.pi / 2
        # after generate the neat configuratiion, pay attention to the difference of urdf ori and manipulator after ori!

        self.manipulator_after = np.concatenate((self.pos_after, items_ori_list_arm), axis=1)
        print('this is manipulation after\n', self.manipulator_after)

        temp_box = URDF.load('./urdf/box_generator/template.urdf')
        self.save_urdf_path_one_img = './urdf/knolling_box/'
        os.makedirs(self.save_urdf_path_one_img, exist_ok=True)
        for i in range(len(self.lwh_list)):
            temp_box.links[0].collisions[0].origin[2, 3] = 0
            length = self.lwh_list[i, 0]
            width = self.lwh_list[i, 1]
            height = self.lwh_list[i, 2]
            temp_box.links[0].visuals[0].geometry.box.size = [length, width, height]
            temp_box.links[0].collisions[0].geometry.box.size = [length, width, height]
            temp_box.links[0].visuals[0].material.color = [np.random.random(), np.random.random(),
                                                           np.random.random(), 1]
            temp_box.save(self.save_urdf_path_one_img + 'box_%d.urdf' % (i))

        self.boxes_index = []
        for i in range(len(self.lwh_list)):
            self.boxes_index.append(p.loadURDF(self.save_urdf_path_one_img + 'box_%d.urdf' % (i),
                                               basePosition=self.pos_after[i],
                                               baseOrientation=p.getQuaternionFromEuler(self.ori_after[i]), useFixedBase=False,
                                               flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))

            r = np.random.uniform(0, 0.9)
            g = np.random.uniform(0, 0.9)
            b = np.random.uniform(0, 0.9)
            p.changeVisualShape(self.boxes_index[i], -1, rgbaColor=(r, g, b, 1))
        for i in range(30):
            p.stepSimulation()

        # if the urdf is lego, all ori after knolling should be 0, not pi / 2
        self.ori_after[:, 2] = 0
        data_after = np.concatenate((self.pos_after[:, :2], self.lwh_list[:, :2], self.ori_after[:, 2].reshape(-1, 1)), axis=1)
        # np.savetxt('./real_world_data_demo/cfg_4_519/labels_after/label_8_%d.txt' % self.evaluations, data_after, fmt='%.03f')

        return self.get_obs('images'), self.manipulator_after