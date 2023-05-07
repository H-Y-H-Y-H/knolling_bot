# Ultralytics YOLO 🚀, GPL-3.0 license
import numpy as np

import sys
sys.path.append('/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_bot/ultralytics')
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

import torch
import cv2

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

def find_keypoints(xpos, ypos, l, w, ori, mm2px):

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

def data_preprocess(xy, lw, ori):

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
        keypoints = find_keypoints(xpos1, ypos1, l, w, yawori, mm2px)
        keypoints_order = np.lexsort((keypoints[:, 0], keypoints[:, 1]))
        keypoints = keypoints[keypoints_order]

        element = np.concatenate(([0], [label_x, label_y], [length, width], keypoints.reshape(-1)))
        label.append(element)

    label = np.asarray(label)

    return label

def plot_and_transform(im, box, label='', color=(0, 0, 0), txt_color=(255, 255, 255), index=None, scaled_xylw=None, keypoints=None, use_lw=True, truth_flag=None):
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
    if use_lw == True:
        length = scaled_xylw[2] / 3
        width = scaled_xylw[3] / 3
        c1 = np.array([length / (2), width / (2)])
        c2 = np.array([length / (2), -width / (2)])
        c3 = np.array([-length / (2), width / (2)])
        c4 = np.array([-length / (2), -width / (2)])
    else:
        length = max(np.linalg.norm(keypoints_mm[0] - keypoints_mm[-1]),
                   np.linalg.norm(keypoints_mm[1] - keypoints_mm[2]))
        width = min(np.linalg.norm(keypoints_mm[0] - keypoints_mm[-1]),
                   np.linalg.norm(keypoints_mm[1] - keypoints_mm[2]))
        c1 = np.array([length / (2), width / (2)])
        c2 = np.array([length / (2), -width / (2)])
        c3 = np.array([-length / (2), width / (2)])
        c4 = np.array([-length / (2), -width / (2)])

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
    label1 = 'index: %d, x: %.3f, y: %.3f' % (index, plot_x, plot_y)
    label2 = 'l: %.3f, w: %.3f, ori: %.3f' % (plot_l, plot_w, my_ori)
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

    result = np.concatenate((keypoints_center, [length], [width], [my_ori]))

    return im, result

def yolov8_predict(cfg=DEFAULT_CFG, use_python=False, img_path=None, data_path=None, model_path=None, real_flag=None, target=None):
    # data_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/datasets/'
    # model_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/YOLOv8/runs/pose/train_standard_1000/weights/best.pt'
    model = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_bot/ultralytics/yolo_runs/train_standard_507/weights/best.pt'
    # source_pth = data_path + img_path
    # source_pth = data_path + 'real_image_collect/'
    # source_pth = data_path + 'yolo_pose4keypoints/images/val/'
    img_path_input = img_path + '.png'
    args = dict(model=model, source=img_path_input, conf=0.2, iou=0.2)
    use_python = True
    if use_python:
        from ultralytics import YOLO
        images = YOLO(model)(**args)
    else:
        predictor = PosePredictor(overrides=args)
        predictor.predict_cli()
    device = 'cuda:0'



    origin_img = cv2.imread(img_path_input)
    # origin_img = cv2.imread(source_pth + 'img_%s.png' % int(i))
    # origin_img = cv2.imread(source_pth + img_path)

    use_lw = True
    if real_flag == False:
        # target = np.loadtxt(data_path + 'yolo_pose4keypoints/labels/val/%012d.txt' % int(i + 800))
        # target = np.loadtxt(data_path + 'knolling_data_small/labels/train/%012d.txt' % int(i))
        target_order = np.lexsort((target[:, 2], target[:, 1]))
        target = target[target_order]

    one_img = images[0]
    j = 0
    pred_result = []


    pred_xylws = one_img.boxes.xywhn.cpu().detach().numpy()
    pred_keypoints = one_img.keypoints.cpu().detach().numpy()
    pred_keypoints[:, :, :2] = pred_keypoints[:, :, :2] / np.array([640, 480])
    pred_keypoints = pred_keypoints.reshape(len(pred_xylws), -1)
    # for elements in one_img:
    #     pred_keypoints.append(elements.keypoints.cpu().detach().numpy())
    #     box = elements.boxes
    #     pred_xylws.append(box.xywhn.cpu().detach().numpy().reshape(-1, ))

    pred = np.concatenate((np.zeros((len(pred_xylws), 1)), pred_xylws, pred_keypoints), axis=1)
    pred_order = np.lexsort((pred[:, 2], pred[:, 1]))
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
        origin_img, result = plot_and_transform(im=origin_img, box=pred_xylw, label='0:, predic', color=(0, 0, 0), txt_color=(255, 255, 255), index=j,
                                        scaled_xylw=pred_xylw, keypoints=pred_keypoint, use_lw=use_lw, truth_flag=False)
        pred_result.append(result)
        print('this is j', j)

        if real_flag == False:
            tar_xylw = np.copy(target[j, 1:5])
            tar_keypoints = np.copy((target[j, 5:]).reshape(-1, 3)[:, :2])
            # tar_keypoints = (target[j, 5:])
            # tar_keypoints[:, 0] *= 640
            # tar_keypoints[:, 1] *= 480
            tar_label = '0: "target"'

            # plot target
            # print('this is tar xylw', tar_xylw)
            # print('this is tar cos sin', tar_keypoints)
            origin_img, _ = plot_and_transform(im=origin_img, box=pred_xylw, label='0: target', color=(255, 255, 0), txt_color=(255, 255, 255), index=j,
                                            scaled_xylw=tar_xylw, keypoints=tar_keypoints, use_lw=use_lw, truth_flag=True)

    cv2.namedWindow('zzz', 0)
    cv2.imshow('zzz', origin_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img_path_output = img_path + '_pred.png'
    cv2.imwrite(img_path_output, origin_img)

    if real_flag == False:
        print('this is length of pred\n', pred[:, 1:5])
        print('this is length of target\n', target[:, 1:5])
        loss_mean = np.mean((target - pred) ** 2)
        loss_std = np.std((target - pred), dtype=np.float64)
        print('this is mean error', loss_mean)
        print('this is std error', loss_std)

    print('this is key point')
    pred_result = np.asarray(pred_result)
    return pred_result

if __name__ == '__main__':

    data_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/datasets/'
    model_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/YOLOv8/runs/pose/train_standard_1000/weights/best.pt'
    zzz_result = yolov8_predict(data_path=data_path, model_path=model_path, real_flag=False)
    print('this is zzz result\n', zzz_result)
    # from ultralytics import YOLO
    #
    # model_path = "/home/ubuntu/Desktop/YOLOv8/runs/pose/train4/weights/"
    #
    # # Load a model
    # model = YOLO('yolov8n-knolling.yaml')  # build from YAML and transfer weights
    # model.load(model_path+'last.pt')
    #
    # source_pth = '/home/ubuntu/Desktop/datasets/knolling_data/images/val'
    #
    #
    # result = model(source=source_pth,conf = 0.5,save=True)
    # print(result)
