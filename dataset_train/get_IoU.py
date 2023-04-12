import numpy as np
import cv2
from shapely.geometry import Polygon
import torch
from sklearn.preprocessing import MinMaxScaler

def calculate_riou(r1, r2):
    rect1 = ((r1[0], r1[1]), (r1[2], r1[3]), r1[4])
    rect2 = ((r2[0], r2[1]), (r2[2], r2[3]), r2[4])
    int_pts = cv2.rotatedRectangleIntersection(rect1, rect2)[1]
    if int_pts is not None:
        order_pts = cv2.convexHull(int_pts, returnPoints=True)
        int_area = cv2.contourArea(order_pts)
    else:
        int_area = 0
    return int_area

def calculate_iou_poly(r1, r2):
    yaw_1 = r1[0]
    yaw_2 = r2[0]
    matrix_1 = np.array([[np.cos(yaw_1), -np.sin(yaw_1)],
                         [np.sin(yaw_1), np.cos(yaw_1)]])
    matrix_2 = np.array([[np.cos(yaw_2), -np.sin(yaw_2)],
                         [np.sin(yaw_2), np.cos(yaw_2)]])
    corner_1 = np.array([[r1[2] / 2, r1[1] / 2],
                         [-r1[2] / 2, r1[1] / 2],
                         [-r1[2] / 2, -r1[1] / 2],
                         [r1[2] / 2, -r1[1] / 2]])
    corner_2 = np.array([[r2[2] / 2, r2[1] / 2],
                         [-r2[2] / 2, r2[1] / 2],
                         [-r2[2] / 2, -r2[1] / 2],
                         [r2[2] / 2, -r2[1] / 2]])
    # print(corner_1)
    corner_1_rotate = (matrix_1.dot(corner_1.T)).T
    # corner_1_rotate = torch.from_numpy(corner_1_rotate)
    corner_2_rotate = (matrix_2.dot(corner_2.T)).T
    # corner_2_rotate = torch.from_numpy(corner_2_rotate)
    print(corner_2_rotate)
    print(corner_1_rotate)
    poly_1 = Polygon(corner_1_rotate)
    poly_2 = Polygon(corner_2_rotate)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    # iou = 0.8

    demo = np.array([[0.008, 0.008, 0, 0.008],
                     [0, -0.008, -0.008, -0.008]])
    scaler = MinMaxScaler()
    scaler.fit(demo)
    # scaler.fit(mm_sc)
    print(scaler.data_max_)
    print(scaler.data_min_)
    grasp_point = np.array([[0, 0.008],
                     [0, -0.008]])

    target_grasp = (matrix_1.dot(grasp_point.T)).T
    pred_grasp = (matrix_2.dot(grasp_point.T)).T
    if target_grasp[0][0] <= target_grasp[1][0]:
        target_grasp[[0, 1]] = target_grasp[[1, 0]]
    if pred_grasp[0][0] <= pred_grasp[1][0]:
        pred_grasp[[0, 1]] = pred_grasp[[1, 0]]
    print(target_grasp)
    print(pred_grasp)
    target_grasp = scaler.transform(target_grasp.reshape(1, -1))
    pred_grasp = scaler.transform(pred_grasp.reshape(1, -1))
    print(target_grasp)
    print(pred_grasp)

    mse_loss = np.mean((target_grasp - pred_grasp) ** 2)
    print(np.sum((target_grasp - pred_grasp) ** 2) / 4)
    iou_loss = np.mean(-np.log(iou))
    print('iou', iou)
    print('iou loss', iou_loss ** 3)
    print('mse loss', mse_loss)
    print('total loss', np.mean((iou_loss ** 3) * mse_loss))

    # 1, 0.023, 0.002
    # iouloss
    # 2.122001132922936
    # mseloss
    # 0.00015229613252995718
    # totalloss
    # 0.0006857725506900522

    # 10, 0.023, 0.024
    # iouloss
    # 0.1682263525986691
    # mseloss
    # 0.01510569559615682
    # totalloss
    # 0.0004274927821739532

    # return iou

if __name__ == '__main__':
    # box1 = np.array([0, 0, 0.018685896, 0.058503106, 0], dtype=np.float32)
    # print(box1)
    # ((0, 0), (-0.018685896, -0.058503106), -0.014965076)
    # box2 = np.array([0, 0, 0.01868, 0.05, 0], dtype=np.float32)
    # print(box2)
    # ((0, 0), (0.0, 0.7080490527414577), 0.4444444444444444)
    # print(calculate_riou(box1, box2))

    rec1 = np.array([1.7, 0.02539, 0.01758], dtype=np.float32)
    rec2 = np.array([1.57, 0.024, 0.016], dtype=np.float32)
    iou = calculate_iou_poly(rec1, rec2)

    # matrix = np.array([[np.cos(0.001), -np.sin(0.001)],
    #                    [np.sin(0.001), np.cos(0.001)]])
    # grasp_point = np.array([[0, 0.015 / 2],
    #                         [0, -0.015 / 2]])
    # grasp_point_rotate = (matrix.dot(grasp_point.T)).T
    # print(grasp_point_rotate)

    # [7.50000000e-03  7.50000000e-03 - 2.41824853e-10  7.50000000e-03
    #  3.30000000e-02]
    # [2.41824853e-10 - 7.50000000e-03 - 7.50000000e-03 - 7.50000000e-03
    #  1.50000000e-02]

    # scaler = torch.tensor([[0.0075, 0.0075, 0, 0.0075, 0.033],
    #                   [0, -0.0075, -0.0075, -0.0075, 0.015]])
    #
    # input = torch.randn(8,5)
    # print(input)
    # output = (input - torch.min(input, 0)[0]) / (torch.max(input, 0)[0] - torch.min(input, 0)[0])
    # print(output)# output =
    #
    # y = torch.tensor([1.732, 1, 1])
    # x = torch.tensor([1, 1, 1.732])
    # print(torch.atan2(y, x))
    #
    # ori = torch.tensor(3.1415 / 2)
    #
    # matrix = torch.tensor([[torch.cos(ori), -torch.sin(ori)],
    #                        [torch.sin(ori), torch.cos(ori)]])
    #
    # grasp_point_2 = torch.tensor([[0.0016, -0.0073],
    #                               [-0.0016, 0.0073]])
    # print(torch.reshape(torch.t((matrix.mm(torch.t(grasp_point_2)))), (-1,)))