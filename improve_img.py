import cv2
from cam_obs import *
import sys
import numpy as np

sys.path.append('/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_bot/Adjust_images')

img = cv2.imread("./Adjust_images/326_testpip_1.png")
# img = np.uint8(np.clip((1.1 * img + 20), 0, 255))
# img = cv2.dilate(img, np.ones((2, 2), np.uint8))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.uint8(np.clip((1.1 * img + 20), 0, 255))


cv2.namedWindow("zzz_origin", 0)
cv2.imshow('zzz_origin', img)
cv2.waitKey(0)
cv2.destroyAllWindows()




















# rgb_planes = cv2.split(img)
# result_planes = []
# pixel = int(np.mean(img[img > 70]))
# print(pixel)
#
# for i in range(len(img)):
#     for j in range(len(img[0])):
#         if np.mean(img[i][j]) < 100 and np.max(img[i][j]) < 125 and np.min(img[i][j]) > 50:
#             img[i][j] = np.uint8(np.array([pixel, pixel, pixel]))
#         if j < 20 or j > 620 or 80 < i < 100 or 560 > i > 540:
#             img[i][j] = np.uint8(np.array([100, 100, 100]))
#         else:
#             pass
#
# cv2.imshow('zzz_result', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# # # img = cv2.medianBlur(img, 3)
# # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#
#     # 把灰白色部分修改为与背景接近的颜色
# # print(len(img[img < 30]))
# # img[img < 40] = pixel
# # result_planes.append(plane)
# #
# # result = cv2.merge(result_planes)
#
#
# cv2.imshow('zzz_result', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# results = np.asarray(detect(img, evaluation=10, real_operate=True, all_truth=None, order_truth=None))
# results = np.asarray(results[:, :5]).astype(np.float32)
# print('this is the result of yolo+resnet', results)