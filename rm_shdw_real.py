import cv2
from cam_obs import *
import sys
import numpy as np

sys.path.append('/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_bot/Adjust_images')
sys.path.append('/home/ubuntu/Desktop/knolling_bot/Adjust_images')

img = cv2.imread("./Test_images/.png")
# img = cv2.dilate(img, np.ones((2, 2), np.uint8))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = np.uint8(np.clip((1.1 * img + 40), 0, 255))
# img = cv2.rectangle(img, (0, 80), (640, 94), (0, 0, 0), thickness=-1)
# img = cv2.rectangle(img, (0, 546), (640, 560), (0, 0, 0), thickness=-1)


cv2.namedWindow("zzz_origin", 0)
cv2.imshow('zzz_origin', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

blank_mask = np.zeros(img.shape, dtype=np.uint8)
original = img.copy()
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([0, 95, 20])
upper = np.array([255, 255, 255])
mask = cv2.inRange(hsv, lower, upper)

cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
cv2.drawContours(blank_mask,cnts, -1, (255,255,255), -1)
# for c in cnts:
#     cv2.drawContours(blank_mask,[c], -1, (255,255,255), -1)
#     break

result = cv2.bitwise_and(original,blank_mask)
pixel = 200
result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
result[result_gray < 1] = pixel

cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# rgb_planes = cv2.split(img)
#
# mask_plane = []
# for plane in rgb_planes: # sequence: BGR
#     # print(plane.shape)
#
#     mask_plane.append(plane < 100)
#     # print(plane < 90)
#
#     cv2.namedWindow("zzz_plane", 0)
#     cv2.imshow('zzz_plane', plane)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# mask = np.logical_and(np.logical_and(mask_plane[0], mask_plane[1]), mask_plane[2])
# print(mask)
# pixel = int(np.mean(img[img > 70]))
#
# img[mask] = pixel
# cv2.namedWindow("zzz_improved", 0)
# cv2.imshow('zzz_improved', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
















# # loop method
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
results = np.asarray(detect(result, evaluation=10, real_operate=True, all_truth=None, order_truth=None))
results = np.asarray(results[:, :5]).astype(np.float32)
print('this is the result of yolo+resnet', results)