import cv2
import numpy as np

original_image = np.ones((40, 100, 3), dtype=np.uint8)

color_start = np.array([50, 50, 50])
color_end = np.array([0, 0, 255])

for i in range(original_image.shape[0]):

    for j in range(int(original_image.shape[1])):
        original_image[i, j, 0] = color_start[0] + j / original_image.shape[1] * (color_end[0] - color_start[0])
        original_image[i, j, 1] = color_start[1] + j / original_image.shape[1] * (color_end[1] - color_start[1])
        original_image[i, j, 2] = color_start[2] + j / original_image.shape[1] * (color_end[2] - color_start[2])

cv2.namedWindow('zzz', 0)
cv2.imshow('zzz', original_image)
cv2.waitKey(0)

cv2.imwrite('./textures/red2.png', original_image)