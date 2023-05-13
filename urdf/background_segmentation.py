import cv2
import numpy as np

raw_img = cv2.imread('./img_1.png')
x = raw_img.shape[0]
y = raw_img.shape[1]
print(raw_img.shape)
raw_img = cv2.resize(raw_img, (640, 480))

rotate_img = cv2.rotate(raw_img, cv2.ROTATE_180)
rotate_clock_img = cv2.rotate(raw_img, cv2.ROTATE_90_CLOCKWISE)
rotate_clock_img = cv2.resize(rotate_clock_img, (640, 480))
rotate_counterclock_img = cv2.rotate(raw_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
rotate_counterclock_img = cv2.resize(rotate_counterclock_img, (640, 480))

cv2.imwrite('./img_1.png', raw_img)
cv2.imwrite('./img_2.png', rotate_img)
cv2.imwrite('./img_3.png', rotate_clock_img)
cv2.imwrite('./img_4.png', rotate_counterclock_img)
# cv2.namedWindow('zzz', 0)
# cv2.imshow('zzz', raw_img)
# cv2.waitKey(0)
