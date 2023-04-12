import cv2

img = cv2.imread('img_1.png')

# ratio = 34 / 30
# x_ratio = 0.965
# y_ratio = 480 * x_ratio * ratio / 640
# print(int((640 - 640 * y_ratio) / 2), int((480 - 480 * x_ratio) / 2))
# print(int((640 - 640 * y_ratio) / 2 + int(640 * y_ratio)), int((480 - 480 * x_ratio) / 2) + int(480 * x_ratio))
# first_point = (int((640 - 640 * y_ratio) / 2), int((480 - 480 * x_ratio) / 2))
# second_point = (int((640 - 640 * y_ratio) / 2 + int(640 * y_ratio)), int((480 - 480 * x_ratio) / 2) + int(480 * x_ratio))

# my_im2 = cv2.rectangle(img, (int((640 - 640 * y_ratio) / 2), int((480 - 480 * x_ratio) / 2) + 80),
#                                (int((640 - 640 * y_ratio) / 2 + int(640 * y_ratio)), int((480 - 480 * x_ratio) / 2) + int(480 * x_ratio) + 80),
#                                (255, 0, 0), 1)

img

cv2.namedWindow("zzz", 0)
cv2.imshow('zzz', img)
cv2.waitKey(0)
cv2.destroyWindow()