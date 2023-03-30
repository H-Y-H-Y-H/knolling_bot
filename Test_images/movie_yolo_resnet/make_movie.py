import cv2
import numpy as np
import glob

fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('project.mp4', fourcc, 1, (640, 640))

img_array = []
for i in range(50):
    img_path = '%d.png' % i
    img = cv2.imread(img_path)
    print(img.shape)
    height, width, layers = img.shape
    size = (width, height)
    out.write(img)
out.release()