import cv2
import sys
import numpy as np

# sys.path.append('/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_bot/Adjust_images')
sys.path.append('/home/ubuntu/Desktop/dataset_train/Dataset')

num = 150000
for i in range(num):
    img = cv2.imread(f"../Dataset/yolo_401_normal/img{i}.png")
    # print(img.shape)

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

    cv2.namedWindow("result", 0)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
