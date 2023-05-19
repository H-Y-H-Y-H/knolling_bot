import cv2
import numpy as np
import os

def pose4keypoints(data_root, target_path, num_clusters):
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(target_path, exist_ok=True)
    os.makedirs(target_path + 'images/', exist_ok=True)
    os.makedirs(target_path + 'labels/', exist_ok=True)
    mm2px = 530 / 0.34  # (1558)
    total_num = 10

    for i in range(total_num):

        raw_img = cv2.imread(data_root + "origin_images/%012d.png" % i)

        # Reshape the image to a 2D array of pixels
        pixels = raw_img.reshape((-1, 3))

        # Convert the pixel values to floating point
        pixels = np.float32(pixels)

        # Define the criteria and apply k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Reshape the labels to the original image shape
        labels = labels.reshape(raw_img.shape[:2])

        # Create masks for each cluster label
        masks = []
        for i in range(num_clusters):
            masks.append(np.uint8(labels == i))

        # Show the segmented regions
        for i, mask in enumerate(masks):
            result = cv2.bitwise_and(raw_img, raw_img, mask=mask)
            cv2.imshow("Segmented Region " + str(i + 1), result)

        # Wait for key press and exit gracefully
        cv2.waitKey(0)
        cv2.destroyAllWindows()


data_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/yolo_pose4keypoints_4/'
target_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/datasets/yolo_pose4keypoints_518_motley/'
pose4keypoints(data_root, target_path, 4)
