## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################
import time

import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
# Start streaming
pipeline.start(config)

count = 0
num = 19
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())

        color_colormap_dim = color_image.shape
        resized_color_image = color_image

        #### add line


        ratio = 34 / 30
        # x_ratio = 0.96
        x_ratio = 0.975
        y_ratio = 480 * x_ratio * ratio / 640
        first_point = int((640 - 640 * y_ratio) / 2), int((480 - 480 * x_ratio) / 2)
        second_point = int((640 - 640 * y_ratio) / 2 + int(640 * y_ratio)), int((480 - 480 * x_ratio) / 2) + int(480 * x_ratio)
        print(first_point)
        print(second_point)

        resized_color_image = cv2.line(resized_color_image, (320, 0), (320, 640), (255, 255, 0), 1)
        resized_color_image = cv2.line(resized_color_image, (0, 240), (640, 240), (255, 255, 0), 1)
        #
        # resized_color_image =cv2.rectangle(resized_color_image, first_point, second_point, (0, 255, 0), 1)

        # visualize_img = cv2.resize(resized_color_image,(1280,960),interpolation = cv2.INTER_AREA)
        # Show images
        cv2.namedWindow('RealSense', 0)
        cv2.imshow('RealSense', resized_color_image)
        # cv2.imwrite("img.png",resized_color_image[112:368, 192:448])
        add = int((640 - 480) / 2)
        resized_color_image = resized_color_image[9:470, 58: 581]
        # resized_color_image = cv2.copyMakeBorder(resized_color_image, add, add, 0, 0, cv2.BORDER_CONSTANT, None, value=0)
        cv2.imwrite("img_4.png",resized_color_image)

        cv2.waitKey(1)
        #
        # if count//300 == 1:
        #     print("go")
        #     cv2.imwrite("data_train/img%s.png" % num, resized_color_image)
        #     num += 1
        #     print("you got 10s")
        #     time.sleep(10)

        # break
        # break
        count += 1


finally:

    # Stop streaming
    pipeline.stop()