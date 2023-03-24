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

# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
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
        # depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        # if not depth_frame or not color_frame:
        #     continue

        # Convert images to numpy arrays
        # depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape
        resized_color_image = color_image
        # If depth and color resolutions are different, resize color image to match depth image for display
        # if depth_colormap_dim != color_colormap_dim:
        #     resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        #     images = np.hstack((resized_color_image, depth_colormap))
        # else:
        #     images = np.hstack((color_image, depth_colormap))
        #     resized_color_image = color_image

        #### add line

        # print(resized_color_image)
        resized_color_image = cv2.line(resized_color_image, (192, 368), (448, 368), (255, 0, 0), 1)
        resized_color_image = cv2.line(resized_color_image, (192, 112), (448, 112), (255, 0, 0), 1)
        resized_color_image = cv2.line(resized_color_image, (192, 368), (192, 112), (0, 0, 0), 1)
        resized_color_image = cv2.line(resized_color_image, (448, 368), (448, 112), (0, 0, 0), 1)
        resized_color_image = cv2.line(resized_color_image, (320, 0), (320, 640), (0, 255, 0), 1)
        resized_color_image = cv2.line(resized_color_image, (0, 240), (640, 240), (0, 255, 0), 1)

        # resized_color_image = cv2.line(resized_color_image, (87, 40), (87, 0), (0, 255, 0), 1)
        # resized_color_image = cv2.line(resized_color_image, (87, 40), (0, 40), (0, 255, 0), 1)
        #
        # resized_color_image = cv2.line(resized_color_image, (553,40), (553,0), (0, 255, 0), 1)
        # resized_color_image = cv2.line(resized_color_image, (553,40), (640,40), (0, 255, 0), 1)
        #
        # resized_color_image = cv2.line(resized_color_image, (87, 440), (0, 440), (0, 255, 0), 1)
        # resized_color_image = cv2.line(resized_color_image, (87, 440), (87, 480), (0, 255, 0), 1)
        #
        # resized_color_image = cv2.line(resized_color_image, (553,440), (640,440), (0, 255, 0), 1)
        # resized_color_image = cv2.line(resized_color_image, (553,440), (553,480), (0, 255, 0), 1)

        resized_color_image =cv2.rectangle(resized_color_image, (65, 15), (575, 465), (0, 255, 0), 1)


        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', resized_color_image)
        # cv2.imwrite("img.png",resized_color_image[112:368, 192:448])
        add = int((640 - 480) / 2)
        resized_color_image = cv2.copyMakeBorder(resized_color_image, add, add, 0, 0, cv2.BORDER_CONSTANT, None, value=0)
        cv2.imwrite("img%.png",resized_color_image)

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