import os
import cv2
import numpy as np
import json

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))

num_color = 7
png_files = [file for file in os.listdir(script_directory) if file.endswith('.png')]
rgb_dict = {}
num_sample = 10

for i in range(num_color):
    img = cv2.imread(script_directory + '/' + png_files[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rows = np.random.choice(img.shape[0], size=num_sample, replace=True)
    cols = np.random.choice(img.shape[1], size=num_sample, replace=True)
    positions = list(zip(rows, cols))
    selected_values = [(img[row, col, :] / 255).tolist() for row, col in positions]
    test = [(img[row, col, :]).tolist() for row, col in positions]

    rgb_dict[png_files[i][:-4]] = selected_values

    # cv2.namedWindow('zzz', 0)
    # cv2.resizeWindow('zzz', 1280, 960)
    # cv2.imshow('zzz', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print('here')

with open('./rgb_info.json', 'w') as f:
    json.dump(rgb_dict, f, indent=4)
print('here')

# print([244, 23] / 255)
#
# img = np.ones((480, 640, 3)) * 0.2
# cv2.namedWindow('zzz', 0)
# cv2.resizeWindow('zzz', 1280, 960)
# cv2.imshow('zzz', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()