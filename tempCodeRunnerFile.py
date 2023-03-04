import cv2
import numpy as np

a = np.uint8(np.random.randint(0, 255, [96,96,3]))
print(a.shape)
# print(a)

for i in range(a.shape[0]):
    print(a[i])
a_split = cv2.split(a)

result_planes = []
result_norm_planes = []
for plane in a_split:
    dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)

result = cv2.merge(result_planes)
result_norm = cv2.merge(result_norm_planes)

cv2.imshow(';aaa', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow(';aaa', result_norm)
cv2.waitKey(0)
cv2.destroyAllWindows()