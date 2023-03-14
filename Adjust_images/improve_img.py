import cv2

img = cv2.imread("./312_testpip.png")

cv2.imshow('zzz_origin', img)
cv2.waitKey(0)
cv2.destroyAllWindows()