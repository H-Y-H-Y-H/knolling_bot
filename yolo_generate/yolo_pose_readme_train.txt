yolo_pose4points_1000
categories: 16x16, 24x16, 32x16
texture: None
quantity: 1000
label: 1, x, y, box_l * 3, box_w * 3, ori
urdf_cfg: lego items

yolo_pose4points_4000
categories: 30 random boxes
texture: None
quantity: 4000
label: 1, x, y, box_l * 3, box_w * 3, ori
urdf_cfg: normal

yolo_pose4points_507
categories: 30 random boxes
texture: None
quantity: 4000
label: 1, x, y, box_l * 3, box_w * 3, ori
urdf_cfg: normal

yolo_pose4points_507_2
categories: 30 random boxes
texture: None
quantity: 4000
label: 1, x, y, yolo_l, yolo_w, ori
urdf_cfg: normal

yolo_pose4points_508
categories: 30 random boxes
texture: None
quantity: 4000
label: 1, x, y, yolo_l, yolo_w, ori
urdf_cfg: normal
change sequence of kpts based on xy

yolo_pose4points_512
categories: 16x16, 24x16, 32x16
texture: None
quantity: 4000
label: 1, x, y, yolo_l, yolo_w, ori
urdf_cfg: lego items
change sequence of kpts based on xy

yolo_pose4points_512_2
categories: 16x16, 24x16, 32x16
texture: None
quantity: 4000
label: 1, x, y, yolo_l, yolo_w, ori
urdf_cfg: lego items
change sequence of kpts based on xy
add cv2.blur(raw_img, (3, 3)

yolo_pose4points_516_3
categories: 100 random boxes (generation_1)
texture: None
quantity: 10000
label: 1, x, y, yolo_l, yolo_w, ori
urdf_cfg: normal
change sequence of kpts based on xy

yolo_pose4points_516_blur
categories: 100 random boxes (generation_1)
texture: None
quantity: 10000
label: 1, x, y, yolo_l, yolo_w, ori
urdf_cfg: normal
change sequence of kpts based on xy
based on 516_2

yolo_pose4points_517
categories: 100 random boxes (generation_1)
texture: None
quantity: 10000
label: 1, x, y, yolo_l, yolo_w, ori
urdf_cfg: normal
change sequence of kpts based on the minimum value of the sum of the distance from the kpts to the corner
no flip idx
epoch: 200

yolo_pose4points_518
categories: 100 random boxes (generation_1)
texture: None
quantity: 10000
label: 1, x, y, yolo_l, yolo_w, ori
urdf_cfg: normal
no sequence of keypoints
epoch: 100

yolo_pose4points_518_2
categories: 100 random boxes (generation_1)
texture: None
quantity: 10000
label: 1, x, y, yolo_l, yolo_w, ori
urdf_cfg: normal
change sequence of kpts based on xy
no flip idx
epoch: 100

yolo_pose4points_518_gray
categories: 100 random boxes (generation_1)
texture: None
quantity: 15000
label: 1, x, y, yolo_l, yolo_w, ori
urdf_cfg: normal
change sequence of kpts based on xy
no flip idx
epoch: 200
gray images!

yolo_pose4points_518_3
categories: 100 random boxes (generation_1)
texture: None
quantity: 15000
label: 1, x, y, yolo_l, yolo_w, ori
urdf_cfg: normal
change sequence of kpts based on xy
no flip idx
epoch: 100
based on 518_2