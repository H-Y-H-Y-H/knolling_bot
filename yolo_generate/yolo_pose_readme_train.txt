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
change sequence of kpts

yolo_pose4points_512
categories: 16x16, 24x16, 32x16
texture: None
quantity: 4000
label: 1, x, y, yolo_l, yolo_w, ori
urdf_cfg: lego items
change sequence of kpts

yolo_pose4points_512_2
categories: 16x16, 24x16, 32x16
texture: None
quantity: 4000
label: 1, x, y, yolo_l, yolo_w, ori
urdf_cfg: lego items
change sequence of kpts
add cv2.blur(raw_img, (3, 3)