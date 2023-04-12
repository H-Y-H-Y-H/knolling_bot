matrix = np.array([[np.cos(yawori), -np.sin(yawori)],
                     [np.sin(yawori), np.cos(yawori)]])
grasp_point = np.array([[0.015, 0],
                        [-0.015, 0]])
grasp_point_rotate = (matrix.dot(grasp_point.T)).T
print('this is grasp point', grasp_point_rotate)

element = []
element.append(xpos1)
element.append(ypos1)
# element.append(rdm_pos_z)
element.append(yawori)
element.append(l)
element.append(w)

# if grasp_point_rotate[0][0] > grasp_point_rotate[1][0]:
#     element.append(grasp_point_rotate[0][0])
#     element.append(grasp_point_rotate[0][1])
#     element.append(grasp_point_rotate[1][0])
#     element.append(grasp_point_rotate[1][1])
# else:
#     element.append(grasp_point_rotate[1][0])
#     element.append(grasp_point_rotate[1][1])
#     element.append(grasp_point_rotate[0][0])
#     element.append(grasp_point_rotate[0][1])