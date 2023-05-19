from urdfpy import URDF
import numpy as np
import os

lego_cube = URDF.load('template.urdf')
urdf_path = './generation_1/'
os.makedirs(urdf_path, exist_ok=True)

num_box = 100
for i in range(num_box):
    color_range = np.random.random(3)
    color_range = np.append(np.around(color_range, decimals=3), 1)
    lego_cube.links[0].visuals[0].material.color = color_range
    length_range = np.random.uniform(0.016, 0.048)
    width_range = np.random.uniform(0.016, min(length_range, 0.036))
    height_range = np.random.uniform(0.010, 0.020)
    lego_cube.links[0].visuals[0].geometry.box.size[0] = np.around(length_range, decimals=3)
    lego_cube.links[0].visuals[0].geometry.box.size[1] = np.around(width_range, decimals=3)
    lego_cube.links[0].visuals[0].geometry.box.size[2] = np.around(height_range, decimals=3)
    lego_cube.links[0].collisions[0].geometry.box.size[0] = np.around(length_range, decimals=3)
    lego_cube.links[0].collisions[0].geometry.box.size[1] = np.around(width_range, decimals=3)
    lego_cube.links[0].collisions[0].geometry.box.size[2] = np.around(height_range, decimals=3)

    lego_cube.save(urdf_path + 'box_%d.urdf' % i)

# for i in range(num_box):
#     lego_cube = URDF.load('box_%d.urdf' % i)
#     # print(lego_cube.links[0].collisions[0].geometry.box.size)
#     lego_cube.links[0].collisions[0].origin[2, 3] = 0
#     lego_cube.save('box_%d.urdf' % i)