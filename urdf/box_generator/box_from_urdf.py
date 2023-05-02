from urdfpy import URDF
import numpy as np


lego_cube = URDF.load('template.urdf')


num_box = 50
for i in range(num_box):
    color_range = np.random.random(3)
    color_range = np.append(np.around(color_range, decimals=3), 1)
    lego_cube.links[0].visuals[0].material.color = color_range
    length_range = np.random.uniform(0.016, 0.048)
    width_range = np.random.uniform(0.016, 0.048)
    lego_cube.links[0].visuals[0].geometry.box.size[0] = np.around(length_range, decimals=3)
    lego_cube.links[0].visuals[0].geometry.box.size[1] = np.around(width_range, decimals=3)
    lego_cube.links[0].collisions[0].geometry.box.size[0] = np.around(length_range, decimals=3)
    lego_cube.links[0].collisions[0].geometry.box.size[1] = np.around(width_range, decimals=3)

    lego_cube.save('box_%d.urdf' % i)
