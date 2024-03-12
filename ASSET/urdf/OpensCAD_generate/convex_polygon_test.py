import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# Generate random points
num_points = 10
min_coord, max_coord = 0, 10
points = np.random.randint(min_coord, max_coord, size=(num_points, 2))

# Calculate convex hull
hull = ConvexHull(points)
convex_polygon = points[hull.vertices]
print(convex_polygon)

# Plot the random points and convex polygon
plt.scatter(points[:, 0], points[:, 1], color='blue', label='Random Points')
plt.plot(convex_polygon[:, 0], convex_polygon[:, 1], color='red', linestyle='-', linewidth=2, label='Convex Polygon')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Random Convex-Like Polygon')
plt.show()
