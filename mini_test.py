import numpy as np
from collections import Counter

with open('real_after_pred.txt', 'r') as file:
    data = file.read().replace(',', ' ')
    data = list(data.split())
    temp_data = np.array([float(d) for d in data]).reshape(-1, 5)
    target = np.copy(temp_data)[:, 2:4]

print(target)

# Convert the array to a list of tuples
cube_tuples = [tuple(cube) for cube in target]

# Count the occurrences of each unique row
occurrences = Counter(cube_tuples)

# Print the count for each unique row
for cube, count in occurrences.items():
    print(cube, ":", count)