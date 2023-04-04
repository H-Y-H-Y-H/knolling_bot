import numpy as np

cur = np.array([-0.027, 0.139, 0.045])
seg = np.array([0, 0.15, 0.049])
a = np.sign(seg - cur)
print(a)