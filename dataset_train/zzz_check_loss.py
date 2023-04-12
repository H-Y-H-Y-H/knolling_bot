import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# close_label = np.loadtxt('../Dataset/label/label_326_close.csv')
# normal_label = np.loadtxt('../Dataset/label/label_326_normal.csv')
# train_label = []
# test_label = []
# xyzyaw3 = np.copy(close_label)
#
# scaler = MinMaxScaler()
# scaler.fit(xyzyaw3)
# # scaler.fit(mm_sc)
# print(scaler.data_max_)
# print(scaler.data_min_)
#
# test = np.array([[0.024, 0.01601, 0.5, 0.5]])
# print(scaler.transform(test))


x = np.linspace(0, np.pi * 2, 100, endpoint=True)
y_sin = np.sin(2*x)
y_cos = np.cos(2*x)
print(len(y_cos))

# plt.plot(x, y_sin)
# plt.plot(x, y_cos)
# plt.show()
print(0.5 * 180 / np.pi)

print(np.cos(1))
print(np.sin(1))