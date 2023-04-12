import matplotlib.pyplot as plt
import numpy as np

all_train_L = np.loadtxt("../log/training_L_yolo_221_2.csv")
all_valid_L = np.loadtxt("../log/testing_L_yolo_221_2.csv")
plt.plot(np.arange(len(all_train_L)), all_train_L, label='training')
plt.plot(np.arange(len(all_valid_L)), all_valid_L, label='validation')
plt.title("Learning Curve")
plt.legend()
plt.show()