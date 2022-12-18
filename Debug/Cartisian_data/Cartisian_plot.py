import matplotlib.pyplot as plt
import numpy as np

plot_cmd = np.loadtxt('cmd.txt')
plot_real = np.loadtxt('real.txt')
plot_step = np.loadtxt('step.txt')

plt.subplot(2, 3, 1)
plt.plot(plot_step, plot_cmd[:, 0].reshape(-1), label='cmd')
plt.plot(plot_step, plot_real[:, 0].reshape(-1), label='real')
plt.title("Motor 0")

plt.subplot(2, 3, 2)
plt.plot(plot_step, plot_cmd[:, 1].reshape(-1), label='cmd')
plt.plot(plot_step, plot_real[:, 1].reshape(-1), label='real')
plt.title("Motor 1")

plt.subplot(2, 3, 3)
plt.plot(plot_step, plot_cmd[:, 2].reshape(-1), label='cmd')
plt.plot(plot_step, plot_real[:, 2].reshape(-1), label='real')
plt.title("Motor 2")

plt.subplot(2, 3, 4)
plt.plot(plot_step, plot_cmd[:, 3].reshape(-1), label='cmd')
plt.plot(plot_step, plot_real[:, 3].reshape(-1), label='real')
plt.title("Motor 3")

plt.subplot(2, 3, 5)
plt.plot(plot_step, plot_cmd[:, 4].reshape(-1), label='cmd')
plt.plot(plot_step, plot_real[:, 4].reshape(-1), label='real')
plt.title("Motor 4")

plt.subplot(2, 3, 6)
plt.plot(plot_step, plot_cmd[:, 5].reshape(-1), label='cmd')
plt.plot(plot_step, plot_real[:, 5].reshape(-1), label='real')
plt.title("Motor 5")

plt.legend(loc='best')
plt.show()