import numpy as np
import matplotlib.pyplot as plt

data_326 = np.loadtxt('../Dataset/label/label_326_normal.csv')
print(data_326.shape)

# cos2x, sin2x
truth_theta = 1.57
ground_truth = np.array([np.cos(4 * truth_theta), np.sin(4 * truth_theta)])

loss = []
for i in range(30):
    test_theta = (i / 30) * np.pi
    test = np.array([np.cos(4 * test_theta), np.sin(4 * test_theta)])

    loss.append(np.mean((ground_truth - test) ** 2))
loss = np.asarray(loss)
print(loss)

x = np.arange(30)
print(x)

plt.plot(x, loss)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
t = np.arange(0.0, 60.0)

truth_0 = 0
# test_0 = 7
delta_f = 0.01

ground_truth = np.array([np.cos(4 * truth_0), np.sin(4 * truth_0)])
loss = []
for i in range(60):
    test_theta = (i / 60) * np.pi
    test_loss = np.array([np.cos(4 * test_theta), np.sin(4 * test_theta)])

    loss.append(np.mean((ground_truth - test_loss) ** 2))
loss = np.asarray(loss)
print(loss)
l = plt.plot(t / 60 * 180, loss)
# plt.show()

# s = test_0 * np.sin(2 * np.pi * truth_0 * t)
# l, = plt.plot(t, s, lw=2)
ax.margins(x=0)

axcolor = 'lightgoldenrodyellow'
ax_truth = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
# ax_test = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

s_truth = Slider(ax_truth, 'truth_angle', 0.1, 3.20, valinit=truth_0, valstep=delta_f)
# s_test = Slider(ax_test, 'test_angle', 0.1, 10.0, valinit=test_0)


def update(val):
    # test = s_test.val
    truth = s_truth.val
    ground_truth = np.array([np.cos(4 * truth), np.sin(4 * truth)])
    loss = []
    for i in range(60):
        test_theta = (i / 60) * np.pi
        test_loss = np.array([np.cos(4 * test_theta), np.sin(4 * test_theta)])

        loss.append(np.mean((ground_truth - test_loss) ** 2))
    loss = np.asarray(loss)
    l[0].set_ydata(loss)
    fig.canvas.draw_idle()


s_truth.on_changed(update)
# s_test.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    s_truth.reset()
    # s_test.reset()
button.on_clicked(reset)

rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


def colorfunc(label):
    l[0].set_color(label)
    fig.canvas.draw_idle()
radio.on_clicked(colorfunc)

# Initialize plot with correct initial active value
colorfunc(radio.value_selected)

plt.show()