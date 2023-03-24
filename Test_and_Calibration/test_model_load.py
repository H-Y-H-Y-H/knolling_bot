import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler

torch.manual_seed(42)

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()

      # First fully connected layer
      self.fc1 = nn.Linear(9, 64)
      self.fc2 = nn.Linear(64, 32)
      self.fc3 = nn.Linear(32, 16)
      self.fc4 = nn.Linear(16, 6)

    def forward(self, x):
        # define forward pass
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def loss(self, pred, target):
        value = (pred - target) ** 2
        return torch.mean(value)


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("Device:", device)

model = Net().to(device)
model.load_state_dict(torch.load("model_pt/far_004_006.pt"))
# model = Net()
# model.load_state_dict(torch.load("model_pt/far_004_006.pt", map_location=torch.device('cuda')))
xyz_real_angle = np.array([[6.041607807956227150e-02, -2.416373010540616428e-02, 5.540401881148097896e-02, 2.998000000000000000e+03, 1.768000000000000000e+03, 1.776000000000000000e+03, 1.442000000000000000e+03, 1.354000000000000000e+03, 1.971000000000000000e+03]])
cmd_angle = np.array([[2964.694091796875, 1793.352783203125, 1793.352783203125, 1473.2642822265625, 1371.4566650390625, 1939.701904296875]])
print(model)
model.eval()

scaler = StandardScaler()
with torch.no_grad():

    xyz_real_angle_after = scaler.fit_transform(xyz_real_angle)
    xyz_real_angle_after = torch.from_numpy(xyz_real_angle_after).float()
    xyz_real_angle_after = xyz_real_angle_after.to(device)

    cmd_angle_after = scaler.fit_transform(cmd_angle)
    cmd_angle_after = torch.from_numpy(cmd_angle_after).float()
    cmd_angle_after = cmd_angle_after.to(device)

    pred_angle = model.forward(xyz_real_angle_after)
    loss = model.loss(pred_angle, cmd_angle_after)
    print(loss.item())
    print(pred_angle)
    new_cmd = pred_angle.cpu().data.numpy()
    pred_angle = scaler.inverse_transform(new_cmd)
    print(pred_angle)
    print(type(pred_angle))