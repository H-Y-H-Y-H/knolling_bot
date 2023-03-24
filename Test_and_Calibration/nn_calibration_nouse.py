#                        .::::.
#                      .::::::::.
#                     :::::::::::
#                  ..:::::::::::'
#               '::::::::::::'
#                 .::::::::::
#            '::::::::::::::..
#                 ..::::::::::::.
#               ``::::::::::::::::
#                ::::``:::::::::'        .:::.
#               ::::'   ':::::'       .::::::::.
#             .::::'     :::::     .:::::::''::::.
#            .:::'       :::::  .:::::::::'  ':::::.
#           .::'        :::::.:::::::::'      '::::::.
#          .::'         ::::::::::::::'         ``:::::
#      ...:::           ::::::::::::'              `::::.
#     ```` ':.          ':::::::::'                  ::::::..
#                        '.:::::'                    ':'``````:.
#                     美女保佑 永无BUG
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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()

      # First fully connected layer
      self.fc1 = nn.Linear(9, 32)
      self.fc2 = nn.Linear(32, 64)
      self.fc5 = nn.Linear(64, 32)
      self.fc6 = nn.Linear(32, 8)
      self.fc7 = nn.Linear(8, 6)


    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)

        return x

    def loss(self, pred, target):

        value = (pred - target) ** 2

        return torch.mean(value)

class VD_Data(Dataset):
    def __init__(self, label_cmd, label_xyz_real, transform=None):
        self.label_cmd = label_cmd
        self.label_xyz_real = label_xyz_real
        self.transform = transform

    def __getitem__(self, idx):
        cmd_sample = self.label_cmd[idx]
        xyz_real_sample = self.label_xyz_real[idx]

        sample = {'cmd': cmd_sample, 'xyz_real': xyz_real_sample}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.label_xyz_real)



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        cmd_sample, xyz_real_sample = sample['cmd'].astype(np.float32), sample['xyz_real'].astype(np.float32)

        return {'cmd': torch.from_numpy(cmd_sample),
                'xyz_real': torch.from_numpy(xyz_real_sample)}

def normalization(data):
    mean_data = torch.mean(data, dim=1)
    std_data = torch.std(data, dim=1)
    n_data = data.sub_(mean_data[:, None]).div_(std_data[:, None])
    return n_data

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("Device:", device)

    xyz_real_label = np.loadtxt('nn_data/xyz_real_nn.csv')
    cmd_label = np.loadtxt('nn_data/cmd_nn.csv')

    log_path = ''

    data_num = 17977
    data_4_train = int(data_num*0.8)

    train_dataset = VD_Data(label_cmd = cmd_label[:data_4_train],
                            label_xyz_real = xyz_real_label[:data_4_train],
                            transform=ToTensor())

    test_dataset = VD_Data(label_cmd = cmd_label[data_4_train:data_num],
                           label_xyz_real = xyz_real_label[data_4_train:data_num],
                           transform=ToTensor())

    num_epochs = 3000
    BATCH_SIZE = 32
    learning_rate = 1e-4

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=0)

    model = Net().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma = 0.1)
    min_loss = + np.inf
    all_train_L, all_valid_L = [], []

    abort_learning = 0

    for epoch in range(num_epochs):
        t0 = time.time()
        train_L, valid_L = [], []

        # Training Procedure
        model.train()
        for batch in train_loader:
            # print('this is batch', batch)
            xyz_real_angle, cmd_angle = batch["xyz_real"], batch["cmd"]

            # # print('this is cmd before', cmd_angle)
            # cmd_angle = normalization(cmd_angle)
            # # print('this is cmd after', cmd_angle)
            # cmd_angle = torch.tensor(cmd_angle, dtype=torch.float)
            # cmd_angle = cmd_angle.to(device)
            #
            # # print('this is xyz before', xyz_real_angle)
            # xyz_real_angle = normalization(xyz_real_angle)
            # # print('this is xyz after', xyz_real_angle)
            # xyz_real_angle = torch.tensor(xyz_real_angle, dtype=torch.float)
            # xyz_real_angle = xyz_real_angle.to(device)

            scaler = StandardScaler()
            # print(cmd_angle)
            cmd_angle_after = scaler.fit_transform(cmd_angle.T)
            # print(cmd_angle_after.T)
            cmd_angle_after = torch.from_numpy(cmd_angle_after.T)
            cmd_angle_after = cmd_angle_after.to(device)
            xyz_real_angle_after = scaler.fit_transform(xyz_real_angle.T)
            xyz_real_angle_after = torch.from_numpy(xyz_real_angle_after.T).float()
            xyz_real_angle_after = xyz_real_angle_after.to(device)

            optimizer.zero_grad()
            pred_angle = model.forward(xyz_real_angle_after)
            pred_angle = pred_angle

            loss = model.loss(pred_angle, cmd_angle_after)
            loss.backward()
            optimizer.step()

            train_L.append(loss.item())

        avg_train_L = np.mean(train_L)
        all_train_L.append(avg_train_L)

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                xyz_real_angle, cmd_angle = batch["xyz_real"], batch["cmd"]

                # # print('this is cmd before', cmd_angle)
                # cmd_angle = normalization(cmd_angle)
                # # print('this is cmd after', cmd_angle)
                # cmd_angle = torch.tensor(cmd_angle, dtype=torch.float)
                # cmd_angle = cmd_angle.to(device)
                #
                # # print('this is xyz before', xyz_real_angle)
                # xyz_real_angle = normalization(xyz_real_angle)
                # # print('this is xyz after', xyz_real_angle)
                # xyz_real_angle = torch.tensor(xyz_real_angle, dtype=torch.float)
                # xyz_real_angle = xyz_real_angle.to(device)

                scaler = StandardScaler()
                cmd_angle_after = scaler.fit_transform(cmd_angle.T)
                cmd_angle_after = torch.from_numpy(cmd_angle_after.T)
                cmd_angle_after = cmd_angle_after.to(device)
                xyz_real_angle_after = scaler.fit_transform(xyz_real_angle.T)
                xyz_real_angle_after = torch.from_numpy(xyz_real_angle_after.T).float()
                xyz_real_angle_after = xyz_real_angle_after.to(device)

                pred_angle = model.forward(xyz_real_angle_after)
                pred_angle = pred_angle

                loss = model.loss(pred_angle, cmd_angle_after)

                valid_L.append(loss.item())

        avg_valid_L = np.mean(valid_L)
        all_valid_L.append(avg_valid_L)
        scheduler.step()

        if avg_valid_L < min_loss:
            print('Training_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_train_L))
            print('Testing_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_valid_L))
            min_loss = avg_valid_L

            PATH = log_path + 'best_model_arm.pt'

            # torch.save({
            #             'model_state_dict': model.state_dict(),
            #             }, PATH)
            torch.save(model.state_dict(), PATH)

            abort_learning = 0
        else:
            abort_learning += 1

        np.savetxt(log_path + "training_L_arm.csv", np.asarray(all_train_L))
        np.savetxt(log_path + "testing_L_arm.csv", np.asarray(all_valid_L))

        if abort_learning > 60:
            break
        t1 = time.time()
        print(epoch, "time used: ", (t1 - t0), "lr:", scheduler.get_last_lr())

    plt.plot(np.arange(len(all_train_L)), all_train_L, label='training')
    plt.plot(np.arange(len(all_valid_L)), all_valid_L, label='validation')
    plt.title("Learning Curve")
    plt.legend()
    plt.savefig(log_path + "lc_arm.png")
    plt.show()

