import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(42)

split_flag = True

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if split_flag == True:
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # First fully connected layer
            self.fc1 = nn.Linear(7, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 16)
            self.fc4 = nn.Linear(16, 4)

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

else:
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


# model = Net()

# print(model)

class VD_Data(Dataset):
    def __init__(self, label_cmd, label_real, transform=None):
        self.label_cmd = label_cmd
        self.label_real = label_real
        self.transform = transform

    def __getitem__(self, idx):
        cmd_sample = self.label_cmd[idx]
        real_sample = self.label_real[idx]

        sample = {'cmd': cmd_sample, 'real': real_sample}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.label_real)



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        cmd_sample, real_sample = sample['cmd'].astype(np.float32), sample['real'].astype(np.float32)

        return {'cmd': torch.from_numpy(cmd_sample),
                'real': torch.from_numpy(real_sample)}

def train():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("Device:", device)

    real_label = np.loadtxt('nn_data/all_distance_0035006_free/xyz_real_nn_split.csv')
    # real_label[:,3:] = real_label[:,3:].astype(int)
    cmd_label = np.loadtxt('nn_data/all_distance_0035006_free/cmd_nn_split.csv')

    log_path = 'model_pt/'

    data_num = len(real_label)
    data_4_train = int(data_num*0.8)

    train_dataset = VD_Data(
        label_cmd = cmd_label[:data_4_train], label_real = real_label[:data_4_train], transform=ToTensor())

    test_dataset = VD_Data(
        label_cmd = cmd_label[data_4_train:data_num], label_real = real_label[data_4_train:data_num], transform=ToTensor())

    num_epochs = 3000
    BATCH_SIZE = 32
    learning_rate = 1e-4

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    model = Net().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma = 0.1)
    min_loss = + np.inf
    all_train_L, all_valid_L = [], []

    abort_learning = 0

    if split_flag == True:
        cmd_sc = [[0, 0, 0, 0], [4096, 4096, 4096, 4096]]
        real_sc = [[-0.001, -0.2001, -0.0001, 0, 0, 0, 0],
                   [0.2501, 0.2001, 0.0601, 4096, 4096, 4096, 4096]]
    else:
        cmd_sc = [[0, 0, 0, 0, 0, 0], [4096, 4096, 4096, 4096, 4096, 4096]]
        real_sc = [[-0.001, -0.2001, -0.0001, 0, 0, 0, 0, 0, 0],
                   [0.2501, 0.2001, 0.0601, 4096, 4096, 4096, 4096, 4096, 4096]]

    for epoch in range(num_epochs):
        t0 = time.time()
        train_L, valid_L = [], []

        # Training Procedure
        model.train()
        for batch in train_loader:
            real_angle, cmd_angle = batch["real"], batch["cmd"]
            # real_angle = real_angle.to(device)
            # print(real_angle)
            scaler = MinMaxScaler()
            scaler.fit(cmd_sc)
            cmd_angle = scaler.transform(cmd_angle)
            cmd_angle = torch.from_numpy(cmd_angle)
            cmd_angle = cmd_angle.to(device)

            scaler_real = MinMaxScaler()
            scaler_real.fit(real_sc)
            real_angle = scaler_real.transform(real_angle).astype(np.float32)
            real_angle = torch.from_numpy(real_angle)
            real_angle = real_angle.to(device)


            optimizer.zero_grad()
            pred_angle = model.forward(real_angle)
            # print('pred',pred_angle)

            loss = model.loss(pred_angle, cmd_angle)

            loss.backward()
            optimizer.step()

            train_L.append(loss.item())

        avg_train_L = np.mean(train_L)
        all_train_L.append(avg_train_L)

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                real_angle, cmd_angle = batch["real"], batch["cmd"]
                # real_angle = real_angle.to(device)

                scaler = MinMaxScaler()
                scaler.fit(cmd_sc)
                # print('cmd',scaler.data_max_)
                cmd_angle = scaler.transform(cmd_angle)
                # print(cmd_angle)
                cmd_angle = torch.from_numpy(cmd_angle)
                cmd_angle = cmd_angle.to(device)

                scaler_real = MinMaxScaler()
                scaler_real.fit(real_sc)
                real_angle = scaler_real.transform(real_angle).astype(np.float32)
                real_angle = torch.from_numpy(real_angle)
                real_angle = real_angle.to(device)

                pred_angle = model.forward(real_angle)

                loss = model.loss(pred_angle, cmd_angle)

                valid_L.append(loss.item())

        avg_valid_L = np.mean(valid_L)
        all_valid_L.append(avg_valid_L)
        scheduler.step()

        if avg_valid_L < min_loss:
            print('Training_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_train_L))
            print('Testing_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_valid_L))
            min_loss = avg_valid_L

            PATH = log_path + 'all_distance_0035006_free_split.pt'

            # torch.save({
            #             'model_state_dict': model.state_dict(),
            #             }, PATH)
            torch.save(model.state_dict(), PATH)

            abort_learning = 0
        else:
            abort_learning += 1

        # np.savetxt(log_path + "training_L_arm.csv", np.asarray(all_train_L))
        # np.savetxt(log_path + "testing_L_arm.csv", np.asarray(all_valid_L))

        if abort_learning > 35:
            break
        t1 = time.time()
        print(epoch, "time used: ", (t1 - t0), "lr:", scheduler.get_last_lr())

    plt.plot(np.arange(len(all_train_L)), all_train_L, label='training')
    plt.plot(np.arange(len(all_valid_L)), all_valid_L, label='validation')
    plt.title("Learning Curve")
    plt.legend()
    # plt.savefig(log_path + "lc_arm.png")
    plt.show()


def eval():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("Device:", device)

    real_label = np.loadtxt('nn_data/all_distance_0035006_free/xyz_real_nn_split.csv')
    # real_label[:,3:] = real_label[:,3:].astype(int)
    cmd_label = np.loadtxt('nn_data/all_distance_0035006_free/cmd_nn_split.csv')

    log_path = 'model_pt/'

    data_num = len(real_label)
    data_4_train = int(data_num*0.8)
    test_num = 0
    test_num2 = test_num + 20

    test_dataset = VD_Data(
        label_cmd = cmd_label[test_num:test_num2], label_real = real_label[test_num:test_num2], transform=ToTensor())

    num_epochs = 1
    BATCH_SIZE = 20

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=0)

    model = Net().to(device)
    model.load_state_dict(torch.load(log_path + 'all_distance_0035006_free_split.pt'))

    if split_flag == True:
        cmd_sc = [[0, 0, 0, 0], [4096, 4096, 4096, 4096]]
        real_sc = [[-0.001, -0.2001, -0.0001, 0, 0, 0, 0],
                   [0.2501, 0.2001, 0.0601, 4096, 4096, 4096, 4096]]
    else:
        cmd_sc = [[0, 0, 0, 0, 0, 0], [4096, 4096, 4096, 4096, 4096, 4096]]
        real_sc = [[-0.001, -0.2001, -0.0001, 0, 0, 0, 0, 0, 0],
                   [0.2501, 0.2001, 0.0601, 4096, 4096, 4096, 4096, 4096, 4096]]

    for epoch in range(num_epochs):
        # Training Procedure
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                real_angle, cmd_angle = batch["real"], batch["cmd"]
                plot_cmd = cmd_angle.cpu().data.numpy()
                plot_real = real_angle[:,3:].cpu().data.numpy()
                # real_angle = real_angle.to(device)
                print('cmd', cmd_angle)
                print('real', real_angle[:,3:])
                scaler = MinMaxScaler()
                scaler.fit(cmd_sc)
                cmd_angle = scaler.transform(cmd_angle)
                cmd_angle = torch.from_numpy(cmd_angle)
                cmd_angle = cmd_angle.to(device)

                scaler_real = MinMaxScaler()
                scaler_real.fit(real_sc)
                real_angle = scaler_real.transform(real_angle).astype(np.float32)
                real_angle = torch.from_numpy(real_angle)
                real_angle = real_angle.to(device)

                pred_angle = model.forward(real_angle)

                loss = model.loss(pred_angle, cmd_angle)
                print('this is loss', loss)

                new_cmd = pred_angle.cpu().data.numpy()
                # print(new_cmd)

                new_cmd = scaler.inverse_transform(new_cmd)

                print(new_cmd)

    for i in range(len(plot_real[0])):
        plt.subplot(1, 4, i + 1)
        plt.plot(np.arange(len(plot_real)), plot_real[:, i], label='real')
        plt.plot(np.arange(len(plot_cmd)), plot_cmd[:, i], label='cmd')
        plt.plot(np.arange(len(new_cmd)), new_cmd[:, i], label='predict')
        plt.title(f'motor{i}')
        plt.legend()
    plt.suptitle("Comparison of cmd and real")
    plt.show()


if __name__ == "__main__":

    # train()

    eval()
