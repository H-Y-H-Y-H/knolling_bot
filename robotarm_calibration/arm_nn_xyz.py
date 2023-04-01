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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(3, 12)
        self.fc2 = nn.Linear(12, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 12)
        self.fc6 = nn.Linear(12, 3)

    def forward(self, x):
        # define forward pass
        x = self.fc1(x)
        # x = torch.sigmoid(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        # nn.Dropout()
        x = self.fc6(x)
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

    real_label = np.loadtxt('nn_data_xyz/all_distance_free_new/real_xyz_nn.csv')
    real_min = np.min(real_label, axis=0).reshape(1, 3)
    real_max = np.max(real_label, axis=0).reshape(1, 3)
    cmd_label = np.loadtxt('nn_data_xyz/all_distance_free_new/cmd_xyz_nn.csv')
    cmd_min = np.min(cmd_label, axis=0).reshape(1, 3)
    cmd_max = np.max(cmd_label, axis=0).reshape(1, 3)

    np.save('nn_data_xyz/all_distance_free_new/real_scale.npy', np.concatenate((real_min, real_max), axis=0))
    np.save('nn_data_xyz/all_distance_free_new/cmd_scale.npy', np.concatenate((cmd_min, cmd_max), axis=0))
    real_label_demo = np.load('nn_data_xyz/all_distance_free_new/real_scale.npy')
    cmd_label_demo = np.load('nn_data_xyz/all_distance_free_new/cmd_scale.npy')

    log_path = 'model_pt_xyz/'

    data_num = len(real_label)
    data_4_train = int(data_num*0.8)

    train_dataset = VD_Data(
        label_cmd = cmd_label[:data_4_train], label_real = real_label[:data_4_train], transform=ToTensor())

    test_dataset = VD_Data(
        label_cmd = cmd_label[data_4_train:data_num], label_real = real_label[data_4_train:data_num], transform=ToTensor())

    num_epochs = 3000
    BATCH_SIZE = 32
    learning_rate = 1e-3

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    model = Net().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma = 0.1)
    min_loss = + np.inf
    all_train_L, all_valid_L = [], []

    abort_learning = 0

    # cmd_sc = [[-0.0001, -0.201, 0.0319],
    #           [0.25001, 0.201, 0.0801]]
    # real_sc = [[-0.01, -0.201, -0.01],
    #            [0.30, 0.201, 0.0601]]

    for epoch in range(num_epochs):
        t0 = time.time()
        train_L, valid_L = [], []

        # Training Procedure
        model.train()
        for batch in train_loader:
            real_xyz, cmd_xyz = batch["real"], batch["cmd"]
            # real_angle = real_angle.to(device)
            # print(real_angle)
            scaler_cmd = MinMaxScaler()
            scaler_cmd.fit(cmd_label_demo)
            cmd_xyz = scaler_cmd.transform(cmd_xyz)
            cmd_xyz = torch.from_numpy(cmd_xyz)
            cmd_xyz = cmd_xyz.to(device)

            scaler_real = MinMaxScaler()
            scaler_real.fit(real_label_demo)
            real_xyz = scaler_real.transform(real_xyz).astype(np.float32)
            real_xyz = torch.from_numpy(real_xyz)
            real_xyz = real_xyz.to(device)


            optimizer.zero_grad()
            pred_xyz = model.forward(real_xyz)
            # print('pred',pred_angle)

            loss = model.loss(pred_xyz, cmd_xyz)

            loss.backward()
            optimizer.step()

            train_L.append(loss.item())

        avg_train_L = np.mean(train_L)
        all_train_L.append(avg_train_L)

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                real_xyz, cmd_xyz = batch["real"], batch["cmd"]
                # real_angle = real_angle.to(device)
                # print(real_angle)
                scaler_cmd = MinMaxScaler()
                scaler_cmd.fit(cmd_label_demo)
                cmd_xyz = scaler_cmd.transform(cmd_xyz)
                cmd_xyz = torch.from_numpy(cmd_xyz)
                cmd_xyz = cmd_xyz.to(device)

                scaler_real = MinMaxScaler()
                scaler_real.fit(real_label_demo)
                real_xyz = scaler_real.transform(real_xyz).astype(np.float32)
                real_xyz = torch.from_numpy(real_xyz)
                real_xyz = real_xyz.to(device)

                optimizer.zero_grad()
                pred_xyz = model.forward(real_xyz)

                loss = model.loss(pred_xyz, cmd_xyz)

                valid_L.append(loss.item())

        avg_valid_L = np.mean(valid_L)
        all_valid_L.append(avg_valid_L)
        scheduler.step()

        if avg_valid_L < min_loss:
            print('Training_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_train_L))
            print('Testing_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_valid_L))
            min_loss = avg_valid_L

            PATH = log_path + 'all_distance_free_new.pt'

            # torch.save({
            #             'model_state_dict': model.state_dict(),
            #             }, PATH)
            torch.save(model.state_dict(), PATH)

            abort_learning = 0
        else:
            abort_learning += 1

        np.savetxt(log_path + "training_L_arm.csv", np.asarray(all_train_L))
        np.savetxt(log_path + "testing_L_arm.csv", np.asarray(all_valid_L))

        if abort_learning > 10:
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

    real_label = np.loadtxt('nn_data_xyz/all_distance_free_new/real_xyz_nn.csv')
    real_min = np.min(real_label, axis=0).reshape(1, 3)
    real_max = np.max(real_label, axis=0).reshape(1, 3)
    cmd_label = np.loadtxt('nn_data_xyz/all_distance_free_new/cmd_xyz_nn.csv')
    cmd_min = np.min(cmd_label, axis=0).reshape(1, 3)
    cmd_max = np.max(cmd_label, axis=0).reshape(1, 3)

    np.save('nn_data_xyz/all_distance_free_new/real_scale.npy', np.concatenate((real_min, real_max), axis=0))
    np.save('nn_data_xyz/all_distance_free_new/cmd_scale.npy', np.concatenate((cmd_min, cmd_max), axis=0))
    real_label_demo = np.load('nn_data_xyz/all_distance_free_new/real_scale.npy')
    cmd_label_demo = np.load('nn_data_xyz/all_distance_free_new/cmd_scale.npy')
    print('this is real scale', real_label_demo)
    print('this is cmd scale', cmd_label_demo)

    log_path = 'model_pt_xyz/'

    data_num = len(real_label)
    data_4_train = int(data_num*0.8)
    test_num = 500
    test_num2 = test_num + 20

    test_dataset = VD_Data(
        label_cmd = cmd_label[test_num:test_num2], label_real = real_label[test_num:test_num2], transform=ToTensor())

    num_epochs = 1
    BATCH_SIZE = 20

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=0)

    model = Net().to(device)
    model.load_state_dict(torch.load(log_path + 'all_distance_free_new.pt'))

    for epoch in range(num_epochs):
        # Training Procedure
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                real_xyz, cmd_xyz = batch["real"], batch["cmd"]
                plot_cmd = cmd_xyz.cpu().data.numpy()
                plot_real = real_xyz.cpu().data.numpy()
                # real_angle = real_angle.to(device)
                # print('cmd', cmd_xyz)
                # print('real', real_xyz)
                scaler_cmd = MinMaxScaler()
                scaler_cmd.fit(cmd_label_demo)
                cmd_xyz = scaler_cmd.transform(cmd_xyz)
                cmd_xyz = torch.from_numpy(cmd_xyz)
                cmd_xyz = cmd_xyz.to(device)

                scaler_real = MinMaxScaler()
                scaler_real.fit(real_label_demo)
                real_xyz = scaler_real.transform(real_xyz).astype(np.float32)
                real_xyz = torch.from_numpy(real_xyz)
                real_xyz = real_xyz.to(device)

                pred_xyz = model.forward(real_xyz)

                loss = model.loss(pred_xyz, cmd_xyz)
                # print('this is loss', loss)

                new_cmd = pred_xyz.cpu().data.numpy()
                # print(new_cmd)

                new_cmd = scaler_cmd.inverse_transform(new_cmd)

                # print(new_cmd)

    for i in range(len(plot_real[0])):
        plt.subplot(1, 3, i + 1)
        plt.plot(np.arange(len(plot_real)), plot_real[:, i], label='real')
        plt.plot(np.arange(len(plot_cmd)), plot_cmd[:, i], label='cmd')
        plt.plot(np.arange(len(new_cmd)), new_cmd[:, i], label='predict')
        plt.title(f'{i}')
        plt.legend()
    plt.suptitle("Comparison of cmd and real")
    plt.show()


if __name__ == "__main__":

    # train()

    eval()
