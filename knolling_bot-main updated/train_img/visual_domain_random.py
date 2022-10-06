import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from VD_model import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class VD_Data(Dataset):
    def __init__(self, img_data, label_data, transform=None):
        self.img_data = img_data
        self.label_data = label_data
        self.transform = transform

    def __getitem__(self, idx):
        img_sample = self.img_data[idx]
        label_sample = self.label_data[idx]

        sample = {'image': img_sample, 'xyzyaw': label_sample}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.img_data)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img_sample, label_sample = sample['image'], sample['xyzyaw']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = img_sample.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image),
                'servo': torch.from_numpy(label_sample)}


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("Device:", device)


    xyzyaw = np.loadtxt('label.csv')
    img_pth = "data/img/"
    log_path = 'model/'


    data_num = 10000
    data_4_train = int(data_num*0.8)

    training_data = []
    for i in range(data_4_train):
        img = plt.imread(img_pth + "%d.png" % i)
        training_data.append(img)

    test_data = []
    for i in range(data_4_train, data_num):
        img = plt.imread(img_pth + "%d.png" % i)
        test_data.append(img)

    train_dataset = VD_Data(
        img_data = training_data, label_data = xyzyaw[:data_4_train], transform=ToTensor())

    test_dataset = VD_Data(
        img_data = test_data, label_data =  xyzyaw[data_4_train:] ,transform=ToTensor())


    num_epochs = 1000
    BATCH_SIZE = 64
    num_out = 4
    learning_rate = 1e-3


    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=0)



    model = ResNet101(img_channel=1, output_size=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    min_loss = + np.inf
    all_train_L, all_valid_L = [], []
    abort_learning = 0
    for epoch in range(num_epochs):
        t0 = time.time()
        train_L, valid_L = [], []

        # Training Procedure
        model.train()
        for batch in train_loader:
            img, xyzyaw = batch["images"], batch["xyzyaw"]
            pred_xyzyaw = model.forward(img)
            loss = model.loss(pred_xyzyaw, xyzyaw)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_L.append(loss.item())

        avg_train_L = np.mean(train_L)
        all_train_L.append(avg_train_L)

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                img, xyzyaw = batch["images"], batch["xyzyaw"]
                pred_xyzyaw = model.forward(img)
                loss = model.loss(pred_xyzyaw, xyzyaw)
                valid_L.append(loss.item())

        avg_valid_L = np.mean(valid_L)
        all_valid_L.append(avg_valid_L)

        if avg_valid_L < min_loss:
            print('Training_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_train_L))
            print('Testing_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_valid_L))
            min_loss = avg_valid_L
            PATH = log_path + '/best_model.pt'
            torch.save(model.state_dict(), PATH)
            abort_learning = 0
        else:
            abort_learning += 1


        np.savetxt(log_path + "training_L.csv", np.asarray(all_train_L))
        np.savetxt(log_path + "testing_L.csv", np.asarray(all_valid_L))

        if abort_learning > 20:
            break
        t1 = time.time()
        print(epoch, "time used: ", (t1 - t0) / (epoch + 1), "lr:", learning_rate)

    plt.plot(np.arange(len(all_train_L)), all_train_L, label='training')
    plt.plot(np.arange(len(all_valid_L)), all_valid_L, label='validation')
    plt.title("Learning Curve")
    plt.legend()
    plt.savefig(log_path + "lc.png")
    plt.show()





