import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from VD_model import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class VD_Data(Dataset):
    def __init__(self, img_pth, dataset_type, label_data, transform=None):
        self.img_pth = img_pth
        self.label_data = label_data
        self.transform = transform
        self.data_type = dataset_type

    def __getitem__(self, idx):
        if self.data_type == "test":
            idx_i = int(idx + 80000)
        else:
            idx_i = idx
        img_sample = plt.imread(self.img_pth + 'IMG%d.png' % idx_i)[:, :, 0]
        label_sample = self.label_data[idx]

        sample = {'image': img_sample, 'xyzyaw': label_sample}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.label_data)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img_sample, label_sample = sample['image'], sample['xyzyaw']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img_sample = np.asarray([img_sample])
        # image = img_sample.transpose((2, 0, 1))

        return {'image': torch.from_numpy(img_sample),
                'xyzyaw': torch.from_numpy(label_sample)}


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("Device:", device)

    data_num = 100000
    xyzyaw = np.loadtxt('../Dataset/label/label_%dk.csv' % (data_num // 1000))
    img_pth = "../Dataset/Data_%dk/" % (data_num // 1000)
    log_path = '../model/'

    xyzyaw = xyzyaw[:, :2]

    data_4_train = int(data_num * 0.8)

    # training_data = []
    # for i in range(data_4_train):
    #     print(i)
    #     img = plt.imread(img_pth + "IMG%d.png" % i)[:,:,0]
    #     training_data.append(img)
    #
    # test_data = []
    # for i in range(data_4_train, data_num):
    #     img = plt.imread(img_pth + "IMG%d.png" % i)[:,:,0]
    #     test_data.append(img)

    print("data loaded")

    train_dataset = VD_Data(
        img_pth=img_pth, dataset_type="train", label_data=xyzyaw[:data_4_train], transform=ToTensor())

    test_dataset = VD_Data(
        img_pth=img_pth, dataset_type="test", label_data=xyzyaw[data_4_train:], transform=ToTensor())

    num_epochs = 50000
    BATCH_SIZE = 32
    num_out = 2
    learning_rate = 1e-4

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4)

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=4)

    # model.eval()

    model = ResNet18(img_channel=1, output_size=num_out).to(device)
    # model.load_state_dict(torch.load(log_path + 'best_model.pt'))
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
            img, xyzyaw = batch["image"], batch["xyzyaw"]
            img = img.cuda()
            xyzyaw = xyzyaw.cuda()

            optimizer.zero_grad()
            pred_xyzyaw = model.forward(img)
            loss = model.loss(pred_xyzyaw, xyzyaw)

            loss.backward()
            optimizer.step()
            train_L.append(loss.item())

        avg_train_L = np.mean(train_L)
        all_train_L.append(avg_train_L)

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                img, xyzyaw = batch["image"], batch["xyzyaw"]
                img = img.cuda()
                xyzyaw = xyzyaw.cuda()
                pred_xyzyaw = model.forward(img)
                loss = model.loss(pred_xyzyaw, xyzyaw)
                valid_L.append(loss.item())

        avg_valid_L = np.mean(valid_L)
        all_valid_L.append(avg_valid_L)

        if avg_valid_L < min_loss:
            print('Training_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_train_L))
            print('Testing_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_valid_L))
            min_loss = avg_valid_L
            PATH = log_path + 'best_model_50k.pt'
            torch.save(model.state_dict(), PATH)
            abort_learning = 0
        else:
            abort_learning += 1

        np.savetxt(log_path + "training_L_50k.csv", np.asarray(all_train_L))
        np.savetxt(log_path + "testing_L_50k.csv", np.asarray(all_valid_L))

        if abort_learning > 20:
            break
        t1 = time.time()
        print(epoch, "time used: ", (t1 - t0) / (epoch + 1), "lr:", learning_rate)

    # all_train_L = np.loadtxt("../model/training_L.csv")[0:10]
    # all_valid_L = np.loadtxt("../model/testing_L.csv")[0:10]
    plt.plot(np.arange(len(all_train_L)), all_train_L, label='training')
    plt.plot(np.arange(len(all_valid_L)), all_valid_L, label='validation')
    plt.title("Learning Curve")
    plt.legend()
    plt.savefig(log_path + "lc_50k.png")
    plt.show()
