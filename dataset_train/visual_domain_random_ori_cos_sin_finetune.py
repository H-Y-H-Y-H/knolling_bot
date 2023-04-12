import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from train_img.VD_model import *
from VD_model_ori_cos_sin_finetune import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random
import cv2
import torchvision.transforms.functional as TF
# from pre_view import *
from sklearn.preprocessing import MinMaxScaler

import wandb

wandb_flag = True

if wandb_flag == True:

    run = wandb.init(project='zzz_object_detection',
                     notes='knolling_bot',
                     tags=['baseline', 'paper1'],
                     name='327_combine_3')
    wandb.config = {
        'data_num': 50000,
        'data_4_train': 0.8,
        'ratio': 0.5,
        'batch_size': 32
    }

torch.manual_seed(42)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class VD_Data(Dataset):
    def __init__(self, img_data, label_data, transform=None):
        self.img_data = img_data
        self.label_data = label_data
        self.transform = transform

    def __getitem__(self, idx):
        img_sample = self.img_data[idx]
        label_sample = self.label_data[idx]

        sample = {'image': img_sample, 'lwcossin': label_sample}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.img_data)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img_sample, label_sample = sample['image'], sample['lwcossin']
        img_sample = np.asarray([img_sample])

        # print(type(img_sample))
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # print(img_sample.shape)
        img_sample = np.squeeze(img_sample)
        img_sample = img_sample[:,:,:3]
        # print(img_sample.shape)

        image = img_sample.transpose((2, 0, 1))
        # print(image.shape)
        img_sample = torch.from_numpy(image)

        return {'image': img_sample,
                'lwcossin': torch.from_numpy(label_sample)}

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = 'cuda:1'
    else:
        device = 'cpu'
    print("Device:", device)
    # torch.cuda.set_device(1)

    # eval_model()
    ################# choose the ratio of close and normal img #################
    model_path = '../model/'
    curve_path = '../curve/'
    log_path = '../log/'

    data_num = 50000
    data_4_train = int(data_num * 0.8)
    ratio = 0.5 # close3, normal7
    close_num_train = data_4_train * ratio
    normal_num_train = data_4_train - close_num_train
    close_num_test = (data_num - data_4_train) * ratio
    normal_num_test = (data_num - data_4_train) - close_num_test

    close_path = "../Dataset/yolo_326_close/"
    normal_path = "../Dataset/yolo_326_normal/"
    close_index = 0
    normal_index = 0
    train_data = []
    test_data = []

    close_label = np.loadtxt('../Dataset/label/label_327_close.csv')
    normal_label = np.loadtxt('../Dataset/label/label_327_normal.csv')
    train_label = []
    test_label = []
    xyzyaw3 = np.copy(close_label)

    for i in range(int(close_num_train)):
        img = plt.imread(close_path + "img%d.png" % close_index)
        train_label.append(close_label[close_index])
        train_data.append(img)
        close_index += 1
    for i in range(int(normal_num_train)):
        img = plt.imread(normal_path + "img%d.png" % normal_index)
        train_label.append(normal_label[normal_index])
        train_data.append(img)
        normal_index += 1
    for i in range(int(close_num_test)):
        img = plt.imread(close_path + "img%d.png" % close_index)
        test_label.append(close_label[close_index])
        test_data.append(img)
        close_index += 1
    for i in range(int(normal_num_test)):
        img = plt.imread(normal_path + "img%d.png" % normal_index)
        test_label.append(normal_label[normal_index])
        test_data.append(img)
        normal_index += 1

    print('this is num of close', close_index)
    print('this is num of normal', normal_index)

    train_label = np.asarray(train_label)
    test_label = np.asarray(test_label)

    train_dataset = VD_Data(
        img_data=train_data, label_data=train_label, transform=ToTensor())

    test_dataset = VD_Data(
        img_data=test_data, label_data=test_label, transform=ToTensor())
    ################# choose the ratio of close and normal img #################

    num_epochs = 100
    BATCH_SIZE = 32
    learning_rate = 1e-5

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4)

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=4)

    # model.eval()

    model = ResNet50(img_channel=3, output_size=4).to(device)
    model.load_state_dict(torch.load('../model/best_model_306_combine.pt'))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma = 0.1)
    min_loss = + np.inf
    all_train_L, all_valid_L = [], []
    pre_array = []
    tar_array = []

    # mm_sc = [[1.571, 0.033], [-1.571, 0.015]]
    scaler = MinMaxScaler()
    scaler.fit(xyzyaw3)
    # scaler.fit(mm_sc)
    print(scaler.data_max_)
    print(scaler.data_min_)

    abort_learning = 0
    for epoch in range(num_epochs):
        t0 = time.time()
        train_L, valid_L = [], []


        # Training Procedure
        model.train()
        for batch in train_loader:
            img, lwcossin = batch["image"], batch["lwcossin"]

            img = img.to(device)
            lwcossin = scaler.transform(lwcossin)
            lwcossin = torch.from_numpy(lwcossin)
            lwcossin = lwcossin.to(device)
            # print(len(xyzyaw))
            optimizer.zero_grad()
            pred_lwcossin = model.forward(img)
            # print('this is the length of pre', len(pred_xyzyaw))
            loss = model.loss(pred_lwcossin, lwcossin, scaler)
            # print('test here', loss)
            loss.backward()
            optimizer.step()

            train_L.append(loss.item())

        avg_train_L = np.mean(train_L)
        # print('this is avg_train', avg_train_L)
        # time.sleep(5)
        all_train_L.append(avg_train_L)

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                img, lwcossin = batch["image"], batch["lwcossin"]
                img = img.to(device)
                # x1y1x2y2 = x1y1x2y2[:, 2:]

                lwcossin = scaler.transform(lwcossin)
                lwcossin = torch.from_numpy(lwcossin)
                lwcossin = lwcossin.to(device)

                pred_lwcossin = model.forward(img)

                loss = model.loss(pred_lwcossin, lwcossin, scaler)

                valid_L.append(loss.item())

        avg_valid_L = np.mean(valid_L)
        all_valid_L.append(avg_valid_L)
        scheduler.step()
        if avg_valid_L < min_loss:
            print('Training_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_train_L))
            print('Testing_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_valid_L))
            min_loss = avg_valid_L

            PATH = model_path + 'best_model_327_combine_3.pt'

            # torch.save({
            #             'model_state_dict': model.state_dict(),
            #             }, PATH)
            torch.save(model.state_dict(), PATH)

            abort_learning = 0
        else:
            abort_learning += 1

        if wandb_flag == True:
            wandb.log({'train loss': all_train_L, 'test loss': all_valid_L})

        np.savetxt(log_path + "training_L_yolo_327_combine_3.csv", np.asarray(all_train_L))
        np.savetxt(log_path + "testing_L_yolo_327_combine_3.csv", np.asarray(all_valid_L))
        # np.savetxt(log_path + "testing_L_yolo_115_ori.csv", np.asarray(all_valid_L))

        if abort_learning > 10:
            break
        t1 = time.time()
        print(epoch, "time used: ", (t1 - t0), "lr:", scheduler.get_last_lr())

    # all_train_L = np.loadtxt("../model/training_L_single.csv")
    # all_valid_L = np.loadtxt("../model/testing_L_single.csv")

    plt.plot(np.arange(len(all_train_L)), all_train_L, label='training')
    plt.plot(np.arange(len(all_valid_L)), all_valid_L, label='validation')
    plt.title("Learning Curve")
    plt.legend()
    plt.savefig(curve_path + "lc_327_combine_3.png")
    # plt.show()

    # wandb.log_artifact(model)
    # wandb.save("model.onnx")
# '''