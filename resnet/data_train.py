from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append('/home/ubuntu/Desktop/knolling_bot')
sys.path.append('/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_bot')
from network import *

import wandb
wandb_flag = True

if wandb_flag == True:

    run = wandb.init(project='zzz_object_detection',
                     notes='knolling_bot',
                     tags=['baseline', 'paper1'],
                     name='413_resnet_101')
    wandb.config = {
        'data_num': 2000,
        'data_4_train': 0.8,
        'ratio': 0.5,
        'batch_size': 4
    }

torch.manual_seed(42)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class VD_Data(Dataset):
    def __init__(self, img_data, label_data):
        self.img_data = img_data
        self.label_data = label_data

    def __getitem__(self, idx):
        img_sample = self.img_data[idx]
        label_sample = self.label_data[idx]

        img_sample = img_sample[:,:,:3]
        img_sample = img_sample.transpose((2, 0, 1))

        # label_sample = scaler.transform(label_sample)

        img_sample = torch.from_numpy(img_sample)
        label_sample = torch.from_numpy(label_sample)

        sample = {'image': img_sample, 'lwcossin': label_sample}

        return sample

    def __len__(self):
        return len(self.img_data)


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print("Device:", device)

    finetune_flag = False
    data_root = '/home/ubuntu/Desktop/knolling_dataset/resnet_super/'
    log = 'log_412/'
    max_item = 15

    # eval_model()

    ################# choose the ratio of close and normal img #################
    model_path = os.path.join(data_root, log, 'model/')
    curve_path = os.path.join(data_root, log, 'curve/')
    log_path = os.path.join(data_root, log, 'log/')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(curve_path):
        os.makedirs(curve_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    data_num = 2000
    data_4_train = int(data_num * 0.8)
    ratio = 0.5 # close3, normal7
    close_num_train = int(data_4_train * ratio)
    normal_num_train = int(data_4_train - close_num_train)
    close_num_test = int((data_num - data_4_train) * ratio)
    normal_num_test = int((data_num - data_4_train) - close_num_test)
    print('this is num of close', int(close_num_train + close_num_test))
    print('this is num of normal', int(normal_num_train + normal_num_test))

    close_img_path = os.path.join(data_root, "input/close_409_img/")
    normal_img_path = os.path.join(data_root, "input/normal_409_img/")
    close_index = 0
    normal_index = 0
    train_data = []
    test_data = []

    close_label_path = os.path.join(data_root, 'label/close_409_label/')
    normal_label_path = os.path.join(data_root, 'label/normal_409_label/')
    train_label = []
    test_label = []

    # train_label = np.concatenate((close_label_path[:close_num_train], normal_label_path[:normal_num_train]))
    # test_label  = np.concatenate((close_label_path[close_num_train:(close_num_train + close_num_test)],
    #                               normal_label_path[normal_num_train:(normal_num_train + close_num_test)]))

    for i in range(close_num_train):
        img = plt.imread(close_img_path + "img%d.png" % i)
        train_data.append(img)
        label = np.loadtxt(close_label_path + "close_409_%d_train.csv" % i)
        train_label.append(label)

    for i in range(normal_num_train):
        img = plt.imread(normal_img_path + "img%d.png" % i)
        train_data.append(img)
        label = np.loadtxt(normal_label_path + "normal_409_%d_train.csv" % i)
        train_label.append(label)

    for i in range(close_num_train, close_num_train + close_num_test):
        img = plt.imread(close_img_path + "img%d.png" % i)
        test_data.append(img)
        label = np.loadtxt(close_label_path + "close_409_%d_train.csv" % i)
        test_label.append(label)

    for i in range(normal_num_train, normal_num_train + normal_num_test):
        img = plt.imread(normal_img_path + "img%d.png" % i)
        test_data.append(img)
        label = np.loadtxt(normal_label_path + "normal_409_%d_train.csv" % i)
        test_label.append(label)

    train_label = np.asarray(train_label)
    test_label = np.asarray(test_label)
    train_data = np.asarray(train_data)
    test_data = np.asarray(test_data)

    label_min_max = np.copy(train_label.reshape(-1, 7))
    for j in range(len(train_label)):
        for i in range(len(train_label[0])):
            # print('this is ', j, i, test_label[j][i])
            if train_label[j][i][2] > 0.2 or train_label[j][i][2] < -0.2 or train_label[j][i][1] > 0.3 or train_label[j][i][1] < 0:
                train_label[j][i] = 0

    for j in range(len(test_label)):
        for i in range(len(test_label[0])):
            # print('this is ', j, i, test_label[j][i])
            if test_label[j][i][2] > 0.2 or test_label[j][i][2] < -0.2 or test_label[j][i][1] > 0.3 or test_label[j][i][1] < 0:
                test_label[j][i] = 0


    norm_parameters = np.array([np.min(label_min_max, axis=0), np.max(label_min_max, axis=0)])
    norm_parameters[1][1] = norm_parameters[0][1] + 1
    # print(norm_parameters)
    # print(norm_parameters[0, :] - norm_parameters[1, :])

    # print(train_label.shape)
    train_label -= norm_parameters[0, :]
    train_label /= (norm_parameters[1, :] - norm_parameters[0, :])
    test_label -= norm_parameters[0, :]
    test_label /= (norm_parameters[1, :] - norm_parameters[0, :])


    # img = torch.from_numpy(img).to(device, dtype=torch.float32)

    train_dataset = VD_Data(
        img_data=train_data, label_data=train_label)

    test_dataset = VD_Data(
        img_data=test_data, label_data=test_label)
    ################# choose the ratio of close and normal img #################

    num_epochs = 100
    BATCH_SIZE = 4
    learning_rate = 1e-4

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4)

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=4)

    model = ResNet101(img_channel=3, output_size=7 * max_item).to(device, dtype=torch.float32)
    if finetune_flag == True:
        model.load_state_dict(torch.load(model_path + 'best_model.pt'))
    else:
        pass

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma = 0.1)
    min_loss = + np.inf
    all_train_L, all_valid_L = [], []
    pre_array = []
    tar_array = []

    # mm_sc = [[1.571, 0.033], [-1.571, 0.015]]
    # scaler = MinMaxScaler()
    # scaler.fit(xyzyaw3)
    # # scaler.fit(mm_sc)
    # print(scaler.data_max_)
    # print(scaler.data_min_)
    print('begin!')

    abort_learning = 0
    for epoch in range(num_epochs):
        t0 = time.time()
        train_L, valid_L = [], []

        # Training Procedure
        model.train()
        # model.eval()
        for batch in train_loader:

            img, lwcossin = batch["image"], batch["lwcossin"]

            img = img.to(device, dtype=torch.float32)
            lwcossin = lwcossin.to(device, dtype=torch.float32)
            # print('this is label shape', lwcossin.shape)

            optimizer.zero_grad()
            # print('this is img', img)
            pred_lwcossin = model.forward(img).reshape(BATCH_SIZE, max_item, 7)
            # print('this is pred', pred_lwcossin)
            # print('this is pred shape', pred_lwcossin.shape)
            # print('this is the length of pre', len(pred_xyzyaw))
            loss = model.loss(pred_lwcossin, lwcossin)
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

                img = img.to(device, dtype=torch.float32)
                lwcossin = lwcossin.to(device, dtype=torch.float32)

                pred_lwcossin = model.forward(img).reshape(BATCH_SIZE, max_item, 7)
                loss = model.loss(pred_lwcossin, lwcossin)

                valid_L.append(loss.item())
        avg_valid_L = np.mean(valid_L)
        all_valid_L.append(avg_valid_L)

        scheduler.step()
        # print('this is min_loss', min_loss)
        # print('this is avg_valid_L', avg_valid_L)
        if avg_valid_L < min_loss:
            print('Training_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_train_L))
            print('Testing_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_valid_L))
            min_loss = avg_valid_L

            PATH = model_path + 'best_model.pt'

            # torch.save({
            #             'model_state_dict': model.state_dict(),
            #             }, PATH)
            torch.save(model.state_dict(), PATH)

            abort_learning = 0
        else:
            abort_learning += 1

        if wandb_flag == True:
            wandb.log({'train loss': all_train_L, 'test loss': all_valid_L})

        np.savetxt(log_path + "training_L_yolo.csv", np.asarray(all_train_L))
        np.savetxt(log_path + "testing_L_yolo.csv", np.asarray(all_valid_L))
        # np.savetxt(log_path + "testing_L_yolo_115_ori.csv", np.asarray(all_valid_L))

        if abort_learning > 20:
            break
        t1 = time.time()
        print(epoch, "time used: ", (t1 - t0), "lr:", scheduler.get_last_lr())

    # all_train_L = np.loadtxt("../model/training_L_single.csv")
    # all_valid_L = np.loadtxt("../model/testing_L_single.csv")

    plt.plot(np.arange(len(all_train_L)), all_train_L, label='training')
    plt.plot(np.arange(len(all_valid_L)), all_valid_L, label='validation')
    plt.title("Learning Curve")
    plt.legend()
    plt.savefig(curve_path + "lc_411.png")
    # plt.show()

    # wandb.log_artifact(model)
    # wandb.save("model.onnx")
# '''