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

torch.manual_seed(42)

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


data_root = '/home/ubuntu/Desktop/knolling_dataset/resnet/'
log = 'log_412_norm_2/'
norm_flag = False

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("Device:", device)

model = ResNet50(img_channel=3, output_size=4).to(device, dtype=torch.float32)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.load_state_dict(torch.load(os.path.join(data_root, log, 'model/best_model.pt'), map_location='cuda:0'))
# add map_location='cuda:0' to run this model trained in multi-gpu environment on single-gpu environment

data_num = 30000
data_4_train = int(data_num * 0.8)
ratio = 0.5 # close3, normal7
close_num_train = int(data_4_train * ratio)
normal_num_train = int(data_4_train - close_num_train)
close_num_test = int((data_num - data_4_train) * ratio)
normal_num_test = int((data_num - data_4_train) - close_num_test)
print('this is num of close', int(close_num_train + close_num_test))
print('this is num of normal', int(normal_num_train + normal_num_test))

close_path = os.path.join(data_root, "input/yolo_407_close/")
normal_path = os.path.join(data_root, "input/yolo_407_normal/")
close_index = 0
normal_index = 0
train_data = []
test_data = []

close_label = np.loadtxt(os.path.join(data_root, 'label/label_407_close_train.csv'))
normal_label = np.loadtxt(os.path.join(data_root, 'label/label_407_normal_train.csv'))
# train_label = []
# test_label = []

train_label = np.concatenate((close_label[:close_num_train],normal_label[:normal_num_train]))
test_label  = np.concatenate((close_label[close_num_train:(close_num_train + close_num_test)],
                              normal_label[normal_num_train:(normal_num_train + close_num_test)]))
norm_parameters = np.concatenate((np.min(train_label,axis=0),np.max(train_label,axis=0)))
norm_parameters[5] = norm_parameters[1] + 1
print(norm_parameters.reshape(2, -1))
print(norm_parameters[4:] - norm_parameters[:4])
train_label -= norm_parameters[:4]
train_label /= (norm_parameters[4:] - norm_parameters[:4])
test_label -= norm_parameters[:4]
test_label /= (norm_parameters[4:] - norm_parameters[:4])

for i in range(close_num_train, close_num_train + close_num_test):
    img = plt.imread(close_path + "img%d.png" % i)
    test_data.append(img)

for i in range(normal_num_train, normal_num_train + normal_num_test):
    img = plt.imread(normal_path + "img%d.png" % i)
    test_data.append(img)

# for i in range(12):
#     img = plt.imread("../img_yolo%d.png" % i)
#     test_data.append(img)

# img = torch.from_numpy(img).to(device, dtype=torch.float32)

test_dataset = VD_Data(
    img_data=test_data, label_data=test_label)
################# choose the ratio of close and normal img #################

num_epochs = 100
BATCH_SIZE = 12
learning_rate = 1e-4

test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=4)

# model = ResNet50(img_channel=3, output_size=4).to(device, dtype=torch.float32)

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


    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            img, lwcossin = batch["image"], batch["lwcossin"]

            ############################## test the shape of img ##############################
            img_show = img.cpu().detach().numpy()
            # for i in range(len(img_show)):
            #     print(img_show[i].shape)
            #     temp = img_show[i]
            #     temp_shape = temp.shape
            #     temp = temp.reshape(temp_shape[1], temp_shape[2], temp_shape[0])
            #     print(temp.shape)
            #     cv2.namedWindow("well", 0)
            #     cv2.imshow('well', temp)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            # quit()
            ############################## test the shape of img ##############################

            img = img.to(device, dtype=torch.float32)
            lwcossin = lwcossin.to(device, dtype=torch.float32)
            # print('this is lwcossin\n', lwcossin)

            lwcossin_origin = lwcossin.cpu() * (norm_parameters[4:] - norm_parameters[:4]) + norm_parameters[:4]
            print('this is lwcossin origin\n', lwcossin_origin)

            pred_lwcossin = model.forward(img)
            # print('this is pred lwcossin\n', pred_lwcossin)

            pred_lwcossin_origin = pred_lwcossin.cpu() * (norm_parameters[4:] - norm_parameters[:4]) + norm_parameters[:4]
            print('this is pred lwcossin origin\n', pred_lwcossin_origin)
            loss = model.loss(pred_lwcossin, lwcossin)

            valid_L.append(loss.item())
    avg_valid_L = np.mean(valid_L)
    all_valid_L.append(avg_valid_L)
    print('this is avg_valid_L',avg_valid_L)

    # scheduler.step()
    # print('this is min_loss', min_loss)
    # print('this is avg_valid_L', avg_valid_L)
    # if avg_valid_L < min_loss:
    #     print('Training_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_train_L))
    #     print('Testing_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_valid_L))
    #     min_loss = avg_valid_L
    #
    #     PATH = model_path + 'best_model.pt'
    #
    #     # torch.save({
    #     #             'model_state_dict': model.state_dict(),
    #     #             }, PATH)
    #     torch.save(model.state_dict(), PATH)
    #
    #     abort_learning = 0
    # else:
    #     abort_learning += 1