from torch import nn, optim
import torch
import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler
from shapely.geometry import Polygon
import time



class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=(1, 1)):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()

        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):

        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, output_size):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.0)
        self.sigmoid = nn.Sigmoid()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet_data architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 412_combine structure
        self.fc0 = nn.Linear(512 * 4, 512 * 4)
        self.fc1 = nn.Linear(512 * 4, 512 * 2)
        self.fc2 = nn.Linear(512 * 2, output_size)


    def forward(self, IMG):

        x = self.conv1(IMG)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 412_combine structure
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc0(x))
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)



        # return torch.cat((x1, x2), dim=-1)
        return x

    def calculate_riou(self, pred, target, scaler):

        pred = pred.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        pred_IoU = scaler.inverse_transform(pred)
        target_IoU = scaler.inverse_transform(target)
        device = 'cuda:1'

        # use shapely library to calculate iou
        total_iou = []
        for i in range(len(pred)):
            if pred_IoU[i][4] <= 0:
                iou = 0.0001
            else:
                width_1 = np.linalg.norm(pred_IoU[i, :2] - pred_IoU[i, 2:4])
                width_2 = np.linalg.norm(target_IoU[i, :2] - target_IoU[i, 2:4])
                yaw_1 = np.arctan2(pred_IoU[i, 1] - pred_IoU[i, 3], pred_IoU[i, 0] - pred_IoU[i, 2])
                yaw_2 = np.arctan2(target_IoU[i, 1] - target_IoU[i, 3], target_IoU[i, 0] - target_IoU[i, 2])
                # yaw_1 = pred_IoU[i][0]
                # yaw_2 = target_IoU[i][0]
                matrix_1 = np.array([[np.cos(yaw_1), -np.sin(yaw_1)],
                                     [np.sin(yaw_1), np.cos(yaw_1)]])
                matrix_2 = np.array([[np.cos(yaw_2), -np.sin(yaw_2)],
                                     [np.sin(yaw_2), np.cos(yaw_2)]])
                corner_1 = np.array([[width_1 / 2, pred_IoU[i][4] / 2],
                                     [-width_1 / 2, pred_IoU[i][4] / 2],
                                     [-width_1 / 2, -pred_IoU[i][4] / 2],
                                     [width_1 / 2, -pred_IoU[i][4] / 2]])
                corner_2 = np.array([[width_2 / 2, target_IoU[i][4] / 2],
                                     [-width_2 / 2, target_IoU[i][4] / 2],
                                     [-width_2 / 2, -target_IoU[i][4] / 2],
                                     [width_2 / 2, -target_IoU[i][4] / 2]])
                # print(corner_1)
                corner_1_rotate = (matrix_1.dot(corner_1.T)).T
                corner_2_rotate = (matrix_2.dot(corner_2.T)).T
                poly_1 = Polygon(corner_1_rotate)
                poly_2 = Polygon(corner_2_rotate)
                iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
            # if iou > 0.8:
            #     print(iou)
            #     print(pred_IoU[i][4:6])
            #     print(target_IoU[i][4:6])
            #     print(corner_1_rotate)
            #     print(corner_2_rotate)
            #     print(yaw_1)
            #     print(yaw_2)
            #     print('***************************')
            # if i == 0:
            #     print('pred', pred_IoU[i])
            #     print('tar', target_IoU[i])
            #     print('iou', iou)
            total_iou.append(iou)

        total_iou = -np.log(np.asarray(total_iou, dtype=np.float32))
        total_iou = torch.from_numpy(total_iou).to(device).reshape((-1, 1))
        # total_iou.requires_grad_()

        return total_iou

    def calculate_widthloss(self, pred, target):

        device = 'cuda:1'
        scaler = torch.tensor([[0.008, 0.008, 0, 0.008],
                               [0, -0.008, -0.008, -0.008]])
        scaler = scaler.to(device)

        pred_two_points = pred * (scaler[0] - scaler[1]) + scaler[1]
        target_two_points = target * (scaler[0] - scaler[1]) + scaler[1]
        pred_width = torch.sqrt(torch.square(pred_two_points[:, 1] - pred_two_points[:, 3]) + torch.square(pred_two_points[:, 0] - pred_two_points[:, 2]))
        target_width = torch.sqrt(torch.square(target_two_points[:, 1] - target_two_points[:, 3]) + torch.square(
            target_two_points[:, 0] - target_two_points[:, 2]))
        error = (pred_width - target_width) ** 2
        # print('this is width error', error[0])

        return error.reshape((-1, 1)) * 1000


    def loss(self, pred, target):

        # print(scaler.data_max_)
        # print(scaler.data_min_)
        # print('this is pred', pred)
        # print('this is target', target)

        value = (pred - target) ** 2

        # print(result)

        return torch.mean(value)

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels,
                  identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet_data 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


def ResNet18(img_channel, output_size):
    return ResNet(block, [2, 2, 2, 2], img_channel, output_size)


def ResNet50(img_channel, output_size):
    return ResNet(block, [3, 4, 6, 3], img_channel, output_size)


def ResNet101(img_channel, output_size):
    return ResNet(block, [3, 4, 23, 3], img_channel, output_size)


def ResNet152(img_channel, output_size):
    return ResNet(block, [3, 8, 36, 3], img_channel, output_size)

if __name__ == "__main__":
    import time
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Plot summary and test speed
    print("start", device)
    model = ResNet50(img_channel=1, output_size=6).to(device)
    img = torch.randn(32, 1, 256, 256).to(device)

    # summary(model, [(5, 128, 128),(1,1,24)])
    t1 = time.time()
    t0 = t1
    for i in range(100):
        outputs = model.forward(img)
        print(outputs.shape)
        outputs = outputs.cpu().detach().numpy()
        t2 = time.time()
        print(t2 - t1)
        t1 = time.time()

    t3 = time.time()
    print("all", (t3 - t0) / 100)
