import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import Transformer
from torch.utils.data import Dataset, DataLoader
import math
import torch.optim as optim
import sys
sys.path.append('../../train_multi_knolling/')
sys.path.append('../')

from pos_encoder import *
import torch.nn.functional as F
from new_model import *
import wandb

# Define the network
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, output_size, do_para):
        super(MLP, self).__init__()
        self.num_layer = num_layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.do_para = do_para
        self.fc_layer = []
        self.drop_layer = nn.Dropout(p=do_para)

        if num_layer == 2:
            self.fc1 = nn.Linear(self.input_size, self.hidden_size)
            self.fc2 = nn.Linear(self.hidden_size, self.output_size)
            self.fc_layer = [self.fc1,self.fc2]

        elif num_layer == 3:
            self.fc1 = nn.Linear(self.input_size, self.hidden_size)
            self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
            self.fc3 = nn.Linear(self.hidden_size, self.output_size)
            self.fc_layer = [self.fc1,self.fc2,self.fc3]

        else:
            self.fc1 = nn.Linear(self.input_size, self.hidden_size)
            self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
            self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
            self.fc4 = nn.Linear(self.hidden_size, self.output_size)
            self.fc_layer = [self.fc1,self.fc2,self.fc3,self.fc4]

    def forward(self, x):
        for j in range(self.num_layer):
            layer = self.fc_layer[j]
            x = F.relu(layer(x))
            x = self.drop_layer(x)

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # out = self.fc3(x)
        return x


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2,do_para =0):
        super(LSTMRegressor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout =do_para)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).to(x.device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.linear(out)
        return out


class CNN2D(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_size=64, dropout=0.5, kernel_size=3, padding=1):
        super(CNN2D, self).__init__()

        # Define convolutional layers
        self.conv1 = nn.Conv2d(input_channels, hidden_size, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size*2, kernel_size=kernel_size, padding=padding)
        self.hidden_size = hidden_size
        # Define fully connected layers
        # if kernel_size==2:
        #     self.fc1 = nn.Linear(8,  hidden_size * 4)
        # elif kernel_size ==1:
        #     self.fc1 = nn.Linear(6,  hidden_size * 4)
        self.fc1 = None
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_channels)


        # Define activation function and dropout for regularization
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = x.reshape(x.shape[0],1,10,2)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))

        # Flatten the output from the convolutional layers
        x = x.view(x.size(0), -1)
        if self.fc1 is None:
            n_features = x.numel() // x.shape[0]
            self.fc1 = nn.Linear(n_features, self.hidden_size*2).to(x.device)

        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize network with the input size, hidden layer size, and number of output classes
# model = MLP(input_size=20, hidden_size=276, output_size=20) # 87788
# model = LSTMRegressor(input_dim=2, hidden_dim=84, output_dim=2, num_layers=2) #  86858

def bl_train():

    # run = wandb.init(dir='C:/Users/yuhan/Downloads/wandb/')
    # model_name = run.name
    # lr = wandb.config.lr
    # batchsize = wandb.config.batch_size
    # drop_out_para = wandb.config.dropout
    # hidden_size = wandb.config.hidden_size

    if bl_name == 'mlp':

        # num_layer = wandb.config.num_layer
        # model = MLP(input_size=20, hidden_size=276, output_size=20)
        model = MLP(input_size=20, hidden_size=hidden_size,num_layer=num_layer, output_size=20,do_para = drop_out_para)

    elif bl_name == 'cnn2d':
        # kernel_size = wandb.config.kernel_size
        # padding = wandb.config.padding

        model = CNN2D(input_channels=1,
                      output_channels=20,
                      hidden_size=hidden_size,
                      dropout=drop_out_para,
                      kernel_size=kernel_size,
                      padding=padding)  # Modify these values based on your needs

    else:
        # num_layer = wandb.config.num_layer
        model = LSTMRegressor(input_dim=2,
                              hidden_dim=hidden_size,
                              output_dim=2,
                              num_layers=num_layer,
                              do_para=drop_out_para)

        # model = LSTMRegressor(input_dim=2, hidden_dim=84, output_dim=2, num_layers=2)
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # load dataset
    train_input_data = []
    train_output_data = []
    input_data = []
    output_data = []
    valid_input_data = []
    valid_output_data = []
    max_seq_length = 10

    # dataset_path = DATAROOT + '/cfg_0/'
    dataset_path = DATAROOT
    for cfg in range(5):
        for NUM_objects in range(10, 11):
            print('load data:', NUM_objects)
            raw_data = np.loadtxt(dataset_path + 'labels_after_%d/num_%d.txt' % (cfg,NUM_objects))

            train_data = raw_data[:int(len(raw_data) * 0.8)] * SCALE_DATA + SHIFT_DATA
            test_data = raw_data[int(len(raw_data) * 0.8):] * SCALE_DATA + SHIFT_DATA

            raw_data = raw_data[int(len(raw_data) * 0.8):]

            # if config.normalize_ :
            #     sub_d = (sub_d-np.min(sub_d,0))/(np.max(sub_d,0)-np.min(sub_d,0))
            train_input = []
            valid_input = []
            train_label = []
            valid_label = []
            for i in range(NUM_objects):
                train_input.append(train_data[:, i * 5 + 2:i * 5 + 4])
                valid_input.append(test_data[:, i * 5 + 2:i * 5 + 4])
                train_label.append(train_data[:, i * 5:i * 5 + 2])
                valid_label.append(test_data[:, i * 5:i * 5 + 2])

            train_input = np.asarray(train_input).transpose(1, 0, 2)
            valid_input = np.asarray(valid_input).transpose(1, 0, 2)
            train_label = np.asarray(train_label).transpose(1, 0, 2)
            valid_label = np.asarray(valid_label).transpose(1, 0, 2)

            train_input_data += list(train_input)
            train_output_data += list(train_label)
            valid_input_data += list(valid_input)
            valid_output_data += list(valid_label)

    train_input = pad_sequences(train_input_data,  max_seq_length=max_seq_length)
    train_label = pad_sequences(train_output_data, max_seq_length=max_seq_length)
    test_input =  pad_sequences(valid_input_data,  max_seq_length=max_seq_length)
    test_label =  pad_sequences(valid_output_data, max_seq_length=max_seq_length)

    train_dataset = CustomDataset(train_input, train_label)
    test_dataset = CustomDataset(test_input, test_label)
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

    num_data = (len(train_dataset), len(test_dataset))
    n_epochs = 10000
    train_loss_list = []
    valid_loss_list = []
    model.to(device)
    abort_learning = 0
    min_loss = np.inf
    log_pth = '10%s/result_r_%s/' % (bl_name, model_name)
    # model_pth = '../data/log_%s/' % model_name
    os.makedirs(log_pth, exist_ok=True)

    if train_flag:

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=200, verbose=True)
        criterion = nn.MSELoss()
        criterion_none_rdc = nn.MSELoss(reduction='none')

        for epoch in range(n_epochs):
            num_objects = np.random.randint(2,11)

            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                # Zero the gradients
                optimizer.zero_grad()
                inputs = torch.from_numpy(np.asarray(inputs, dtype=np.float32)).to(device)
                labels = torch.from_numpy(np.asarray(labels, dtype=np.float32)).to(device)

                inputs[:,2*num_objects:] = 0
                labels[:,2*num_objects:] = 0

                if bl_name == 'mlp':
                    inputs = inputs.flatten(1)
                    labels = labels.flatten(1)
                elif bl_name == 'lstm':
                    inputs = inputs.transpose(1, 0)
                    labels = labels.transpose(1, 0)
                else:
                    inputs = inputs.flatten(1)
                    labels = labels.flatten(1)
                # Forward pass
                outputs = model(inputs)
                # Calculate loss
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Update training loss
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)

            print(f"Epoch: {epoch + 1}/{n_epochs}.. Training Loss: {epoch_loss:.4f}",num_objects)


            # Validation phase
            model.eval()
            running_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = torch.from_numpy(np.asarray(inputs, dtype=np.float32)).to(device)
                    labels = torch.from_numpy(np.asarray(labels, dtype=np.float32)).to(device)

                    inputs[:, 2 * num_objects:] = 0
                    labels[:, 2 * num_objects:] = 0

                    if bl_name == 'mlp':
                        inputs = inputs.flatten(1)
                        labels = labels.flatten(1)
                    elif bl_name == 'lstm':
                        inputs = inputs.transpose(1, 0)
                        labels = labels.transpose(1, 0)
                    else:
                        inputs = inputs.flatten(1)
                        labels = labels.flatten(1)

                    outputs = model(inputs)

                    # Since MSE is used, make sure the labels are float type and unsqueeze it to have the same shape as outputs
                    labels = labels.float().unsqueeze(1)

                    loss = criterion(outputs, labels)

                    running_loss += loss.item() * inputs.size(0)

            avg_loss = running_loss / len(val_loader.dataset)

            if avg_loss < min_loss:
                min_loss = avg_loss
                PATH = log_pth + '/best_model.pt'
                torch.save(model.state_dict(), PATH)
                abort_learning = 0
            else:
                abort_learning += 1

            print(f"Epoch: {epoch + 1}/{n_epochs}.. Validation Loss: {avg_loss:.4f}",num_objects)
            scheduler.step(avg_loss)
            if abort_learning > 10:
                break

            wandb.log({
                'valid_loss': avg_loss
            })

    else:
        for num_objects in range(2, 11, 2):

            PATH = log_pth+'/best_model.pt'
            checkpoint = torch.load(PATH, map_location=device)
            model_dict = model.state_dict()
            partial_state_dict = {k : v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(partial_state_dict)
            model.load_state_dict(model_dict)

            # test phase
            model.eval()
            running_loss = 0.0
            with torch.no_grad():
                outputs_list = []
                loss_list = []
                for inputs, labels in val_loader:
                    inputs[:, num_objects:] = 0
                    labels[:, num_objects:] = 0

                    inputs = torch.from_numpy(np.asarray(inputs, dtype=np.float32)).to(device)
                    labels = torch.from_numpy(np.asarray(labels, dtype=np.float32)).to(device)

                    if bl_name == 'lstm':
                        inputs = inputs.transpose(1, 0)
                        labels = labels.transpose(1, 0)
                    else:
                        inputs = inputs.flatten(1)
                        labels = labels.flatten(1)

                    outputs = model(inputs)
                    outputs = (outputs - SHIFT_DATA) / SCALE_DATA
                    labels  = (labels - SHIFT_DATA) / SCALE_DATA

                    if bl_name == 'lstm':
                        labels = labels.transpose(1, 0)
                        labels = labels.reshape(len(labels), -1)
                        outputs = outputs.transpose(1, 0)
                        outputs = outputs.reshape(len(outputs), -1)
                    else:
                        labels [:, 2 * num_objects:] = 0
                        outputs[:, 2 * num_objects:] = 0
                        outputs[:, 2 * num_objects:] = 0

                    # loss_log = criterion_none_rdc(outputs, labels)
                    loss_log = abs(outputs - labels)
                    # Since MSE is used, make sure the labels are float
                    # type and unsqueeze it to have the same shape as outputs

                    # labels = labels.float().unsqueeze(1)
                    # loss = criterion(outputs, labels)
                    # running_loss += loss.item() * inputs.size(0)
                    loss_log = loss_log.cpu().detach().numpy()
                    outputs_list.append(outputs.cpu().detach().numpy())
                    loss_list.append(loss_log)

                outputs_list = np.concatenate(outputs_list)
                loss_list = np.concatenate(loss_list)
                outputs_list = (outputs_list.reshape(-1, len(outputs_list[0])) - SHIFT_DATA) / SCALE_DATA

                # for i in range(num_objects):
                #     raw_data[:, i * 5:i * 5 + 2] = outputs_list[:, i * 2:i * 2 + 2]
                #     raw_data[:, i * 5 + 4] = 0

                # np.savetxt(log_pth + 'num_%d.csv' % num_objects, raw_data)

                np.savetxt(log_pth + 'loss_list_num_%d.csv' % num_objects, np.asarray(loss_list))

if __name__ == '__main__':
    sweep_configuration_mlp = {
        'method': 'random',
        'name': 'sweep',
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss'
            },
        'parameters': {
            'batch_size': {'values': [256,512,1024]},
            'lr': {'max': 1e-3, 'min': 1e-5},
            'hidden_size': {'max': 300, 'min':200},
            'num_layer': {'max': 4, 'min': 2},
            'dropout': {'max': 0.2, 'min': 0.}
         }
    }

    sweep_configuration_lstm = {
        'method': 'random',
        'name': 'sweep',
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss'
            },
        'parameters': {
            'batch_size': {'values': [256, 512, 1024]},
            'lr': {'max': 1e-3, 'min': 1e-5},
            'hidden_size': {'max': 100, 'min':50},
            'num_layer': {'max': 3, 'min': 2},
            'dropout': {'max':0.2, 'min':0.}
         }
    }

    sweep_configuration_cnn2d = {
        'method': 'random',
        'name': 'sweep',
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss'
        },
        'parameters': {
            'batch_size': {'values': [256, 512, 1024]},
            'lr': {'max': 1e-3, 'min': 1e-5},
            'hidden_size': {'max': 100, 'min': 80},
            'kernel_size': {'max': 2, 'min': 1},
            'dropout': {'max':0.2, 'min':0.},
            'padding': {'max': 2, 'min': 1}
         }
    }

    # bl_name = 'cnn2d'
    # bl_name = 'lstm'
    bl_name = 'mlp'
    train_flag = False

    # sweep_id = wandb.sweep(sweep=sweep_configuration_mlp, project="knolling_baseline_%s" % bl_name)
    #
    # wandb.agent(sweep_id, function=bl_train, count=10)


    name_list = ['hearty-sweep-10', 'earnest-sweep-9','dazzling-sweep-8', 'splendid-sweep-7',
            'jolly-sweep-6', 'glowing-sweep-5', 'feasible-sweep-4', 'kind-sweep-3', 'ethereal-sweep-2', 'sweepy-sweep-1']

    # name_list =['fiery-sweep-10','apricot-sweep-9','helpful-sweep-8','generous-sweep-7','stoic-sweep-6',
    #             'fearless-sweep-5','effortless-sweep-4','amber-sweep-3','clean-sweep-2','solar-sweep-1']

#     name_list = ['swept-sweep-10','deep-sweep-9','jumping-sweep-8','rural-sweep-7','flowing-sweep-6',
# 'northern-sweep-5', 'fresh-sweep-4','trim-sweep-3','whole-sweep-2','light-sweep-1']

    # knolling_baseline, knolling_baseline_lstm, knolling_baseline_cnn2d
    api = wandb.Api()
    runs = api.runs('knolling_baseline_%s' % bl_name)
    para_list = []
    for n in range(9,10):
        model_name = name_list[n]
        for run in runs:
            # print(run.name)
            if run.name == model_name:
                print('found:',model_name)
                config = {k:v for k, v in run.config.items() if not k.startswith('_')}

        config = argparse.Namespace(**config)
        batchsize = 1024

        if bl_name == 'cnn2d':
            kernel_size = config.kernel_size
            padding = config.padding

        hidden_size = config.hidden_size
        num_layer = config.num_layer
        drop_out_para = config.dropout


        para = bl_train()
        # para_list.append(para)
    # np.savetxt('para_list_%s10.csv'%bl_name,para_list)


