import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import Transformer
from torch.utils.data import Dataset, DataLoader
import math
import torch.optim as optim
from pos_encoder import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# input min&max: [0.016, 0.048]
# label min&max: [-0.14599999962002042, 0.294500007390976]
# input_min,input_max = 0.016,0.048
# label_min,label_max = -0.14599999962002042, 0.294500007390976
SCALE_DATA = 100
SHIFT_DATA = 50
DATAROOT = "C:/Users/yuhan/Downloads/learning_data_512_large/"

def pad_sequences(sequences, max_seq_length=10, pad_value=0):
    padded_sequences = []
    for i in tqdm(range(len(sequences))):
        seq = sequences[i]
        if len(seq) < max_seq_length:
            padding_length = max_seq_length - len(seq)
            padded_seq = list(seq) + [[pad_value] * 2 for _ in range(padding_length)]
            padded_sequences.append(padded_seq)
        else:
            padded_sequences.append(seq)

    padded_sequences = np.asarray(padded_sequences)
    return padded_sequences


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.output_data[idx]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# def debug_posencoding():
    # positional_encoding = PositionalEncoding(d_model = 2)
    # x = torch.arange(11,21).repeat(10, 1).reshape(-1,10,2)
    # x = x.transpose(1,0)
    # print(x)
    # x = positional_encoding(x)
    # print(x)
    # quit()

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()

        self.attention = nn.MultiheadAttention(embed_dim=embed_size,
                                               num_heads=heads, batch_first=False)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        atten_outputs, atten_output_weights = self.attention(value, key, query)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(atten_outputs + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
            self,
            input_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size

        self.layers = nn.ModuleList(
            [TransformerBlock(
                embed_size,
                heads,
                dropout=dropout,
                forward_expansion=forward_expansion)
            ] * num_layers
        )

        self.dropout = nn.Dropout(dropout)
        self.fc_in = nn.Linear(input_size,embed_size)

    def forward(self, x):
        x = self.fc_in(x)
        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            x = layer(x, x, x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout,device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        # self.attention = SelfAttention(embed_size, heads=heads)
        self.attention = nn.MultiheadAttention(embed_dim=embed_size,
                                               num_heads=heads, batch_first=False)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key):
        atten_outputs, atten_output_weights = self.attention(x, x, x)
        query = self.dropout(self.norm(atten_outputs + x))
        out = self.transformer_block(value, key, query)
        return out


class Decoder(nn.Module):
    def __init__(
            self,
            input_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            embed_size,
    ):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout,device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.activate_F = nn.ReLU()
        self.fc_in = nn.Linear(input_size, embed_size)

    def forward(self, enc_out,x):
        x = self.fc_in(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_out, enc_out)
        out = self.fc_out(x)
        return out


class ProximityAwareLoss(nn.Module):
    def __init__(self, primary_loss_fn, min_distance):
        super(ProximityAwareLoss, self).__init__()
        self.primary_loss_fn = primary_loss_fn
        self.min_distance = min_distance

    def forward(self, predictions, targets):
        # Primary loss (e.g., mean squared error)
        primary_loss = self.primary_loss_fn(predictions, targets)

        # Proximity penalty
        n_objects = predictions.shape[0]
        proximity_penalty = 0

        for i in range(n_objects):
            for j in range(i+1, n_objects):
                distance = torch.norm(predictions[i] - predictions[j])
                if distance < self.min_distance:
                    proximity_penalty += (self.min_distance - distance) ** 2

        # Total loss
        total_loss = primary_loss + proximity_penalty
        return total_loss


class Knolling_Transformer(nn.Module):
    def __init__(
            self,
            input_length=10,
            input_size=2,
            output_size = 2,
            map_embed_d_dim=128,
            num_layers=6,
            forward_expansion=4,
            heads=2,
            dropout=0.,
            all_zero_target=0,
            pos_encoding_Flag = False,
            forwardtype = 0,
            high_dim_encoder=False,
            all_steps = False
    ):

        super(Knolling_Transformer, self).__init__()
        self.pos_encoding_Flag = pos_encoding_Flag
        self.all_zero_target = all_zero_target
        self.forwardtype = forwardtype
        self.high_dim_encoder = high_dim_encoder
        self.all_steps = all_steps


        self.positional_encoding = PositionalEncoding(d_model = input_size, max_len=input_length)

        n_freqs = 5
        self.position_encoder = PositionalEncoder(d_input = 2, n_freqs =n_freqs)

        if high_dim_encoder:
            input_size = input_size * (1 + 2 * n_freqs)

        self.encoder = Encoder(
            input_size,
            embed_size = map_embed_d_dim,
            num_layers = num_layers,
            heads = heads,
            forward_expansion = forward_expansion,
            dropout=dropout
        )

        self.decoder = Decoder(
            input_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            embed_size=map_embed_d_dim,
        )

        self.l1 = nn.Linear(map_embed_d_dim, map_embed_d_dim*2)
        self.l2 = nn.Linear(map_embed_d_dim*2, map_embed_d_dim)

        # self.bn1 = nn.BatchNorm1d(map_embed_d_dim)
        self.l_out = nn.Linear(map_embed_d_dim, output_size)
        self.acti = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tart_x_gt=None):

        if self.pos_encoding_Flag == True:
            x = self.positional_encoding(x)

        if self.high_dim_encoder:
            x = self.position_encoder(x)
            tart_x_gt_high = self.position_encoder(tart_x_gt)
        else:
            tart_x_gt_high = tart_x_gt

        # x_mask = torch.ones_like(x, dtype=torch.bool)
        # x_mask[:obj_num] = False
        # x.masked_fill_(x_mask, 0)

        enc_x = self.encoder(x)

        if self.forwardtype == 0:
            x = self.l1(enc_x)
            x = self.acti(x)
            x = self.l2(x)
            x = self.acti(x)

            out = self.l_out(x)

            # out = self.decoder(enc_x, tart_x_gt)
            # out = self.l_out(out)

        elif self.forwardtype == 2:
            out = self.decoder(enc_x, tart_x_gt_high)
            out = self.l_out(out)

        else:
            # Autoregressive decoding
            tart_x = torch.clone(tart_x_gt)
            # out = torch.zeros(enc_x.size(-1)).to(
            #     enc_x.device)  # Initialize with the correct shape and move to the same device as enc_x
            outputs = []
            results = 0
            obj_num = 10  # maximun 10
            for t in range(obj_num):
                tart_x_gt_high = self.position_encoder(tart_x)
                dec_output = self.decoder(enc_x, tart_x_gt_high)
                out = self.l_out(dec_output)
                if t == obj_num-1:
                    results = out
                out = out[t]
                outputs.append(out.unsqueeze(0))

                if t == 0:
                    tart_x = torch.cat((out.unsqueeze(0), tart_x[1:]), dim=0)

                elif t < tart_x.size(0):
                    tart_x = torch.cat((torch.cat(outputs), tart_x[t + 1:]), dim=0)

            if self.all_steps:
                return results

            elif obj_num < tart_x.size(0):
                pad_data_shape = outputs[0].shape

                outputs = outputs + [torch.zeros(pad_data_shape, device=device) for _ in range(tart_x.size(0) - obj_num)]
            out = torch.cat(outputs, dim=0)
            return out


    def maskedMSELoss(self, predictions, target, ignore_index = -100):
        mask = target.ne(ignore_index)
        mse_loss = (predictions - target).pow(2) * mask
        mse_loss = mse_loss.sum() / mask.sum()

        return mse_loss

if __name__ == "__main__":

    max_length = 10
    d_dim = 32
    layers_num = 4
    heads_num = 4
    print(d_dim,layers_num,heads_num)
    model = Knolling_Transformer(
        temperature= 1,
        input_length=max_length,
        input_size=2,
        map_embed_d_dim=d_dim,
        num_layers=layers_num,
        forward_expansion=4,
        heads=heads_num,
        dropout=0.0,
        all_zero_target=0,
        pos_encoding_Flag = True,
        forwardtype= 1,
        high_dim_encoder = True
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20, verbose=True) #tainloss


    BATCH_SIZE = 64
    EPOCHS = 10000
    NUM_objects = 10
    noise_std = 0.1
    partial_dataset = 1000
    dataset_path = DATAROOT + 'cfg_1/'
    raw_data = np.loadtxt(dataset_path + 'labels_after/num_%d.txt' % NUM_objects)[:partial_dataset]*SCALE_DATA+SHIFT_DATA
    # raw_data +=1
    train_data = raw_data[:int(len(raw_data) * 0.8)]
    test_data = raw_data[int(len(raw_data) * 0.8):]
    print('NUM OBJECT: ', NUM_objects)
    print('BATCH: ',    BATCH_SIZE)
    print('noise_std ', noise_std)
    print('partial data ', partial_dataset)


    train_input = []
    test_input = []
    train_label = []
    test_label = []
    for i in range(NUM_objects):
        train_input.append(train_data[:, i * 5 + 2:i * 5 + 4])
        test_input.append(test_data[:, i * 5 + 2:i * 5 + 4])
        train_label.append(train_data[:, i * 5:i * 5 + 2])
        test_label.append(test_data[:, i * 5:i * 5 + 2])


    train_input = np.asarray(train_input).transpose(1, 0, 2)
    test_input = np.asarray(test_input).transpose(1, 0, 2)
    train_label = np.asarray(train_label).transpose(1, 0, 2)
    test_label = np.asarray(test_label).transpose(1, 0, 2)

    train_input = pad_sequences(train_input, max_seq_length=max_length)
    test_input  = pad_sequences(test_input , max_seq_length=max_length)
    train_label = pad_sequences(train_label, max_seq_length=max_length)
    test_label  = pad_sequences(test_label , max_seq_length=max_length)

    train_dataset = CustomDataset(train_input, train_label)
    test_dataset = CustomDataset(test_input, test_label)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.to(device)
    # Training loop

    noise_std = torch.tensor(noise_std).to(device)

    for epoch in range(EPOCHS):
        prin_flag = True
        model.train()
        for input_batch, target_batch in train_loader:
            # Zero the gradients
            optimizer.zero_grad()

            input_batch = torch.from_numpy(np.asarray(input_batch, dtype=np.float32)).to(device)
            target_batch = torch.from_numpy(np.asarray(target_batch, dtype=np.float32)).to(device)
            input_batch = input_batch.transpose(1,0)
            target_batch = target_batch.transpose(1,0)

            # zero to False
            target_batch_atten_mask = (target_batch == 0).bool()
            target_batch.masked_fill_(target_batch_atten_mask, -100)

            # Mask the target:
            mask_point = np.random.randint(0, target_batch.size(0)+1)
            # mask_point = 0
            mask = torch.ones_like(target_batch, dtype=torch.bool)
            mask[:mask_point] = False
            input_target_batch = torch.clone(target_batch)

            # Add noise
            input_target_batch = torch.normal(input_target_batch, noise_std)
            input_batch = torch.normal(input_batch, noise_std)

            input_target_batch.masked_fill_(mask, -100)

            label_mask = torch.ones_like(target_batch, dtype=torch.bool)
            label_mask[:(mask_point+1)] = False
            target_batch.masked_fill_(label_mask, -100)

            # Forward pass

            output_batch = model(input_batch, input_target_batch)

            # Calculate loss
            loss = model.maskedMSELoss(output_batch, target_batch)

            if epoch %10 ==0 and prin_flag:
                # print('-')
                # print('output',output_batch[:,0].flatten())
                # print('target',target_batch[:,0].flatten())
                prin_flag = False

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()


        # Validation
        # model.eval()
        # validation_loss = 0.0
        # with torch.no_grad():
        #     for input_batch, target_batch in test_loader:
        #         input_batch = torch.from_numpy(np.asarray(input_batch, dtype=np.float32)).to(device)
        #         target_batch = torch.from_numpy(np.asarray(target_batch, dtype=np.float32)).to(device)
        #
        #         # Forward pass
        #         input_batch = input_batch.transpose(1, 0)
        #         target_batch = target_batch.transpose(1, 0)
        #         output_batch = model(input_batch)
        #
        #         # Calculate loss
        #         loss = criterion(output_batch, target_batch)
        #         validation_loss += loss.item()
        #
        # validation_loss /= len(test_loader)
        # print(f"Epoch {epoch + 1}/{EPOCHS}, Training Loss: {loss.item()}, Validation Loss: {validation_loss}")

        print(f"Epoch {epoch + 1}/{EPOCHS}, Training Loss: {loss.item()}")
        scheduler.step(loss.item())
