import os
from model import *
from torch.utils.data import DataLoader, random_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_model_batch(val_loader, model, log_path, num_obj=10):
    loss_function = torch.nn.MSELoss()

    model.to(device)
    model.eval()

    test_loss_list = []
    outputs = []

    with torch.no_grad():
        total_loss = 0
        for input_batch, target_batch in val_loader:
            input_batch = torch.from_numpy(np.asarray(input_batch, dtype=np.float32)).to(device)
            target_batch = torch.from_numpy(np.asarray(target_batch, dtype=np.float32)).to(device)
            input_batch = input_batch.transpose(1, 0)
            target_batch = target_batch.transpose(1, 0)

            # zero to False
            target_batch_atten_mask = (target_batch == 0).bool()
            target_batch.masked_fill_(target_batch_atten_mask, -100)

            # create all -100 input for decoder
            mask = torch.ones_like(target_batch, dtype=torch.bool)
            input_target_batch = torch.clone(target_batch)
            input_target_batch.masked_fill_(mask, -100)

            # Forward pass
            predictions = model(input_batch, obj_num=num_obj, tart_x_gt=input_target_batch)

            target_batch[num_obj:] = -100
            loss = model.maskedMSELoss(predictions, target_batch)

            print('output', predictions[:, 0].flatten())
            print('target', target_batch[:, 0].flatten())
            total_loss += loss.item()

            print('test_loss',loss)

            predictions = predictions.transpose(1, 0)
            target_batch = target_batch.transpose(1, 0)

            outputs.append(predictions.detach().cpu().numpy())

            numpy_pred =  (predictions.detach().cpu().numpy() - SHIFT_DATA) / SCALE_DATA
            numpy_label = (target_batch.detach().cpu().numpy() - SHIFT_DATA) / SCALE_DATA

            numpy_loss = (numpy_pred-numpy_label)**2
            numpy_loss = numpy_loss.reshape(len(numpy_loss),-1)
            numpy_loss[:,num_obj*2:] = 0

            # print('numpy_loss',numpy_loss)
            test_loss_list.append(numpy_loss)

    test_loss_list = np.concatenate(test_loss_list)
    outputs = np.concatenate(outputs)
    outputs = (outputs.reshape(-1, len(outputs[0]) * 2) - SHIFT_DATA) / SCALE_DATA
    np.savetxt(log_path + '/test_loss_list_num_%d.csv'%num_obj, np.asarray(test_loss_list))
    # np.savetxt(log_path + '/outputs.csv', outputs)

    return outputs, test_loss_list


if __name__ == '__main__':
    import wandb
    import argparse

    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs("robotics/knolling")

    name = "cosmic-serenity-151"
    # model_name = 'latest_model.pt'
    model_name = "best_model.pt"

    summary_list, config_list, name_list = [], [], []
    config = None
    for run in runs:
        if run.name == name:
            print("found: ", name)
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}
    print(config)
    config = argparse.Namespace(**config)
    NORM = True

    # load dataset
    train_input_data = []
    train_output_data = []
    input_data = []
    output_data = []
    valid_input_data = []
    valid_output_data = []
    cfg = 4
    dataset_path = DATAROOT + '/cfg_%d/' % cfg
    # dataset_path = DATAROOT + '/cfg_%d_521/' % cfg


    raw_data = 0
    for NUM_objects in range(10, 11):
        print('load data:', NUM_objects)
        raw_data = np.loadtxt(dataset_path + 'labels_after/num_%d.txt' % NUM_objects)

        # raw_data = np.loadtxt(dataset_path + 'real_before/num_%d_d9.txt' % NUM_objects)
        # if len(raw_data[0]) != 50:
        #     raw_data = np.hstack((raw_data,np.zeros((len(raw_data),50-len(raw_data[0])))))


        raw_data = raw_data[int(len(raw_data) * 0.8):]
        test_data = raw_data * SCALE_DATA + SHIFT_DATA
        valid_input = []
        valid_label = []
        for i in range(NUM_objects):
            valid_input.append(test_data[:, i * 5 + 2:i * 5 + 4])
            valid_label.append(test_data[:, i * 5:i * 5 + 2])

        valid_input = np.asarray(valid_input).transpose(1, 0, 2)
        valid_label = np.asarray(valid_label).transpose(1, 0, 2)

        valid_input_data += list(valid_input)
        valid_output_data += list(valid_label)

    test_input_padded = pad_sequences(valid_input_data, max_seq_length=config.max_seq_length)
    test_label_padded = pad_sequences(valid_output_data, max_seq_length=config.max_seq_length)

    test_dataset = CustomDataset(test_input_padded, test_label_padded)

    val_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    model = Knolling_Transformer(
        input_length=config.max_seq_length,
        input_size=2,
        map_embed_d_dim=config.map_embed_d_dim,
        num_layers=config.num_layers,
        forward_expansion=config.forward_expansion,
        heads=config.num_attention_heads,
        dropout=config.dropout_prob,
        # all_zero_target=config.all_zero_target,
        pos_encoding_Flag=config.pos_encoding_Flag,
        forwardtype=1,
        high_dim_encoder=config.high_dim_encoder,
        all_steps= True
    )

    # Number of parameters: 87458
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    PATH = 'data/log_%s/%s' % (name, model_name)
    checkpoint = torch.load(PATH, map_location=device)
    model.load_state_dict(checkpoint)

    log_path = 'results/%s' % name
    os.makedirs(log_path, exist_ok=True)
    # NUM_objects = 10
    for NUM_objects in range(10,11):
        outputs, loss_list = test_model_batch(val_loader, model, log_path, num_obj=NUM_objects)

        for i in range(NUM_objects):
            raw_data[:, i * 5:i * 5 + 2] = outputs[:, i * 2:i * 2 + 2]
            raw_data[:, i * 5 + 4] = 0
        log_folder = 'results/%s/cfg_%d/pred_after/' % (name, cfg)
        os.makedirs(log_folder, exist_ok=True)
        print(log_folder)
        np.savetxt(log_folder + '/num_%d.txt' % NUM_objects, raw_data)
