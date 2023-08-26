from datetime import datetime
import os
from new_model import *

if __name__ == '__main__':
    import wandb

    wandb.init(project='knolling_multi')  #,mode = 'disabled'
    config = wandb.config

    running_name = wandb.run.name

    config.forwardtype = 1
    config.map_embed_d_dim = 32
    config.num_attention_heads = 4
    config.num_layers = 4
    config.dropout_prob = 0.0
    config.max_seq_length = 5
    config.lr = 1e-4
    config.batch_size = 512
    config.log_pth = 'data/%s/' % running_name
    config.noise_std = 0.
    config.pos_encoding_Flag = True
    config.all_zero_target = 0  # 1 tart_x = zero like, 0: tart_x = tart_x
    config.forward_expansion = 4
    config.pre_trained = False
    config.high_dim_encoder = True
    config.all_steps = True
    config.object_num = -1
    os.makedirs(config.log_pth, exist_ok=True)

    model = Knolling_Transformer(
        input_length=config.max_seq_length,
        input_size=2,
        map_embed_d_dim=config.map_embed_d_dim,
        num_layers=config.num_layers,
        forward_expansion=config.forward_expansion,
        heads=config.num_attention_heads,
        dropout=config.dropout_prob,
        all_zero_target=config.all_zero_target,
        pos_encoding_Flag=config.pos_encoding_Flag,
        forwardtype=config.forwardtype,
        high_dim_encoder=config.high_dim_encoder,
        all_steps = config.all_steps,
        max_obj_num = 5,
        num_gaussians = 5)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config.model_params = num_params

    if config.pre_trained:
        pre_name = 'light-grass-6'
        PATH = 'data/%s/best_model.pt' % pre_name
        checkpoint = torch.load(PATH, map_location=device)
        model.load_state_dict(checkpoint)

    # load dataset
    train_input_data = []
    train_output_data = []
    input_data = []
    output_data = []
    valid_input_data = []
    valid_output_data = []

    DATA_CUT = 500000

    for cfg in range(5):
        dataset_path = DATAROOT + 'labels_after_%d/' % cfg
        for NUM_objects in range(5,6):
            print('load data:', NUM_objects)

            raw_data = np.loadtxt(dataset_path + 'num_%d.txt' %NUM_objects)[:DATA_CUT]
            raw_data = raw_data * SCALE_DATA + SHIFT_DATA

            train_data = raw_data[:int(len(raw_data) * 0.8)]
            test_data = raw_data[int(len(raw_data) * 0.8):]

            train_input = []
            valid_input = []
            train_label = []
            valid_label = []
            for i in range(NUM_objects):
                train_input.append(train_data[:, i * 5 + 2:i * 5 + 4])
                valid_input.append(test_data [:, i * 5 + 2:i * 5 + 4])
                train_label.append(train_data[:, i * 5:i * 5 + 2])
                valid_label.append(test_data [:, i * 5:i * 5 + 2])

            train_input = np.asarray(train_input).transpose(1, 0, 2)
            valid_input = np.asarray(valid_input).transpose(1, 0, 2)
            train_label = np.asarray(train_label).transpose(1, 0, 2)
            valid_label = np.asarray(valid_label).transpose(1, 0, 2)

            train_input_data += list(train_input)
            train_output_data += list(train_label)
            valid_input_data += list(valid_input)
            valid_output_data += list(valid_label)

    train_input_padded = pad_sequences(train_input_data, max_seq_length=config.max_seq_length)
    train_label_padded = pad_sequences(train_output_data, max_seq_length=config.max_seq_length)
    test_input_padded = pad_sequences(valid_input_data, max_seq_length=config.max_seq_length)
    test_label_padded = pad_sequences(valid_output_data, max_seq_length=config.max_seq_length)

    train_dataset = CustomDataset(train_input_padded, train_label_padded)
    test_dataset = CustomDataset(test_input_padded, test_label_padded)

    config.num_data = (len(train_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2000, verbose=True)

    num_epochs = 10000
    train_loss_list = []
    valid_loss_list = []
    model.to(device)
    abort_learning = 0
    min_loss = np.inf
    noise_std = torch.tensor(config.noise_std).to(device)
    for epoch in range(num_epochs):
        print_flag = True
        model.train()
        train_loss = 0

        if config.forwardtype == 2:
            # object_num = np.random.randint(0, target_batch.size(0) + 1)
            # object number > masked number
            object_num = config.object_num

            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()

                input_batch = torch.from_numpy(np.asarray(input_batch, dtype=np.float32)).to(device)
                target_batch = torch.from_numpy(np.asarray(target_batch, dtype=np.float32)).to(device)
                input_batch = input_batch.transpose(1, 0)
                target_batch = target_batch.transpose(1, 0)

                # zero to False
                target_batch_atten_mask = (target_batch == 0).bool()
                target_batch.masked_fill_(target_batch_atten_mask, -100)

                # Mask the target:
                mask_point = np.random.randint(0, object_num + 1)

                mask = torch.ones_like(target_batch, dtype=torch.bool)
                mask[:mask_point] = False
                input_target_batch = torch.clone(target_batch)

                # Add noise
                # input_target_batch = torch.normal(input_target_batch, noise_std)
                input_batch = torch.normal(input_batch, noise_std)

                input_target_batch.masked_fill_(mask, -100)

                label_mask = torch.ones_like(target_batch, dtype=torch.bool)
                label_mask[:(mask_point + 1)] = False
                target_batch.masked_fill_(label_mask, -100)

                # Forward pass
                # object number > masked number
                output_batch = model(input_batch, obj_num=object_num, tart_x_gt=input_target_batch)

                # Calculate loss
                loss = model.maskedMSELoss(output_batch, target_batch)

                if epoch % 10 == 0 and print_flag:
                    print('output', output_batch[:, 0].flatten())
                    print('target', target_batch[:, 0].flatten())
                    print_flag = False

                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss = train_loss / len(train_loader)
            train_loss_list.append(train_loss)
            # Validate
            model.eval()
            print_flag = True
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

                    # Mask the target:
                    mask_point = np.random.randint(0, object_num + 1)

                    mask = torch.ones_like(target_batch, dtype=torch.bool)
                    mask[:mask_point] = False
                    input_target_batch = torch.clone(target_batch)
                    input_target_batch.masked_fill_(mask, -100)

                    label_mask = torch.ones_like(target_batch, dtype=torch.bool)
                    label_mask[:(mask_point + 1)] = False
                    target_batch.masked_fill_(label_mask, -100)

                    # Forward pass
                    output_batch = model(input_batch, obj_num=object_num, tart_x_gt=input_target_batch)

                    # Calculate loss
                    loss = model.maskedMSELoss(output_batch, target_batch)

                    if epoch % 10 == 0 and print_flag:
                        print('val_output', output_batch[:, 0].flatten())
                        print('val_target', target_batch[:, 0].flatten())
                        print_flag = False

                    total_loss += loss.item()
                avg_loss = total_loss / len(val_loader)
                scheduler.step(avg_loss)

                train_loss_list.append(avg_loss)

                if avg_loss < min_loss:
                    min_loss = avg_loss
                    PATH = config.log_pth + '/best_model.pt'
                    torch.save(model.state_dict(), PATH)
                    abort_learning = 0
                else:
                    abort_learning += 1

                if epoch % 100 == 0:
                    torch.save(model.state_dict(), config.log_pth + '/latest_model.pt')

                print(f"{datetime.now()}Epoch {epoch},train loss: {train_loss}, validation loss: {avg_loss},"
                      f" no_improvements: {abort_learning}")

                wandb.log({"train loss": train_loss,
                           "valid loss": avg_loss,
                           "learning rate": optimizer.param_groups[0]['lr'],
                           "min loss": min_loss})

        else:
            # # object number > masked number
            # if config.object_num == -1:
            #     object_num = np.random.randint(5, 11) #[0,11)
            # else:
            #     object_num = config.object_num

            # print("OBJ:", object_num)
            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()

                input_batch = torch.from_numpy(np.asarray(input_batch, dtype=np.float32)).to(device)
                target_batch = torch.from_numpy(np.asarray(target_batch, dtype=np.float32)).to(device)
                input_batch = input_batch.transpose(1, 0)
                target_batch = target_batch.transpose(1, 0)

                # # shuffle input data
                # idx = torch.randperm(input_batch.shape[0])
                # input_batch  = input_batch[idx]
                # target_batch = target_batch[idx]

                # zero to False
                # input_batch_atten_mask = (input_batch == 0).bool()
                # input_batch = torch.normal(input_batch, noise_std)  ## Add noise
                # input_batch.masked_fill_(input_batch_atten_mask, -100)

                target_batch_atten_mask = (target_batch == 0).bool()
                target_batch.masked_fill_(target_batch_atten_mask, -100)

                # create all -100 input for decoder
                mask = torch.ones_like(target_batch, dtype=torch.bool)
                input_target_batch = torch.clone(target_batch)

                # input_target_batch = torch.normal(input_target_batch, noise_std)
                input_target_batch.masked_fill_(mask, -100)

                # label_mask = torch.ones_like(target_batch, dtype=torch.bool)
                # label_mask[:object_num] = False
                # target_batch.masked_fill_(label_mask, -100)

                # Forward pass
                # object number > masked number
                output_batch = model(input_batch,
                                     # obj_num=object_num,
                                     tart_x_gt=input_target_batch)

                # Calculate loss
                loss = model.maskedMSELoss(output_batch, target_batch)

                if epoch % 10 == 0 and print_flag:
                    print('output', output_batch[:, 0].flatten())
                    print('target', target_batch[:, 0].flatten())
                    print_flag = False

                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss = train_loss / len(train_loader)
            train_loss_list.append(train_loss)
            # Validate
            model.eval()
            print_flag = True
            with torch.no_grad():
                total_loss = 0
                for input_batch, target_batch in val_loader:
                    input_batch = torch.from_numpy(np.asarray(input_batch, dtype=np.float32)).to(device)
                    target_batch = torch.from_numpy(np.asarray(target_batch, dtype=np.float32)).to(device)
                    input_batch = input_batch.transpose(1, 0)
                    target_batch = target_batch.transpose(1, 0)

                    # # zero to False
                    # input_batch_atten_mask = (input_batch == 0).bool()
                    # input_batch = torch.normal(input_batch, noise_std)  ## Add noise
                    # input_batch.masked_fill_(input_batch_atten_mask, -100)

                    target_batch_atten_mask = (target_batch == 0).bool()
                    target_batch.masked_fill_(target_batch_atten_mask, -100)

                    # create all -100 input for decoder
                    mask = torch.ones_like(target_batch, dtype=torch.bool)
                    input_target_batch = torch.clone(target_batch)

                    # input_target_batch = torch.normal(input_target_batch, noise_std)
                    input_target_batch.masked_fill_(mask, -100)

                    # label_mask = torch.ones_like(target_batch, dtype=torch.bool)
                    # label_mask[:object_num] = False
                    # target_batch.masked_fill_(label_mask, -100)

                    # Forward pass
                    # object number > masked number
                    output_batch = model(input_batch,
                                         # obj_num=object_num,
                                         tart_x_gt=input_target_batch)


                    # Calculate loss
                    loss = model.maskedMSELoss(output_batch, target_batch)

                    if epoch % 10 == 0 and print_flag:
                        print('val_output', output_batch[:, 0].flatten())
                        print('val_target', target_batch[:, 0].flatten())
                        print_flag = False

                    total_loss += loss.item()
                avg_loss = total_loss / len(val_loader)
                scheduler.step(avg_loss)

                # train_loss_list.append(avg_loss)
                valid_loss_list.append(avg_loss)

                if avg_loss < min_loss:
                    min_loss = avg_loss
                    PATH = config.log_pth + '/best_model.pt'
                    torch.save(model.state_dict(), PATH)
                    abort_learning = 0
                else:
                    abort_learning += 1

                if epoch % 100 == 0:
                    torch.save(model.state_dict(), config.log_pth + '/latest_model.pt')

                print(f"{datetime.now()}Epoch {epoch},train loss: {train_loss}, validation loss: {avg_loss},"
                      f" no_improvements: {abort_learning}")

                wandb.log({"train loss": train_loss,
                           "valid loss": avg_loss,
                           "learning rate": optimizer.param_groups[0]['lr'],
                           "min loss": min_loss})
