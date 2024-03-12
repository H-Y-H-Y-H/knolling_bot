import os
import yaml
import numpy as np
from model_structure import *
from model_structure import *
from torch.utils.data import DataLoader, random_split

best_success_rate = 0
worst_success_rate = 100
best_index = 0
worst_index = 0
success_list = []


def test_model_batch(val_loader, model, log_path, selec_list, num_obj=10):

    model.to(device)
    model.eval()

    test_loss_list = []
    outputs = []
    ll_loss_list= []
    ms_min_sample_loss_list= []
    overlap_loss_list= []
    pos_loss_list= []
    v_entropy_loss_list= []

    with torch.no_grad():
        total_loss = 0

        for input_batch, target_batch in val_loader:
            input_batch = torch.from_numpy(np.asarray(input_batch, dtype=np.float32)).to(device)
            target_batch = torch.from_numpy(np.asarray(target_batch, dtype=np.float32)).to(device)
            input_batch = input_batch.transpose(1, 0)
            target_batch = target_batch.transpose(1, 0)

            # zero to False
            target_batch_atten_mask = (target_batch == 0).bool()
            target_batch.masked_fill_(target_batch_atten_mask, MASK_VALUE)

            # create all MASK_VALUE input for decoder
            mask = torch.ones_like(target_batch, dtype=torch.bool)
            input_target_batch = torch.clone(target_batch)
            input_target_batch.masked_fill_(mask, MASK_VALUE)

            # Forward pass.
            # if MIN_PRED:
            output_batch_min, pi, sigma, mu, ms_min_sample_loss = model.forward_min(input_batch,
                                                                            tart_x_gt=input_target_batch,
                                                                            gt_decoder=target_batch)
            ms_min_sample_loss = model.masked_MSE_loss(output_batch_min[:model.in_obj_num],
                                                       target_batch[:model.in_obj_num],Output_scaler=False).transpose(1, 0)

            # else:
            output_batch, _, _, _ = model.forward(input_batch,tart_x_gt=input_target_batch,given_idx = selec_list)


            output_batch_min = output_batch_min[: model.in_obj_num]
            output_batch = output_batch[: model.in_obj_num]
            # Calculate log-likelihood loss
            ll_loss = model.mdn_loss_function(pi, sigma, mu, target_batch[:model.in_obj_num],
                                              Output_scaler=False).transpose(1, 0)

            # Calculate collision loss
            if model.in_obj_num > 1:
                overlap_loss_sampled = calculate_collision_loss(output_batch.transpose(0, 1),
                                                                input_batch[: model.in_obj_num].transpose(0, 1),
                                                                scale=False,
                                                                Output_scaler=False)

                overlap_loss_min = calculate_collision_loss(output_batch_min.transpose(0, 1),
                                                            input_batch[: model.in_obj_num].transpose(0, 1),
                                                            scale=False,
                                                            Output_scaler=False)  # .transpose(1, 0)
                overlap_loss = overlap_loss_sampled + overlap_loss_min

            else:
                overlap_loss = torch.zeros((), device=device)

            # Calcluate position loss
            # pos_loss = model.masked_MSE_loss(output_batch, target_batch[:model.in_obj_num],Output_scaler=False).transpose(1, 0)
            pos_loss = model.masked_MSE_loss(output_batch, target_batch[:model.in_obj_num],Output_scaler=False).transpose(1, 0)

            # Calucluate Entropy loss:
            v_entropy_loss = entropy_loss(pi, Output_scaler=False).transpose(1, 0)

            ll_loss_list.append(ll_loss.detach().cpu().numpy())
            ms_min_sample_loss_list.append(ms_min_sample_loss.detach().cpu().numpy())
            overlap_loss_list.append(overlap_loss.detach().cpu().numpy())
            pos_loss_list.append(pos_loss.detach().cpu().numpy())
            v_entropy_loss_list.append(v_entropy_loss.detach().cpu().numpy())

            if MIN_PRED:
                outputs.append(output_batch_min.transpose(1, 0).detach().cpu().numpy())
            else:
                outputs.append(output_batch.transpose(1, 0).detach().cpu().numpy())
            k_min = 1
            k_ll = .0001
            k_op = .01
            k_pos = 0.01
            k_en = .0001
            loss = k_ll * ll_loss + ms_min_sample_loss*k_min + k_op * overlap_loss.unsqueeze(1) + k_pos * pos_loss \
                        + k_en*v_entropy_loss

            test_loss_list.append(loss.detach().cpu().numpy())

    test_loss_list = np.concatenate(test_loss_list)
    outputs = np.concatenate(outputs)

    ll_loss_list= np.concatenate(ll_loss_list)
    ms_min_sample_loss_list= np.concatenate(ms_min_sample_loss_list)
    overlap_loss_list= np.concatenate(overlap_loss_list)
    pos_loss_list= np.concatenate(pos_loss_list)
    v_entropy_loss_list= np.concatenate(v_entropy_loss_list)

    outputs = (outputs.reshape(-1, len(outputs[0]) * 2) - config.SHIFT_DATA) / config.SCALE_DATA
    np.savetxt(log_path + '/test_loss_list%d.csv' % num_obj, np.asarray(test_loss_list))
    np.savetxt(log_path + '/ll_loss%d.csv' % num_obj, ll_loss_list)
    np.savetxt(log_path + '/ms_min_sample_loss%d.csv' % num_obj, ms_min_sample_loss_list)
    np.savetxt(log_path + '/overlap_loss%d.csv' % num_obj, overlap_loss_list)
    np.savetxt(log_path + '/pos_loss%d.csv' % num_obj, pos_loss_list)
    np.savetxt(log_path + '/v_entropy_loss%d.csv' % num_obj, v_entropy_loss_list)

    return outputs, test_loss_list


if __name__ == '__main__':
    import wandb
    import argparse
    info_per_object = 7
    file_num = 10

    test_sweep_flag = False
    use_yaml = True

    # api = wandb.Api()
    # Project is specified by <entity/project-name>
    # runs = api.runs("knolling0205_2_overlap")

    method_list = ['flowing-moon-116', 'eternal-sweep-1', 'restful-sweep-1',
                   'dazzling-sweep-1', 'sage-sweep-1', 'magic-tree-145']

    name = method_list[5]

    model_name = "best_model.pt"

    with open(f'data/{name}/config.yaml', 'r') as yaml_file:
        config_dict = yaml.safe_load(yaml_file)
    config = {k: v for k, v in config_dict.items() if not k.startswith('_')}

    config = argparse.Namespace(**config)
    MIN_PRED = True


    # load the test dataset
    test_num_scenario = 20000
    solu_num = 12 # 12
    SHIFT_DATASET_ID = 0 # color 3,4,5


    for object_num in range(10,11):
        valid_lw_data = []
        valid_pos_data = []
        total_raw_data = []
        obj_name_list = []
        print('load data:', object_num)
        raw_data = np.loadtxt(DATAROOT + 'num_%d_after_%d.txt' % (file_num, SHIFT_DATASET_ID))[:,:object_num*info_per_object]
        obj_name_data = np.loadtxt(DATAROOT + 'num_%d_after_name_%d.txt' % (file_num, SHIFT_DATASET_ID), dtype=str)[:,:object_num]

        # raw_data = np.loadtxt('test_batch.txt')
        # np.savetxt('test_batch.txt',raw_data,fmt='%s')

        raw_data = raw_data[-test_num_scenario:]
        obj_name_data = obj_name_data[-test_num_scenario:]

        total_raw_data.append(raw_data)
        obj_name_list.append(obj_name_data)

        test_data = raw_data * config.SCALE_DATA + config.SHIFT_DATA
        valid_lw = []
        valid_pos = []

        for i in range(object_num):
            valid_lw.append(test_data[:, i * info_per_object + 2:i * info_per_object + 4])
            valid_pos.append(test_data[:, i * info_per_object:i * info_per_object + 2])

        valid_lw = np.asarray(valid_lw).transpose(1, 0, 2)
        valid_pos = np.asarray(valid_pos).transpose(1, 0, 2)

        valid_lw_data += list(valid_lw)
        valid_pos_data += list(valid_pos)

        test_input_padded = pad_sequences(valid_lw_data, max_seq_length=config.max_seq_length)
        test_label_padded = pad_sequences(valid_pos_data, max_seq_length=config.max_seq_length)

        test_dataset = CustomDataset(test_input_padded, test_label_padded)
        val_loader = DataLoader(test_dataset, batch_size=512, shuffle=False) # 不能用shuffle True，不然evaluate面积时对不上号


        model = Knolling_Transformer(
                input_length=config.max_seq_length,
                input_size=2,
                map_embed_d_dim=config.map_embed_d_dim,
                num_layers=config.num_layers,
                forward_expansion=config.forward_expansion,
                heads=config.num_attention_heads,
                dropout=config.dropout_prob,
                all_zero_target=config.all_zero_target,
                forwardtype=config.forwardtype,
                all_steps=config.all_steps,
                in_obj_num=object_num,
                num_gaussians=config.num_gaussian
            )

        # Number of parameters: 87458
        print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        PATH = 'data/%s/%s' % (name, model_name)
        checkpoint = torch.load(PATH, map_location=device)
        model.load_state_dict(checkpoint)

        if MIN_PRED:
            log_path = f'./results/{name}_min/{SHIFT_DATASET_ID}/'
        else:
            log_path = f'./results/{name}/{SHIFT_DATASET_ID}/'
        os.makedirs(log_path, exist_ok=True)

        raw_data = np.concatenate(total_raw_data)
        obj_name_list = np.concatenate(obj_name_list)


        np.savetxt(log_path + '/num_%d_gt.txt' % object_num, raw_data)
        np.savetxt(log_path+'/obj_name_%d.txt' % object_num, obj_name_list,fmt="%s")
        n_solu = 20 #config.num_gaussian**object_num
        m = config.num_gaussian
        n = object_num
        import random
        for id_solutions in range(n_solu):
            # selec_list = to_base_4(id_solutions,object_num,n_gaussian=config.num_gaussian)
            selec_list = random.choices(range(m), k=n)
            outputs, loss_list = test_model_batch(val_loader, model, log_path, num_obj=object_num,selec_list=selec_list)

            for i in range(object_num):
                raw_data[:, i * info_per_object:i * info_per_object + 2] = outputs[:, i * 2:i * 2 + 2]

            np.savetxt(log_path + f'/num_{object_num}_pred_{id_solutions}.txt', raw_data)


