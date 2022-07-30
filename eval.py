from typing import Dict, List, Union

import copy
import math
import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from pytorch3d.transforms import quaternion_to_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from tqdm import tqdm
import visdom

from data import Cam2HandDset, get_unnormalize_transform
from loss import position_loss, rotation_loss, convert_yhat_to_transforms
from models import PoseRegressor
from rotation_learning import A_vec_to_quat
from viz import VisdomVisualizer, PlotlyScene, plot_transform, generate_plotly_loss_figure

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# bookkeeping things
# experiment_name = "07262022_fulldset_overfit"
# experiment_name = "07272022_fulldset_splits"
experiment_name = "07282022_fulldset_splits_30e"
output_directory = os.path.expanduser("~/localdata/cam2hand")
exp_folder = f"{output_directory}/{experiment_name}"
replay_dir = f"{exp_folder}/replay"
splits = ["train", "val", "test"]
for split in splits:
    split_folder = f"{replay_dir}/{split}"
    if not os.path.exists(split_folder):
        os.makedirs(split_folder)
loss_stats = {}

# dataset
batch_size = 256
dset_path = os.path.expanduser("~/localdata/cam2hand_dset")

# model
trained_model_path = f"{exp_folder}/models/poseregressor.pth"
model = PoseRegressor(output_size=26)
model.load_model(trained_model_path)
model.to(device)
model.eval()


def ordered_unique(lst: list):
    ret = {}
    for val in lst:
        if val in ret:
            continue
        else:
            ret[val] = True
    return list(ret.keys())

# datasets
train_dset = Cam2HandDset(dset_path, data_split="train")
val_dset = Cam2HandDset(dset_path, data_split="val")
test_dset = Cam2HandDset(dset_path, data_split="test")
train_dataloader = DataLoader(train_dset, batch_size=batch_size, shuffle=False, num_workers=16)
val_dataloader = DataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=16)
test_dataloader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=16)
dataloaders = [train_dataloader, val_dataloader, test_dataloader]

# replay/evaluate
for split, dataloader in zip(splits, dataloaders):
    # storage for replayed data
    subj_action_dict = {}
    loss_stats[split] = {
        "pos_loss": [],
        "rot_loss": [],
        "total_loss": [],
    }
    split_folder = f"{replay_dir}/{split}"

    for batch_num, data_dict in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            x = data_dict["image"].to(device)
            y = data_dict["flattened_T_cam_hands"].to(device)
            batch_T_world_cam = data_dict["T_world_cam"]
            batch_T_cam_lhand = data_dict["T_cam_lhand"]
            batch_T_cam_rhand = data_dict["T_cam_rhand"]
            subject_str = data_dict["subject"]
            action_str = data_dict["action"]
            sequence_idx = data_dict["seq_idx"]
            batch_size = x.size()[0]

            y_hat = model(x)
            pos_loss, pos_info = position_loss(y, y_hat)
            rot_loss, _ = rotation_loss(y, y_hat)
            total_loss = pos_loss + rot_loss

            for k, v in pos_info.items():
                if k not in loss_stats[split]:
                    loss_stats[split][k] = [v]
                else:
                    loss_stats[split][k].append(v)

            loss_stats[split]["pos_loss"].append(pos_loss.item())
            loss_stats[split]["rot_loss"].append(rot_loss.item())
            loss_stats[split]["total_loss"].append(total_loss.item())

            T_cam_lhandHat, T_cam_rhandHat = convert_yhat_to_transforms(y_hat.detach().cpu())

        # bookkeeping
        subjects_in_batch = ordered_unique(subject_str)
        actions_in_batch = ordered_unique(action_str)

        if len(actions_in_batch) == 1:
            assert len(subjects_in_batch) == 1
            subj_str = subjects_in_batch[0]
            if subj_str not in subj_action_dict:
                subj_action_dict[subj_str] = {}

            action_str = actions_in_batch[0]
            if action_str not in subj_action_dict[subj_str]:
                replay_dict = {}
                replay_dict["T_world_cam"] = batch_T_world_cam
                replay_dict["T_cam_lhand"] = batch_T_cam_lhand
                replay_dict["T_cam_rhand"] = batch_T_cam_rhand
                replay_dict["T_cam_lhandHat"] = T_cam_lhandHat
                replay_dict["T_cam_rhandHat"] = T_cam_rhandHat
                replay_dict["seq_idx"] = sequence_idx
                subj_action_dict[subj_str][action_str] = replay_dict
            else:
                replay_dict = subj_action_dict[subj_str][action_str]
                replay_dict["T_world_cam"] = torch.cat([replay_dict["T_world_cam"], batch_T_world_cam], dim=0)
                replay_dict["T_cam_lhand"] = torch.cat([replay_dict["T_cam_lhand"], batch_T_cam_lhand], dim=0)
                replay_dict["T_cam_rhand"] = torch.cat([replay_dict["T_cam_rhand"], batch_T_cam_rhand], dim=0)
                replay_dict["T_cam_lhandHat"] = torch.cat([replay_dict["T_cam_lhandHat"], T_cam_lhandHat], dim=0)
                replay_dict["T_cam_rhandHat"] = torch.cat([replay_dict["T_cam_rhandHat"], T_cam_rhandHat], dim=0)
                replay_dict["seq_idx"] = torch.cat([replay_dict["seq_idx"], sequence_idx], dim=0)
        elif len(actions_in_batch) == 2:
            action_0 = actions_in_batch[0]
            action_1 = actions_in_batch[1]
            new_action_idx = action_str.index(actions_in_batch[1])
            if len(subjects_in_batch) == 1:
                subject_0 = subjects_in_batch[0]
                subject_1 = subject_0
                assert subject_0 in subj_action_dict
                assert action_0 in subj_action_dict[subject_0]
                assert action_1 not in subj_action_dict[subject_0]
            elif len(subjects_in_batch) == 2:
                subject_0 = subjects_in_batch[0]
                subject_1 = subjects_in_batch[1]
                assert subject_0 in subj_action_dict
                assert subject_1 not in subj_action_dict
                new_subj_idx = subject_str.index(subjects_in_batch[1])
                assert new_subj_idx == new_action_idx
                assert action_0 in subj_action_dict[subject_0]
            else:
                assert False

            replay_dict = subj_action_dict[subject_0][action_0]
            replay_dict["T_world_cam"] = torch.cat([replay_dict["T_world_cam"], batch_T_world_cam[:new_action_idx]], dim=0)
            replay_dict["T_cam_lhand"] = torch.cat([replay_dict["T_cam_lhand"], batch_T_cam_lhand[:new_action_idx]], dim=0)
            replay_dict["T_cam_rhand"] = torch.cat([replay_dict["T_cam_rhand"], batch_T_cam_rhand[:new_action_idx]], dim=0)
            replay_dict["T_cam_lhandHat"] = torch.cat([replay_dict["T_cam_lhandHat"], T_cam_lhandHat[:new_action_idx]], dim=0)
            replay_dict["T_cam_rhandHat"] = torch.cat([replay_dict["T_cam_rhandHat"], T_cam_rhandHat[:new_action_idx]], dim=0)
            replay_dict["seq_idx"] = torch.cat([replay_dict["seq_idx"], sequence_idx[:new_action_idx]], dim=0)

            replay_dict = {}
            replay_dict["T_world_cam"] = batch_T_world_cam[new_action_idx:]
            replay_dict["T_cam_lhand"] = batch_T_cam_lhand[new_action_idx:]
            replay_dict["T_cam_rhand"] = batch_T_cam_rhand[new_action_idx:]
            replay_dict["T_cam_lhandHat"] = T_cam_lhandHat[new_action_idx:]
            replay_dict["T_cam_rhandHat"] = T_cam_rhandHat[new_action_idx:]
            replay_dict["seq_idx"] = sequence_idx[new_action_idx:]
            if subject_1 not in subj_action_dict:
                subj_action_dict[subject_1] = {}
            subj_action_dict[subject_1][action_1] = replay_dict

        else:
            assert False # womp womp

    # save loss data for the whole dataset
    split_losses_path = f"{split_folder}/losses.pkl"
    pickle.dump(loss_stats[split], open(split_losses_path, 'wb'))
    print(f"saved losses dict to {split_losses_path}")
    for k, v in loss_stats[split].items():
        if "xyz" in k:
            xyz_error = np.mean(np.vstack(v), axis=0)
            print(f"     {k}: {xyz_error}")
        else:
            print(f"     {k}: {np.mean(v)}")

    # save per subject dict to disk
    for subject_num in range(1, 11):
        save_dict = subj_action_dict[f"s{subject_num}"]
        per_subject_trajs_path = f"{split_folder}/s{subject_num}.pkl"
        pickle.dump(save_dict, open(per_subject_trajs_path, 'wb'))
        print(f"saved trajectories to {per_subject_trajs_path}")
