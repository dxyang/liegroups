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

from meshcat_viewer import get_visualizer, draw_point_cloud, draw_transform


def replay_sequence(subject_num: int, split: str, dbg: bool = True):
    # bookkeeping things
    # experiment_name = "07262022_fulldset_overfit"
    experiment_name = "07272022_fulldset_splits"
    # experiment_name = "07282022_fulldset_splits_30e"
    output_directory = os.path.expanduser("~/localdata/cam2hand")
    exp_folder = f"{output_directory}/{experiment_name}"
    assert split in ["train", "test", "val"]
    replay_file = f"{exp_folder}/replay/{split}/s{subject_num}.pkl"

    # viz things
    viz = get_visualizer(zmq_url="tcp://127.0.0.1:6001")
    viz.delete()

    all_action_trajs = pickle.load(open(replay_file, 'rb'))

    actions = [a for a in all_action_trajs.keys()]
    random.shuffle(actions)
    action_idx = 0
    while action_idx < len(actions):
        action = actions[action_idx]
        traj = all_action_trajs[action]
        seq_idxs = list(traj['seq_idx'].numpy())
        T_world_cams = traj['T_world_cam'].numpy()
        T_cam_lhands = traj['T_cam_lhand'].numpy()
        T_cam_rhands = traj['T_cam_rhand'].numpy()
        T_cam_lhandHats = traj['T_cam_lhandHat'].numpy()
        T_cam_rhandHats = traj['T_cam_rhandHat'].numpy()

        print(f"s{subject_num}: {action}")

        # maybe this works?
        min_idx = np.min(seq_idxs)
        max_idx = np.max(seq_idxs)

        for t in range(min_idx, max_idx + 1):
            if t not in seq_idxs:
                print(f"index {t} not in image dataset")
                continue

            idx = seq_idxs.index(t)

            T_world_lhandHat = np.matmul(T_world_cams[idx], T_cam_lhandHats[idx])
            T_world_rhandHat = np.matmul(T_world_cams[idx], T_cam_rhandHats[idx])
            T_world_lhand = np.matmul(T_world_cams[idx], T_cam_lhands[idx])
            T_world_rhand = np.matmul(T_world_cams[idx], T_cam_rhands[idx])

            draw_transform(viz, "cam", T_world_cams[idx], linewidth=15, length=0.1)
            draw_transform(viz, "lhandhat", T_world_lhandHat, linewidth=10, length=0.25)
            draw_transform(viz, "rhandhat", T_world_rhandHat, linewidth=10, length=0.25)
            draw_transform(viz, "lhand", T_world_lhand)
            draw_transform(viz, "rhand", T_world_rhand)

        if dbg:
            res = input(f"done with {action}, continue? (y/n/pdb)")
            if res == 'n':
                continue
            elif res == 'pdb':
                import pdb; pdb.set_trace()
            else:
                action_idx += 1
        else:
            action_idx += 1

if __name__ == "__main__":
    split = "test"
    s_idx = 1
    replay_sequence(subject_num=s_idx, split=split)

