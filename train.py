from typing import Dict, List, Union

import copy
import math
import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import visdom

from data import Cam2HandDset, get_unnormalize_transform
from loss import position_loss, rotation_loss, convert_yhat_to_transforms
from models import PoseRegressor
from viz import VisdomVisualizer, PlotlyScene, plot_transform, generate_plotly_loss_figure

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# bookkeeping things
# experiment_name = "07262022_fulldset_overfit"
# experiment_name = "07272022_fulldset_splits"
experiment_name = "07282022_fulldset_splits_30e"
output_directory = os.path.expanduser("~/localdata/cam2hand")
exp_folder = f"{output_directory}/{experiment_name}"
model_dir = f"{exp_folder}/models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# train
epochs = 30
batch_size = 256

# dataset
dset_path = os.path.expanduser("~/localdata/cam2hand_dset")
train_dset = Cam2HandDset(dset_path, data_split="train")
val_dset = Cam2HandDset(dset_path, data_split="val")
train_dataloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=16)
val_dataloader = DataLoader(val_dset, batch_size=batch_size, shuffle=True, num_workers=16)

# model
translation_output_size = 3 # x y z
rotation_output_size = 10 # per https://github.com/utiasSTARS/bingham-rotation-learning
output_size = 2 * (translation_output_size + rotation_output_size)
model = PoseRegressor(output_size=output_size)
model.to(device)

optimizer = optim.Adam(model.parameters())

# training visualization
vis = visdom.Visdom()
visualizer = VisdomVisualizer(vis, f"{experiment_name}")
unnormalize_transform = get_unnormalize_transform()
pbar = tqdm(range(epochs))
iteration = 0
loss_dict = {
    "train": {
        "iterations": [],
        "loss": [],
        "pos_loss": [],
        "rot_loss": [],
    },
    "val": {
        "iterations": [],
        "loss": [],
        "pos_loss": [],
        "rot_loss": [],
    },
}
for e in pbar:
    for data_dict in tqdm(train_dataloader):
        x = data_dict["image"].to(device)
        y = data_dict["flattened_T_cam_hands"].to(device)
        batch_T_world_cam = data_dict["T_world_cam"].to(device)
        batch_T_cam_lhand = data_dict["T_cam_lhand"].to(device)
        batch_T_cam_rhand = data_dict["T_cam_rhand"].to(device)

        optimizer.zero_grad()

        y_hat = model(x)
        pos_loss, _ = position_loss(y, y_hat)
        rot_loss, _ = rotation_loss(y, y_hat)

        loss = pos_loss + rot_loss
        loss.backward()
        optimizer.step()

        # bookkeeping
        data = {'iteration':iteration, 'train_loss':loss.item(), 'pos': pos_loss.item(), 'rot': rot_loss.item()}
        loss_dict["train"]['iterations'].append(iteration)
        loss_dict["train"]['loss'].append(loss.item())
        loss_dict["train"]['pos_loss'].append(pos_loss.item())
        loss_dict["train"]['rot_loss'].append(rot_loss.item())
        if iteration % 10 == 0:
            pbar.set_postfix(data)

        # visualize the first in each batch
        if iteration % 10 == 0:
            worldcentric_scene = PlotlyScene(
                size=(600, 600), x_range=(-2, 2), y_range=(-2, 2), z_range=(-2, 2)
            )
            cameracentric_scene = PlotlyScene(
                size=(600, 600), x_range=(-2, 2), y_range=(-2, 2), z_range=(-2, 2)
            )
            batch_T_cam_lhandHat, batch_T_cam_rhandHat = convert_yhat_to_transforms(y_hat.detach().float())

            img = unnormalize_transform(x.detach())[0].permute((1, 2, 0)).cpu().numpy()
            T_world_cam = batch_T_world_cam[0]
            T_cam_lhand = batch_T_cam_lhand[0]
            T_cam_rhand = batch_T_cam_rhand[0]
            T_cam_lhandHat = batch_T_cam_lhandHat[0]
            T_cam_rhandHat = batch_T_cam_rhandHat[0]
            T_world_lhand = torch.matmul(T_world_cam, T_cam_lhand).cpu().numpy()
            T_world_rhand = torch.matmul(T_world_cam, T_cam_rhand).cpu().numpy()
            T_world_lhandHat = torch.matmul(T_world_cam, T_cam_lhandHat).cpu().numpy()
            T_world_rhandHat = torch.matmul(T_world_cam, T_cam_rhandHat).cpu().numpy()

            plot_transform(worldcentric_scene.figure, np.eye(4), label="world")
            plot_transform(worldcentric_scene.figure, T_world_lhand, label="lhand")
            plot_transform(worldcentric_scene.figure, T_world_rhand, label="rhand")
            plot_transform(worldcentric_scene.figure, T_world_lhandHat, label="lhandHat")
            plot_transform(worldcentric_scene.figure, T_world_rhandHat, label="rhandHat")
            visualizer.plot_scene(worldcentric_scene, window_name="world")

            plot_transform(cameracentric_scene.figure, np.eye(4), label="cam")
            plot_transform(cameracentric_scene.figure, T_cam_lhand.cpu().numpy(), label="lhand")
            plot_transform(cameracentric_scene.figure, T_cam_rhand.cpu().numpy(), label="rhand")
            plot_transform(cameracentric_scene.figure, T_cam_lhandHat.cpu().numpy(), label="lhandHat")
            plot_transform(cameracentric_scene.figure, T_cam_rhandHat.cpu().numpy(), label="rhandHat")
            visualizer.plot_scene(cameracentric_scene, window_name="cam")

            visualizer.plot_rgb(rgb_hwc=img, window_name="image")

            loss_curve = generate_plotly_loss_figure(loss_dict)
            visualizer.plot_plotlyfigure(loss_curve, window_name="losses")

        # periodically run on the val dataset
        if iteration % 10000 == 0:
            with torch.no_grad():
                model.eval()
                pos_losses = []
                rot_losses = []
                total_losses = []
                for data_dict in tqdm(val_dataloader):
                    x = data_dict["image"].to(device)
                    y = data_dict["flattened_T_cam_hands"].to(device)
                    y_hat = model(x)
                    pos_loss, _ = position_loss(y, y_hat)
                    rot_loss, _ = rotation_loss(y, y_hat)
                    loss = pos_loss + rot_loss
                    pos_losses.append(pos_loss.item())
                    rot_losses.append(rot_loss.item())
                    total_losses.append(loss.item())
                loss_dict["val"]['iterations'].append(iteration)
                loss_dict["val"]['loss'].append(np.mean(total_losses))
                loss_dict["val"]['pos_loss'].append(np.mean(pos_losses))
                loss_dict["val"]['rot_loss'].append(np.mean(rot_losses))

                model.train()


        iteration += 1

    if e % 2 == 0:
        e_str = str(e).zfill(3)
        model_save_path = f"{model_dir}/poseregressor_e{e_str}.pth"
        model.save_model(model_save_path)
        print(f"wrote epoch {e} model to: {model_save_path}")

        pickle.dump(loss_dict, open(f"{model_dir}/poseregressor_losses_dict.pkl", "wb"))
        print(f"wrote losses to: {model_dir}/poseregressor_losses_dict.pkl")

model_save_path = f"{model_dir}/poseregressor.pth"
model.save_model(model_save_path)
pickle.dump(loss_dict, open(f"{model_dir}/poseregressor_losses_dict.pkl", "wb"))
print(f"wrote losses to: {model_dir}/poseregressor_losses_dict.pkl")

res = input("training done! (enter/pdb)")
if res == 'pdb':
    import pdb; pdb.set_trace()
