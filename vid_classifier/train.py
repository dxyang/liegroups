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

from vid_classifier.data import PointMassFrameDataset, get_unnormalize_transform
from vid_classifier.models import FrameClassifier
from viz import VisdomVisualizer, PlotlyScene, plot_transform, generate_plotly_loss_figure

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# bookkeeping things
experiment_name = "07302022_pointmass"
output_directory = os.path.expanduser("~/localdata/vid_classifier")
exp_folder = f"{output_directory}/{experiment_name}"
model_dir = f"{exp_folder}/models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# train
epochs = 200
batch_size = 128

# dataset
train_dset = PointMassFrameDataset(data_split="train")
val_dset = PointMassFrameDataset(data_split="val")
train_dataloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=16)
val_dataloader = DataLoader(val_dset, batch_size=batch_size, shuffle=True, num_workers=16)

# model
model_output_size = 1
model = FrameClassifier(output_size=model_output_size)
model.to(device)

optimizer = optim.Adam(model.parameters())
loss_f = torch.nn.BCEWithLogitsLoss()

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
        "same_ep_gt_loss": [],
        "same_ep_lt_loss": [],
        "diff_ep_loss": [],
    },
    "val": {
        "iterations": [],
        "loss": [],
        "same_ep_gt_loss": [],
        "same_ep_lt_loss": [],
        "diff_ep_loss": [],
    },
}
for e in pbar:
    for data_dict in tqdm(train_dataloader):
        xs = data_dict["image"].to(device)
        gt_xs = data_dict["gt_image"].to(device)
        lt_xs = data_dict["lt_image"].to(device)
        other_xs = data_dict["other_image"].to(device)
        goals = data_dict["goal"].to(device)

        optimizer.zero_grad()

        logits = model(xs, goals)
        gt_logits = model(gt_xs, goals)
        lt_logits = model(lt_xs, goals)
        other_logits = model(other_xs, goals)

        # figure out what we're comparing
        bs = xs.size()[0]
        gt_pair_logits = torch.cat([logits, gt_logits], dim=1) # BS x 2
        lt_pair_logits = torch.cat([logits, lt_logits], dim=1) # BS x 2
        other_pair_logits = torch.cat([logits, other_logits], dim=1) # BS x 2
        gt_pair_labels = torch.cat([torch.tensor([1, 0]).unsqueeze(0) for _ in range(bs)]).float().to(device)
        lt_pair_labels = torch.cat([torch.tensor([0, 1]).unsqueeze(0) for _ in range(bs)]).float().to(device)
        other_pair_labels = torch.cat([torch.tensor([1, 0]).unsqueeze(0) for _ in range(bs)]).float().to(device)

        # calculate the loss
        same_ep_gt_loss = loss_f(gt_pair_logits, gt_pair_labels)
        same_ep_lt_loss = loss_f(lt_pair_logits, lt_pair_labels)
        diff_ep_loss = loss_f(other_pair_logits, other_pair_labels)
        loss = same_ep_gt_loss + same_ep_lt_loss + diff_ep_loss

        loss.backward()
        optimizer.step()

        # bookkeeping
        data = {
            'iteration':iteration,
            'train_loss':loss.item(),
            'train_same_ep_gt_loss': same_ep_gt_loss.item(),
            'train_same_ep_lt_loss': same_ep_lt_loss.item(),
            'train_diff_ep_loss': diff_ep_loss.item(),
        }

        loss_dict["train"]['iterations'].append(iteration)
        loss_dict["train"]['loss'].append(loss.item())
        loss_dict["train"]['same_ep_gt_loss'].append(same_ep_gt_loss.item())
        loss_dict["train"]['same_ep_lt_loss'].append(same_ep_lt_loss.item())
        loss_dict["train"]['diff_ep_loss'].append(diff_ep_loss.item())
        if iteration % 10 == 0:
            pbar.set_postfix(data)

        if iteration % 10 == 0:
            loss_curve = generate_plotly_loss_figure(loss_dict)
            visualizer.plot_plotlyfigure(loss_curve, window_name="losses")

        # periodically run on the val dataset
        if e % 2 == 0:
            with torch.no_grad():
                model.eval()

                for data_dict in tqdm(val_dataloader):
                    xs = data_dict["image"].to(device)
                    gt_xs = data_dict["gt_image"].to(device)
                    lt_xs = data_dict["lt_image"].to(device)
                    other_xs = data_dict["other_image"].to(device)
                    goals = data_dict["goal"].to(device)

                    logits = model(xs, goals)
                    gt_logits = model(gt_xs, goals)
                    lt_logits = model(lt_xs, goals)
                    other_logits = model(other_xs, goals)

                    # figure out what we're comparing
                    bs = xs.size()[0]
                    gt_pair_logits = torch.cat([logits, gt_logits], dim=1) # BS x 2
                    lt_pair_logits = torch.cat([logits, lt_logits], dim=1) # BS x 2
                    other_pair_logits = torch.cat([logits, other_logits], dim=1) # BS x 2
                    gt_pair_labels = torch.cat([torch.tensor([1, 0]).unsqueeze(0) for _ in range(bs)]).float().to(device)
                    lt_pair_labels = torch.cat([torch.tensor([0, 1]).unsqueeze(0) for _ in range(bs)]).float().to(device)
                    other_pair_labels = torch.cat([torch.tensor([1, 0]).unsqueeze(0) for _ in range(bs)]).float().to(device)

                    # calculate the loss
                    same_ep_gt_loss = loss_f(gt_pair_logits, gt_pair_labels)
                    same_ep_lt_loss = loss_f(lt_pair_logits, lt_pair_labels)
                    diff_ep_loss = loss_f(other_pair_logits, other_pair_labels)
                    loss = same_ep_gt_loss + same_ep_lt_loss + diff_ep_loss

                    loss_dict["val"]['iterations'].append(iteration)
                    loss_dict["val"]['loss'].append(loss.item())
                    loss_dict["val"]['same_ep_gt_loss'].append(same_ep_gt_loss.item())
                    loss_dict["val"]['same_ep_lt_loss'].append(same_ep_lt_loss.item())
                    loss_dict["val"]['diff_ep_loss'].append(diff_ep_loss.item())

                model.train()


        iteration += 1

    if e % 2 == 0:
        e_str = str(e).zfill(3)
        model_save_path = f"{model_dir}/frameclassifier_e{e_str}.pth"
        model.save_model(model_save_path)
        print(f"wrote epoch {e} model to: {model_save_path}")

        pickle.dump(loss_dict, open(f"{model_dir}/frameclassifier_losses_dict.pkl", "wb"))
        print(f"wrote losses to: {model_dir}/frameclassifier_losses_dict.pkl")

model_save_path = f"{model_dir}/frameclassifier.pth"
model.save_model(model_save_path)
pickle.dump(loss_dict, open(f"{model_dir}/frameclassifier_losses_dict.pkl", "wb"))
print(f"wrote losses to: {model_dir}/frameclassifier_losses_dict.pkl")

res = input("training done! (enter/pdb)")
if res == 'pdb':
    import pdb; pdb.set_trace()
