from typing import List

import copy
import glob
import os
import pickle
import random

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import transforms
from tqdm import tqdm
from pytorch3d import transforms as rotation_transforms

TRAIN_TEST_VAL_SPLIT_PATH = os.path.expanduser("~/localdata/cam2hand_dset/splits.pkl")

def get_resnet_preprocess_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
    ])

def get_unnormalize_transform():
    return transforms.Compose([
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    ])

POINTMASS_DSET_PATH = os.path.expanduser("~/localdata/vid_classifier/pointmass.hdf5")


class PointMassFrameDataset(Dataset):
    def __init__(self, dset_path: str = POINTMASS_DSET_PATH, data_split: str = "train", hack_size: int = None):
        self.dset_path = dset_path

        assert data_split in [
            "train", # first 70% of episodes
            "test",  # next 20% of episodes
            "val",   # next 10% of episodes
            "all"
        ]
        self.split = data_split

        self.f = h5py.File(POINTMASS_DSET_PATH, "r")

        self.ep_keys = [int(k) for k in self.f.keys()]
        self.ep_keys.sort()
        if self.split == "train":
            cutoff = int(0.7 * len(self.ep_keys))
            self.ep_keys = self.ep_keys[:cutoff]
        elif self.split == "test":
            cutoffa = int(0.7 * len(self.ep_keys))
            cutoffb = int(0.9 * len(self.ep_keys))
            self.ep_keys = self.ep_keys[cutoffa:cutoffb]
        elif self.split == "val":
            cutoff = int(0.9 * len(self.ep_keys))
            self.ep_keys = self.ep_keys[cutoff:]

        self.num_episodes = len(self.ep_keys)
        self.ep_length = self.f[str(self.ep_keys[0])]["imgs"][:].shape[0]

        self.length = self.num_episodes * self.ep_length

        self.img_transforms = get_resnet_preprocess_transforms()

    def __len__(self):
        return self.length

    def idx_to_ep_subidx(self, idx):
        curr = idx
        for ep_num in self.ep_keys:
            assert curr >= 0
            if curr < self.ep_length:
                return ep_num, curr
            else:
                curr -= self.ep_length


    def ep_subidx_to_idx(self, ep_num: int, subidx: int):
        idx = 0
        for _ep_num in self.ep_keys:
            if ep_num != _ep_num:
                idx += self.ep_length
            else:
                return idx + subidx


    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError

        ep_num, subidx = self.idx_to_ep_subidx(idx)
        # print(f"[{self.split}] {ep_num} {subidx}")

        traj_dict = self.f[str(ep_num)]

        img = traj_dict["imgs"][subidx, :]
        pp_img = self.img_transforms(img)

        goal = traj_dict["goals"][:]

        return {
            "image": pp_img,
            "traj_num": ep_num,
            "step": subidx,
            "goal": goal,
        }


if __name__ == "__main__":
    # sanity_check_split_dict()
    # exit()

    # generate_train_test_val_splits()
    # exit()

    # combine_subject_hdf5_data(os.path.expanduser("~/localdata/cam2hand_dset"))
    # exit()

    all_cam2hand_dset = PointMassFrameDataset(data_split="all")
    train_cam2hand_dset = PointMassFrameDataset(data_split="train")
    test_cam2hand_dset = PointMassFrameDataset(data_split="test")
    val_cam2hand_dset = PointMassFrameDataset(data_split="val")

    for val in tqdm(all_cam2hand_dset):
        pass
    input()
    for val in tqdm(train_cam2hand_dset):
        pass
    input()
    for val in tqdm(test_cam2hand_dset):
        pass
    input()
    for val in tqdm(val_cam2hand_dset):
        pass
    input()

    # dataloader = DataLoader(cam2hand_dset, batch_size=64, num_workers=12, shuffle=False)
    # for data_dict in tqdm(dataloader):
    #     pass

    exit()
    unnormalize = get_unnormalize_transform()
    img_save_folder = os.path.expanduser("~/localdata/dset_bridge/models/TEST")
    if not os.path.isdir(f"{img_save_folder}/s1"):
        os.makedirs(f"{img_save_folder}/s1")
        os.makedirs(f"{img_save_folder}/s2")
    for idx, (s1, s2, rand_vec) in enumerate(seq2seq_singleenv_dset):
        plot_idx_str = str(idx).zfill(len(str(len(seq2seq_singleenv_dset))))
        for t_idx in range(200):
            t_idx_str = str(t_idx).zfill(len(str(s1.shape[0])))
            img1 = s1[t_idx]
            img2 = s2[0][t_idx]
            torchvision.utils.save_image(unnormalize(img1), f"{img_save_folder}/s1/{plot_idx_str}_{t_idx_str}.png")
            torchvision.utils.save_image(unnormalize(img2), f"{img_save_folder}/s2/{plot_idx_str}_{t_idx_str}.png")
        if idx == 1:
            break

    import pdb; pdb.set_trace()
    s1, s2, rand_vec = seq2seq_singleenv_dset[0]

    data_file_list = [
        "~/localdata/dset_bridge/reach-v2-goal-observable_data.hdf5",
        "~/localdata/dset_bridge/push-v2-goal-observable_data.hdf5",
        "~/localdata/dset_bridge/pick-place-v2-goal-observable_data.hdf5",
    ]
    seq2seq_multienv_dset = RobotTrajMultiEnvDataset(data_file_list, use_imgs=True, overfit=True)
    count = 0

    for s1, _, _ in seq2seq_multienv_dset:
        count += 1
        if count % 100 == 0:
            print(count)
