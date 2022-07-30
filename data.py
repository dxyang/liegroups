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

def combine_subject_hdf5_data(dset_path: str = None):
    '''
    one off helper function to condense all the hdf5 files because running
    into system limits of having too many open at once
    '''
    all_traj_fps = glob.glob(os.path.expanduser(f"{dset_path}/*/*/*.hdf5"))
    subject_fps = {}
    for num in range(1, 11):
        subject_fps[f"s{num}"] = []

    # separate out the hdf5 files by subject
    for fp in all_traj_fps:
        subj_str = fp.split('/')[-3]
        assert subj_str in subject_fps
        subject_fps[subj_str].append(fp)

    # create a new hdf5 file for each subject
    for subj_str, hdf5_fp_list in subject_fps.items():
        new_hdf_fp = f"{dset_path}/{subj_str}/trajectory_data.hdf5"
        print(new_hdf_fp)
        new_f = h5py.File(new_hdf_fp, 'w')
        for action_fp in hdf5_fp_list:
            action_str = action_fp.split('/')[-2]

            traj_group = new_f.create_group(action_str)

            old_traj_f = h5py.File(action_fp, 'r')
            for k in old_traj_f.keys():
                traj_group[k] = old_traj_f[k][:]
            old_traj_f.close()

        new_f.close()


class Cam2HandDset(Dataset):
    def __init__(self, dset_path: str = None, subject_num: int = None, data_split: str = "train", hack_size: int = None):
        self.dset_path = dset_path
        if subject_num is not None:
            assert subject_num < 11 and subject_num > 0
            self.subject_nums = [subject_num]
        else:
            self.subject_nums = [i for i in range(1, 11)]
        self.all_traj_fps = [f"{dset_path}/s{i}/trajectory_data.hdf5" for i in self.subject_nums]


        assert data_split in ["train", "test", "val", "all"]
        self.split = data_split
        self.split_dict = pickle.load(open(TRAIN_TEST_VAL_SPLIT_PATH, "rb"))

        # a lot of images don't have any hands in them. determine which ones we'll use for training data
        self.NUM_PTS_IN_HAND = 778
        self.VISIBLE_THRESHOLD = 0.25

        self.subj_to_f = {}
        self.subj_action_to_valid_idxs = {}

        print(f"preprocessing dataset to determine number of valid frames")
        for fp in tqdm(self.all_traj_fps):
            subj_f = h5py.File(fp, "r")
            subject = fp.split('/')[-2]
            self.subj_to_f[subject] = subj_f
            actions_done_by_subject = subj_f.keys()

            action_dict = {}
            for action in actions_done_by_subject:
                if self.split != "all":
                    if (subject, action) not in self.split_dict[self.split]:
                        continue

                action_traj = subj_f[action]

                is_lh_visible = action_traj['num_pts_visible_lhand'][:] > self.VISIBLE_THRESHOLD * self.NUM_PTS_IN_HAND
                is_rh_visible = action_traj['num_pts_visible_rhand'][:] > self.VISIBLE_THRESHOLD * self.NUM_PTS_IN_HAND

                is_lh_or_rh_visible = np.logical_or(is_lh_visible, is_rh_visible)

                left_valid_idx = np.where(is_lh_visible)[0]
                right_valid_idx = np.where(is_rh_visible)[0]
                any_valid_idx = np.where(is_lh_or_rh_visible)[0]

                action_dict[action] = {'left': left_valid_idx, 'right': right_valid_idx, 'any': any_valid_idx}

            self.subj_action_to_valid_idxs[subject] = action_dict


        # this is useful for things that consume this dataset to organize outputs
        self.subj_action_dict = {}

        # dataset length
        self.length = 0
        for subj, action_trajs in self.subj_action_to_valid_idxs.items():
            if subj not in self.subj_action_dict:
                self.subj_action_dict[subj] = {}

            for action, valid_dict in action_trajs.items():
                self.length += len(valid_dict['any'])

                if action not in self.subj_action_dict[subj]:
                    self.subj_action_dict[subj][action] = {}


        # other
        self.img_preprocess = get_resnet_preprocess_transforms()
        self.hack_size = hack_size

        print(f"dataset has {self.length} valid image frames!")
        if self.hack_size is not None:
            print(f"but just using {self.hack_size} for debug")
            self.length = self.hack_size

    def __len__(self):
        return self.length

    def get_subj_action_dict(self):
        return copy.copy(self.subj_action_dict)

    def idx_to_subj_action_subidx(self, idx):
        curr = idx
        for subj, actions_to_valid_idxs in self.subj_action_to_valid_idxs.items():
            for action, valid_idxs in actions_to_valid_idxs.items():
                num_valid_idxs = len(valid_idxs["any"])
                if curr < num_valid_idxs:
                    return subj, action, valid_idxs["any"][curr]
                else:
                    curr -= num_valid_idxs
                    if curr < 0:
                        assert False # this shouldn't happen

    def subj_action_subidx_to_idx(self, subj, action, subidx):
        global_idx = 0
        for _subj, _actions_to_valid_idxs in self.subj_action_to_valid_idxs.items():
            for _action, valid_idxs in _actions_to_valid_idxs.items():
                num_valid_idxs = len(valid_idxs["any"])
                if _subj != subj:
                    global_idx += num_valid_idxs
                elif _action != action:
                    global_idx += num_valid_idxs
                else:
                    return global_idx + subidx
        assert False # this shouldn't happen

    def __getitem__(self, idx):
        return_dict = {}

        if idx < 0 or idx >= self.length:
            raise IndexError

        subj_str, action_str, subidx = self.idx_to_subj_action_subidx(idx)

        # get transform info
        f = self.subj_to_f[subj_str]
        traj = f[action_str]

        T_world_cam = traj["T_world_camera"][subidx]

        if traj["num_pts_visible_lhand"][subidx] > self.VISIBLE_THRESHOLD * self.NUM_PTS_IN_HAND:
            T_cam_lhand = traj["T_camera_lhand"][subidx]
        else:
            T_cam_lhand = np.eye(4)

        if traj["num_pts_visible_rhand"][subidx] > self.VISIBLE_THRESHOLD * self.NUM_PTS_IN_HAND:
            T_cam_rhand = traj["T_camera_rhand"][subidx]
        else:
            T_cam_rhand = np.eye(4)

        # flatten transforms for neural nets
        T_cam_lhand_tensor = torch.from_numpy(T_cam_lhand[:3, :3])
        T_cam_rhand_tensor = torch.from_numpy(T_cam_rhand[:3, :3])
        q_lhand = rotation_transforms.matrix_to_quaternion(T_cam_lhand_tensor)
        q_rhand = rotation_transforms.matrix_to_quaternion(T_cam_rhand_tensor)
        pos_lhand = torch.from_numpy(T_cam_lhand[:3, 3])
        pos_rhand = torch.from_numpy(T_cam_rhand[:3, 3])
        nn_output = torch.cat([pos_lhand, q_lhand, pos_rhand, q_rhand])

        # load the image
        image_num_str = str(subidx).zfill(4)
        image_path = f"{self.dset_path}/{subj_str}/{action_str}/{image_num_str}.png"
        img = cv2.imread(image_path)
        resized_normed_img = self.img_preprocess(img)
        # print(f"{traj['num_pts_visible_lhand'][subidx]}, {traj['num_pts_visible_rhand'][subidx]}")
        assert (traj["num_pts_visible_lhand"][subidx] > self.VISIBLE_THRESHOLD * self.NUM_PTS_IN_HAND) or \
               (traj["num_pts_visible_rhand"][subidx] > self.VISIBLE_THRESHOLD * self.NUM_PTS_IN_HAND)

        return {
            "image": resized_normed_img,
            "T_world_cam": T_world_cam.astype(np.float32),
            "T_cam_lhand": T_cam_lhand.astype(np.float32),
            "T_cam_rhand": T_cam_rhand.astype(np.float32),
            "flattened_T_cam_hands": nn_output.float(), # size 14 (two 7 things)
            "action": action_str,
            "subject": subj_str,
            "seq_idx": subidx,
            "lhand_pts_visible": traj["num_pts_visible_lhand"][subidx],
            "rhand_pts_visible": traj["num_pts_visible_rhand"][subidx],
        }

def generate_train_test_val_splits():
    cam2hand_dset = Cam2HandDset(os.path.expanduser("~/localdata/cam2hand_dset"))
    subj_action_dict = cam2hand_dset.get_subj_action_dict()

    subj_action_tuples = []

    for subj, v in subj_action_dict.items():
        for action, empty in v.items():
                subj_action_tuples.append((subj, action))

    train_end = int(0.7 * len(subj_action_tuples))
    val_end = int(0.1 * len(subj_action_tuples))
    random.shuffle(subj_action_tuples)

    train_set = subj_action_tuples[:train_end]
    val_set = subj_action_tuples[train_end:train_end+val_end]
    test_set = subj_action_tuples[train_end+val_end:]

    splits_dict = {
        "train": train_set,
        "val": val_set,
        "test": test_set
    }

    pickle.dump(splits_dict, open(TRAIN_TEST_VAL_SPLIT_PATH, "wb"))
    print(f"dataset splits saved to {TRAIN_TEST_VAL_SPLIT_PATH}")


def sanity_check_split_dict():
    split_dict = pickle.load(open(TRAIN_TEST_VAL_SPLIT_PATH, "rb"))
    for split_str, subj_action_tuple_list in split_dict.items():
        subj_dict = {}
        for subj, action in subj_action_tuple_list:
            if subj in subj_dict:
                subj_dict[subj] += 1
            else:
                subj_dict[subj] = 1

        for i in range(1, 11):
            assert subj_dict[f"s{i}"] > 0

if __name__ == "__main__":
    # sanity_check_split_dict()
    # exit()

    # generate_train_test_val_splits()
    # exit()

    # combine_subject_hdf5_data(os.path.expanduser("~/localdata/cam2hand_dset"))
    # exit()

    all_cam2hand_dset = Cam2HandDset(os.path.expanduser("~/localdata/cam2hand_dset"), data_split="all")
    train_cam2hand_dset = Cam2HandDset(os.path.expanduser("~/localdata/cam2hand_dset"), data_split="train")
    test_cam2hand_dset = Cam2HandDset(os.path.expanduser("~/localdata/cam2hand_dset"), data_split="test")
    val_cam2hand_dset = Cam2HandDset(os.path.expanduser("~/localdata/cam2hand_dset"), data_split="val")

    for val in tqdm(all_cam2hand_dset):
        pass

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
