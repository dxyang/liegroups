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
import torchvision

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FrameClassifier(nn.Module):
    def __init__(self, output_size: int, pretrained: bool = True):
        super(FrameClassifier, self).__init__()

        # get the backbone
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        fc_in_size = self.resnet.fc.in_features # 512
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add our head to it
        self.output_size = output_size
        self.fc_head = nn.Sequential(
            nn.Linear(fc_in_size + 2, 256), # +2 because of the goal!
            nn.PReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x: torch.Tensor, goal: torch.Tensor):
        x = self.resnet(x).squeeze(dim=2).squeeze(dim=2)
        x = torch.cat([x, goal], dim=1)
        x = self.fc_head(x)
        return x

    def save_model(self, model_path: str):
        print(f"saved frame classification model to {model_path}")
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path: str):
        print(f"loaded frame classification model from {model_path}")
        self.load_state_dict(torch.load(model_path))