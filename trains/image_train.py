import torch
import torchvision
import torchvision.transforms as transforms, Compose
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import random_split
from torch import nn

import argparse
import random
import numpy as np
from tqdm import tqdm
import sys
from matplotlib import pyplot as plt
from setproctitle import setproctitle
import os

from models.imagevae import VanillaVAE
from datasets.imagedataset import ImageDataset

# シードの設定
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(opt):
    # シードの設定
    seed_everything(opt.seed)

    # デバイスの設定
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ハイパーパラメータの設定
    epochs = opt.epochs
    patience = opt.patience
    learning_rate = opt.learning_rate
    batch_size = opt.batch_size
    cache = opt.cache

    train_dataset = ImageDataset(root='./data', split='train', cache=cache)
    val_dataset = ImageDataset(root='./data', split='val', cache=cache)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # モデルの定義
    model = VanillaVAE(in_channels=3, latent_dim=1024, hidden_dims=[32, 64, 128, 256, 512])

    # モデルの並列化
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)