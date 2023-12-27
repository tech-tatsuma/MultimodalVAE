import cv2
import os
import torch
import pickle
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageNet

class ImageDataset(Dataset):
    def __init__(self, root, split='train', cache="./", transform=None):
        """
        :param root: ImageNet データセットが格納されているディレクトリのパス
        :param split: データセットの分割 ('train' または 'val')
        :param cache: キャッシュディレクトリへのパス
        :param transform: データに適用する変換
        """
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]) if transform is None else transform
        self.dataset = ImageNet(root=root, split=split, transform=self.transform)
        self.cache_dir = cache

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        cache_path = os.path.join(self.cache_dir, f"{idx}.pkl")

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                image = pickle.load(f)
            return image

        image, _ = self.dataset[idx]

        with open(cache_path, 'wb') as f:
            pickle.dump(image, f)

        return image