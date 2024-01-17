import cv2
import os
import torch
import pickle
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import Kinetics400

class VideoDataset(Dataset):
    def __init__(self, directory, cache="./"):
        """
        :param directory: 動画データが格納されているディレクトリのパス
        :param frame_rate: 目的のフレームレート
        :param clip_length: クリップの長さ（秒）
        """
        self.directory = directory
        self.cache_dir = cache
        self.videos = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp4')]

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):

        # 動画データのパスを取得
        video_path = self.videos[idx]

        # 動画データを読み込み
        cap = cv2.VideoCapture(video_path)

        # キャッシュのパスを取得
        cache_path = os.path.join(self.cache_dir, f"{idx}.pkl")

        # キャッシュが存在すればそこから読み込み
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                frames = pickle.load(f)
            return frames

        frames = []

        for _ in range(8):  # 最初の8フレームのみ読み込む
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        # 画像をテンソルに変更
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        # 画像をリストに格納
        frames = [transform(frame) for frame in frames]
        
        # 画像のリストをテンソルに変換し、それを動画データとする
        frames = torch.stack(frames)

        # キャッシュに保存
        with open(cache_path, 'wb') as f:
            pickle.dump(frames, f)

        return frames