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

from models.videovae import VideoVAE
from datasets.videodataset import VideoDataset, KineticsDatasets

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
    iskinetics = opt.kinetics

    # デーセットの作成
    if iskinetics=='true':
        dataset = KineticsDatasets(root='./data')
    else:
        dataset = VideoDataset(directory='./data', frame_rate=25, clip_length=1, cache=cache)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # DataLoaderの作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # モデルの定義
    model = VideoVAE(in_channels=3, latent_dim=1024, hidden_dims=[32, 64, 128, 256, 512], video_shape=(3,25,64,64))

    # モデルの並列化
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # モデルの転送
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 早期終了のためのパラメータの初期化
    val_loss_min = None
    val_loss_min_epoch = 0

    # 学習曲線のための配列の初期化
    train_losses = []
    val_losses = []

    # モデルの訓練
    for epoch in tqdm(range(epochs)):

        # 各種パラメータの初期化
        train_loss = 0.0
        val_loss = 0.0

        # モデルをtrainモードに設定
        model.train()

        # trainデータのロード
        for data in train_loader:
            # データをデバイスに移動
            inputs = data.to(device)

            # 勾配の初期化
            optimizer.zero_grad()

            # forwardパス
            results = model(inputs)
            recons = results[0]
            mu = results[2]
            log_var = results[3]

            # 損失の計算
            recons_loss = F.mse_loss(recons, inputs)
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recons_loss + kld_loss

            # backwardパス
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # モデルを評価モードの設定
        model.eval()
        
        # 検証データでの評価
        with torch.no_grad():
            for data in val_loader:
                
                # データをGPUに転送
                inputs = data.to(device)

                results = model(inputs)

                recons = results[0]
                mu = results[2]
                log_var = results[3]

                # 損失の計算
                recons_loss = F.mse_loss(recons, inputs)
                kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recons_loss + kld_loss
                
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}')
        sys.stdout.flush()

        # メモリーを最適化する
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # バリデーションロスが下がった時は結果を保存する
        if val_loss_min is None or val_loss < val_loss_min:
            model_save_directory = './latestresult'
            model_save_name = f'./latestresult/lr{learning_rate}_ep{epochs}_pa{patience}intweak.pt'
            if not os.path.exists(model_save_directory):
                os.makedirs(model_save_directory)
            torch.save(model.state_dict(), model_save_name)
            val_loss_min = val_loss
            val_loss_min_epoch = epoch
            
        # もしバリデーションロスが一定期間下がらなかったらその時点で学習を終わらせる
        elif (epoch - val_loss_min_epoch) >= patience:
            print('Early stopping due to validation loss not improving for {} epochs'.format(patience))
            break

    # テストの実行
    test_loss = []
    with torch.no_grad():
        for data in test_loader:
            inputs = data.to(device)
            
            results = model(inputs)

            recons = results[0]
            mu = results[2]
            log_var = results[3]

            # 損失の計算
            recons_loss = F.mse_loss(recons, inputs)
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recons_loss + kld_loss
        

    # テストの結果を表示
    mean_test = sum(test_loss) / len(test_loss)
    print(f'Test Loss: {mean_test:.4f}')

    # 学習プロセスをグラフ化し、保存する
    plt.figure(figsize=(15, 5))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.title("Training and Validation Loss")
    graph_save_directory = './latestresult'
    graph_save_name = f'{graph_save_directory}/lr{learning_rate}_ep{epochs}_pa{patience}intweak.png'

    if not os.path.exists(graph_save_directory):
        os.makedirs(graph_save_directory)

    plt.savefig(graph_save_name)


    return train_loss, val_loss_min

if __name__ == '__main__':
    # プロセス名の設定
    setproctitle("VideoVAE")

    # パーサーの設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int, required=True, help='epochs')
    parser.add_argument('--learning_rate',type=float, default=0.001, help='learning rate')
    parser.add_argument('--patience', type=int, default=5, help='patience')
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generators')
    parser.add_argument('--batch',type=int, default=20, help='batch size')
    parser.add_argument('--cache', type=str, default='./', help='cache directory path')
    parser.add_argument('--kinetics', type=str, default='true')

    # オプションを標準出力する
    print(opt)
    print('-----biginning training-----')
    train_loss, val_loss = train(opt)
    print('final train loss: ',train_loss)
    print('final validation loss: ', val_loss)
    print('-----completing training-----')