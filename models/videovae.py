import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple


class VideoVAE(BaseVAE): # BaseVAEを継承したVideoVAEクラスの定義

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List[int] = None, 
                 video_shape: Tuple[int, int, int, int] = (3, 16, 64, 64), **kwargs) -> None:
        super(VideoVAE, self).__init__()

        self.latent_dim = latent_dim # 潜在空間の次元数を設定
        self.in_channels, self.frames, self.height, self.width = video_shape

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # エンコーダの構築（Conv3dを使用）
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(self.in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU())
            )
            self.in_channels = h_dim

        self.encoder = nn.Sequential(*modules) # エンコーダネットワークを構築
        self._calculate_flatten_size() # エンコーダの出力サイズを計算
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim) # 平均μを出力する全結合層
        self.fc_var = nn.Linear(self.flatten_size, latent_dim) # 分散σを出力する全結合層


        # デコーダの構築
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.frames // (2 ** len(hidden_dims)) * self.height // (2 ** len(hidden_dims)) * self.width // (2 ** len(hidden_dims)))

        hidden_dims.reverse() # 隠れ層の次元数を逆順に

        # デコーダ層の構築
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm3d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules) # デコーダネットワークを構築

        # 最終層（動画フレームのサイズに復元するための層）
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose3d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
                            nn.BatchNorm3d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv3d(hidden_dims[-1], out_channels=self.in_channels, kernel_size=3, padding=1),
                            nn.Tanh())

    def _calculate_flatten_size(self):
        with torch.no_grad():
            sample_input = torch.zeros(1, self.in_channels, self.frames, self.height, self.width)
            output = self.encoder(sample_input)
            self.flatten_size = int(torch.prod(torch.tensor(output.shape[1:])))

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        エンコーダネットワークを通じて入力をエンコードし、潜在コードを返す。
        :param input: (Tensor) エンコーダへの入力テンソル[B x C x H x W]
        :return: (Tensor) 潜在コードのリスト
        """

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        N(0,1)からN(mu,var)をサンプリングするための再パラメータ化トリック。
        :param mu: (Tensor) 潜在ガウスの平均 [B x D]
        :param logvar: (Tensor) 潜在ガウスの標準偏差 [B x D]
        :return: (Tensor) [B x D]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        N(0,1)からN(mu, var)をサンプリングするための再パラメータ化トリック。
        :param mu: (Tensor) 潜在ガウスの平均 [B x D]
        :param logvar: (Tensor) 潜在ガウスの標準偏差 [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:

        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        VAEの損失関数を計算。
        KL(N(μ, σ), N(0, 1)) = log(1/σ) + (σ^2 + μ^2)/2 - 1/2
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs.get('M_N', 1) # デフォルトのKLD重み
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> torch.Tensor:
        """
        潜在空間からサンプリングし、対応するビデオデータを返す。
        :param num_samples: (int) サンプル数
        :param current_device: (int) モデルを実行するデバイス
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim) # 潜在空間からランダムにサンプリング

        z = z.to(current_device) # サンプリングされた値をデバイスに移動

        samples = self.decode(z) # デコードしてサンプルを得る
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    入力画像xが与えられた場合、再構築された画像を返します。
    :param x: (Tensor) 入力テンソル [B x C x F x H x W]
    :return: (Tensor) 再構築されたテンソル [B x C x F x H x W]
    """
    # 入力データの形状の確認
    if len(x.shape) != 5:
        raise ValueError("input data shape error: [B x C x F x H x W]")

    # エンコーダとデコーダを通して再構築
    recons = self.forward(x)[0]

    return recons