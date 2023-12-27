import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class VanillaVAE(BaseVAE): # BaseVAEを継承したVanillaVAEクラスの定義


    def __init__(self,
                 in_channels: int, # 入力チャネル数
                 latent_dim: int, # 潜在空間の次元数
                 hidden_dims: List = None, # 隠れ層の次元数のリスト
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim # 潜在空間の次元数を設定

        modules = [] # エンコーダのモジュールを格納するリスト
        if hidden_dims is None: # 隠れ層の次元数が指定されていない場合のデフォルト値
            hidden_dims = [32, 64, 128, 256, 512]

        # エンコーダの構築
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim # 入力チャネルを更新

        self.encoder = nn.Sequential(*modules) # エンコーダネットワークを構築
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim) # 平均μを出力する全結合層
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim) # 分散σを出力する全結合層


        # デコーダの構築
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4) # デコーダの入力層

        hidden_dims.reverse() # 隠れ層の次元数を逆順に

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules) # デコーダネットワークを構築

        self.final_layer = nn.Sequential( # 最終層
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh()) # 出力を[-1, 1]の範囲に正規化

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        エンコーダネットワークを通じて入力をエンコードし、潜在コードを返す。
        :param input: (Tensor) エンコーダへの入力テンソル[B x C x H x W]
        :return: (Tensor) 潜在コードのリスト
        """

        result = self.encoder(input) # 入力をエンコーダに通す
        result = torch.flatten(result, start_dim=1) # 結果をフラット化

        # 潜在ガウス分布のmuとvar成分に結果を分割
        mu = self.fc_mu(result) # 平均μを計算
        log_var = self.fc_var(result) # 分散σを計算

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        N(0,1)からN(mu,var)をサンプリングするための再パラメータ化トリック。
        :param mu: (Tensor) 潜在ガウスの平均 [B x D]
        :param logvar: (Tensor) 潜在ガウスの標準偏差 [B x D]
        :return: (Tensor) [B x D]
        """
        result = self.decoder_input(z) # 標準偏差を計算
        result = result.view(-1, 512, 2, 2) # 入力をエンコード
        result = self.decoder(result) # 際パラメータ化
        result = self.final_layer(result) # デコードされた値、入力、mu、log_varを返す
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        N(0,1)からN(mu, var)をサンプリングするための再パラメータ化トリック。
        :param mu: (Tensor) 潜在ガウスの平均 [B x D]
        :param logvar: (Tensor) 潜在ガウスの標準偏差 [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar) # 標準偏差を計算
        eps = torch.randn_like(std) # 標準正規分布からサンプリング
        return eps * std + mu # サンプリングされた値を返す

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:

        mu, log_var = self.encode(input) # 入力をエンコード
        z = self.reparameterize(mu, log_var) # 再パラメータ化
        return  [self.decode(z), input, mu, log_var] # デコードされた値、入力、mu、log_varを返す

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

        kld_weight = kwargs['M_N'] # データセットからのミニバッチサンプルを考慮
        recons_loss =F.mse_loss(recons, input) # 再構築損失を計算


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0) # KLD損失を計算

        loss = recons_loss + kld_weight * kld_loss # 総損失を計算
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        潜在空間からサンプリングし、対応する画像空間マップを返す。
        :param num_samples: (int) サンプル数
        :param current_device: (int) モデルを実行するデバイス
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim) # 潜在空間からランダムにサンプリング

        z = z.to(current_device) # サンプリングされた値をデバイスに移動

        samples = self.decode(z) # デコードしてサンプルを得る
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        入力画像xが与えられた場合、再構築された画像を返します。
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0] # forwardメソッドを使って入力画像を再構築し、返す