import cv2
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from models import VideoVAE

# 学習済みモデルのロード
model = VideoVAE(in_channels=3, latent_dim=64)  # パラメータは適宜調整
model.load_state_dict(torch.load('path_to_saved_model.pth'))  # 保存されたモデルのパス
model.eval()

# 動画の読み込みと前処理
cap = cv2.VideoCapture('path_to_input_video.mp4')  # 入力動画のパス
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)
cap.release()

# 前処理
transform = Compose([
    Resize((64, 64)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
frames = [transform(frame) for frame in frames]
frames_tensor = torch.stack(frames).unsqueeze(0)  # バッチ次元を追加

# 再構成動画の生成
with torch.no_grad():
    reconstructed_frames, _, _, _ = model(frames_tensor)
reconstructed_frames = reconstructed_frames.squeeze(0)  # バッチ次元を削除

# 再構成動画の保存または表示
# 以下は再構成されたフレームを動画として保存する例
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (64, 64))
for frame in reconstructed_frames:
    frame = frame.permute(1, 2, 0).numpy()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write((frame * 255).astype('uint8'))
out.release()
