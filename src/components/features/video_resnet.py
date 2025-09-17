import torch, numpy as np, cv2
from torchvision import models, transforms

class VideoResNet18Utt:
    def __init__(self, device:str="cpu"):
        self.device = torch.device(device)
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = torch.nn.Sequential(*list(m.children())[:-1]).to(self.device).eval()
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def _sample_frames(self, video_path, target_fps=2, max_frames=32):
        cap = cv2.VideoCapture(video_path)
        frames = []
        if not cap.isOpened(): return frames
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(int(src_fps // target_fps), 1)
        i = 0
        while True:
            ret = cap.read()[0]
            if not ret: break
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if pos % step == 0:
                ret, frame = cap.read()
            else:
                continue
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if len(frames) >= max_frames: break
        cap.release()
        return frames

    def __call__(self, video_path:str) -> np.ndarray:
        frames = self._sample_frames(video_path)
        if not frames:
            return np.zeros(512, dtype=np.float32)
        batch = torch.stack([self.tf(f) for f in frames]).to(self.device)  # [N,3,224,224]
        with torch.no_grad():
            feats = self.backbone(batch).squeeze(-1).squeeze(-1)  # [N,512]
        return feats.mean(dim=0).cpu().numpy().astype(np.float32)  # UTTERANCE pooled
