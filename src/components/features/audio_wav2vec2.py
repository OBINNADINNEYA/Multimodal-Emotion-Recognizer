import torch, numpy as np, librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

class AudioWav2Vec2Utt:
    def __init__(self, device:str="cpu", model_name:str="facebook/wav2vec2-base"):
        self.device = torch.device(device)
        self.proc = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device).eval()

    def __call__(self, wav_path:str) -> np.ndarray:
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        inputs = self.proc(y, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        with torch.no_grad():
            hs = self.model(inputs).last_hidden_state[0].cpu().numpy()  # [T, C]
        return hs.mean(axis=0).astype(np.float32)  # UTTERANCE pooled
