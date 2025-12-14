from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import torch
import torch.nn as nn
from torchaudio import transforms, datasets
import io
import soundfile as sf


class CheckUrban(nn.Module):
  def __init__(self):
    super().__init__()
    self.first = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((8, 8))

    )
    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

  # def forward(self, x):
  #   x = x.unsqueeze(1)
  #   x = self.first(x)
  #   x = self.second(x)
  #   return x

  def forward(self, x):
      # если x имеет 4 измерения (batch, channel, height, width) — не трогай
      if x.dim() == 3:
          x = x.unsqueeze(1)
      elif x.dim() == 5:
          x = x.squeeze(2)

      x = self.first(x)
      x = self.second(x)
      return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.MelSpectrogram(
    sample_rate=22050,
    n_mels=64
)

max_len = 500
genres = torch.load('name_urban.pth')
index_to_labels = {ind: lab for ind, lab in enumerate(genres)}
model = CheckUrban()
model.load_state_dict(torch.load('model_urban.pth', map_location=device))
model.to(device)
model.eval()



def change_audio(waveform, sample_rate):
    if sample_rate != 22050:
        new_sr = transforms.Resample(orig_freq=sample_rate, new_freq=22050)
        waveform = new_sr(torch.tensor(waveform))

    spec = transform(waveform).squeeze(0)

    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]

    if spec.shape[1] < max_len:
        pad_amount = max_len - spec.shape[1]
        spec = torch.nn.functional.pad(spec, (0, pad_amount))

    return spec

audio_app = FastAPI()


@audio_app.post('/predict/')
async def check_image(file: UploadFile = File(...)):
    try:
        data = await file.read()

        if not data:
            raise HTTPException(status_code =400, detail='Файл кошулган жок')


        # wf, sr = sf.read(io.BytesIO(data), dtype='float32')
        # wf = torch.tensor(wf).T

        wf, sr = sf.read(io.BytesIO(data), dtype='float32')

        # Если стерео → усредняем каналы (делаем моно)
        if wf.ndim > 1:
            wf = wf.mean(axis=1)

        # Преобразуем в тензор [1, N]
        wf = torch.tensor(wf).unsqueeze(0)


        spec = change_audio(wf, sr).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(spec)
            pred = torch.argmax(y_pred, dim=1).item()


        return {
            "Индекс": pred,
            "Жанр": index_to_labels[pred]
        }



    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





































if __name__ == "__main__":
    uvicorn.run(audio_app, host='127.0.0.1', port=8000)