```PYTHON
```!pip install git+https://github.com/OutofAi/tigersound.git
```

```PYTHON
import tigersound.look2hear.models
import torchaudio
import torch
import uuid
import os


device = "cuda"

dnr_model = tigersound.look2hear.models.TIGERDNR.from_pretrained("JusperLee/TIGER-DnR", cache_dir="cache").to("cuda").eval()

audio_file = "sample.mp3"
audio, sr = torchaudio.load(audio_file)
audio = audio.to(device)

with torch.no_grad():
    dialog, effect, music = dnr_model(audio[None])

    
session_id = uuid.uuid4().hex[:8]
output_dir = os.path.join("output_dnr", session_id)
os.makedirs(output_dir, exist_ok=True)

paths = {
    "dialog": os.path.join(output_dir, "dialog.wav"),
    "effect": os.path.join(output_dir, "effect.wav"),
    "music": os.path.join(output_dir, "music.wav"),
}

torchaudio.save(paths["dialog"], dialog.cpu(), sr)
torchaudio.save(paths["effect"], effect.cpu(), sr)
torchaudio.save(paths["music"], music.cpu(), sr)
```
