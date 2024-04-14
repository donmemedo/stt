import io

import librosa
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def get_model():
    checkpoint_path = "ser_model.pt"  # 'densenet121_model.pt'
    model = models.densenet121(pretrained=True)
    torch.save(model.state_dict(), "ser_model.pt")
    model.classifier = nn.Sequential(
        nn.Linear(1024, 512), nn.LeakyReLU(), nn.Linear(512, 8)
    )
    try:
        model.load_state_dict(
            torch.load(checkpoint_path, map_location="cuda"), strict=False
        )
    except:
        model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu"), strict=False
        )
    model.eval()
    return model


def get_tensor(image_bytes):
    my_transforms = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # image=Image.open(io.BytesIO(image_bytes))
    y, sr = librosa.load(io.BytesIO(image_bytes))
    S_full, phase = librosa.magphase(librosa.stft(y))
    img = np.stack((S_full,) * 3, axis=-1)
    PIL_image = Image.fromarray((img * 255).astype(np.uint8))
    return my_transforms(PIL_image).unsqueeze(0)
