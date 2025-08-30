from pathlib import Path
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import modal
import torchaudio
import torch.nn as nn
import torchaudio.transforms as T
from torch.optim.lr_scheduler import OneCycleLR

from model import AudioCNN

app = modal.App("audio-cnn")

image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"])
         .run_commands([
             "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
             "cd /tmp && unzip esc50.zip",
             "mkdir -p /opt/esc50-data",
             "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
             "rm -rf /tmp/esc50.zip /tmp/ESC-50-master"
         ])
         .add_local_python_source("model"))

volume = modal.Volume.from_name("esc50-data", create_if_missing=True)
model_volume = modal.Volume.from_name("esc-model", create_if_missing=True)

class ESC50Dataset(Dataset):
    def __init__(self, data_dir, metadata_file, split="train", transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata_file = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform
        
        if split == 'train':
            self.metadata = self.metadata_file[self.metadata_file['fold'] != 5].copy()
        else:
            self.metadata = self.metadata_file[self.metadata_file['fold'] == 5].copy()
            
            
        self.classes = sorted(self.metadata['category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.metadata.loc[:, 'label_idx'] = self.metadata['category'].map(self.class_to_idx)
        
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / row['filename']
        
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if waveform[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if self.transform:
            spectrogram = self.transform(waveform)
        else:
            spectrogram = waveform

        return spectrogram, row['label_idx']

@app.function(image=image, gpu="A10G", volumes={"/data": volume, "/models": model_volume}, timeout=60*60*3)
def train():
    esc50_dir = Path("/opt/esc50-data")
    
    train_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB(),
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=80)
    )


    val_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB(),
    )
    
    train_dataset = ESC50Dataset(
        data_dir=esc50_dir,
        metadata_file=esc50_dir / "meta" / "esc50.csv",
        split="train",
        transform=train_transform
    )

    val_dataset = ESC50Dataset(
        data_dir=esc50_dir,
        metadata_file=esc50_dir / "meta" / "esc50.csv",
        split="val",
        transform=val_transform
    )
    
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(
        num_classes=len(train_dataset.classes)
    )
    model.to(device)
    
    num_epochs = 100
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0005,
        weight_decay=0.01
    )


    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.02,
        steps_per_epoch=len(train_dataloader),
        epochs=num_epochs,
        pct_start=0.1,
    )

    best_accuracy = 0.0
    
    print("Starting training...")
    


@app.local_entrypoint()
def main():
    train.remote()