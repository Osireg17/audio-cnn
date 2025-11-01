from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import modal
import torchaudio
import torch.nn as nn
import torchaudio.transforms as T
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if self.transform:
            spectrogram = self.transform(waveform)
        else:
            spectrogram = waveform

        return spectrogram, row['label_idx']
    

def mixup_data(x, y):
    lam = np.random.beta(1.0, 1.0)  # More conservative mixup
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device) # shuffle the batch of audios
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam): # used to compute the loss for mixup data
    # What this does is to compute the loss for both the original and shuffled labels and weight them by lam and (1-lam)
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

@app.function(image=image, gpu="A10G", volumes={"/data": volume, "/models": model_volume}, timeout=60*60*3)
def train():
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Training started at {timestamp}")
    log_dir = f'/models/tensorboard_logs/run_{timestamp}'
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to {log_dir}")
    
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
        lr=0.001,
        weight_decay=0.0001
    )


    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.003,
        steps_per_epoch=len(train_dataloader),
        epochs=num_epochs,
        pct_start=0.3,
    )

    best_accuracy = 0.0 # Track the best validation accuracy
    
    print("Starting training loop...")
    for epoch in range(num_epochs):
        model.train()
        
        epoch_loss = 0.0 # accumulate loss batch wise
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            
            if np.random.rand() < 0.5: # apply mixup with a probability of 0.5
                data, targets_a, targets_b, lam = mixup_data(data, target)
                outputs = model(data)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                
            else:
                outputs = model(data)
                loss = criterion(outputs, target)
                
            optimizer.zero_grad() # remove the gradients from the previous step
            loss.backward() # compute the gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # clip gradients to prevent explosion
            optimizer.step() # update the weights
            scheduler.step() # update the learning rate
            
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(Loss=f'{loss.item():.4f}')
            
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_epoch_loss:.4f}")

        # Log training metrics to TensorBoard
        writer.add_scalar('Loss/Train', avg_epoch_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        model.eval()
        
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_dataloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1) # get the index of the max log-probability
                total += target.size(0)
                correct += (predicted == target).sum().item()


        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Log validation metrics to TensorBoard
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)
        writer.add_scalar('Accuracy/Best', best_accuracy, epoch)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': best_accuracy,
                'epoch': epoch,
                'classes': train_dataset.classes
            }, "/models/best_model.pth")
            print(f"Best model saved with accuracy: {best_accuracy:.4f}")
            
            
    print(f"Training complete!, best accuracy: {best_accuracy:.2f}%")

    # Close TensorBoard writer
    writer.close()
    print(f"TensorBoard logs saved to {log_dir}")
            
        
@app.local_entrypoint()
def main():
    train.remote()