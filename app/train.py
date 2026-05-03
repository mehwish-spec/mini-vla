import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import mlflow
import mlflow.pytorch
import numpy as np
from sklearn.metrics import f1_score, classification_report
import os
import time
from PIL import Image
from torchvision import transforms
import pandas as pd
from app.model import VLAModel, ACTION_CLASSES

def collate_fn(batch):
    images, instructions, labels = zip(*batch)
    return list(images), list(instructions), torch.tensor(labels)

class VLADataset(torch.utils.data.Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        image = self.transform(image)
        return image, row['instruction'], int(row['label'])

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_idx, (images, instructions, labels) in enumerate(loader):
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images, instructions)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if batch_idx % 10 == 0:
            print(f'  Batch {batch_idx}/{len(loader)} loss: {loss.item():.4f}')

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, f1

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, instructions, labels in loader:
            labels = labels.to(device)
            logits = model(images, instructions)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return avg_loss, f1, accuracy, all_preds, all_labels

def train(epochs=10, batch_size=16, lr=1e-3):
    device = torch.device('cpu')
    print(f'Training on: {device}')

    train_dataset = VLADataset('data/processed/train.csv')
    val_dataset = VLADataset('data/processed/val.csv')

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            collate_fn=collate_fn)

    model = VLAModel(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    mlflow.set_experiment('mini-vla')
    best_f1 = 0

    with mlflow.start_run():
        mlflow.log_params({
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'model': 'CLIP-ViT-B32 + fusion head',
            'trainable_params': 689669
        })

        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            train_loss, train_f1 = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_f1, val_acc, preds, labels = eval_epoch(
                model, val_loader, criterion, device
            )
            scheduler.step()

            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_f1': train_f1,
                'val_loss': val_loss,
                'val_f1': val_f1,
                'val_accuracy': val_acc,
            }, step=epoch)

            print(f'  Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}')
            print(f'  Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}')

            if val_f1 > best_f1:
                best_f1 = val_f1
                os.makedirs('models/checkpoints', exist_ok=True)
                torch.save(model.state_dict(), 'models/checkpoints/best_model.pt')
                print(f'  New best model saved! F1: {best_f1:.4f}')

        print(f'Training complete! Best F1: {best_f1:.4f}')
        mlflow.log_metric('best_val_f1', best_f1)

        report = classification_report(labels, preds, target_names=ACTION_CLASSES)
        print('Classification Report:')
        print(report)

    return model, best_f1

if __name__ == '__main__':
    model, best_f1 = train(epochs=5, batch_size=16, lr=1e-3)
    print(f'Done! Best F1: {best_f1:.4f}')
