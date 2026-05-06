import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from src.dataset import FetalUltrasoundDataset, get_transforms
from src.model import create_fetal_model
from src.utils import get_device, save_checkpoint

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix(loss=loss.item(), acc=100.*correct/total)
        
    return running_loss / total, 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / total, 100. * correct / total

def main(
    data_dir: str,
    csv_path: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    model_name: str = 'efficientnet_b0'
):
    device = get_device()
    print(f"Using device: {device}")
    
    # Data loaders
    train_dataset = FetalUltrasoundDataset(
        csv_path, data_dir, split='Train', 
        transform=get_transforms(is_train=True)
    )
    test_dataset = FetalUltrasoundDataset(
        csv_path, data_dir, split='Test', 
        transform=get_transforms(is_train=False)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Model, Loss, Optimizer
    model = create_fetal_model(model_name=model_name, num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, "best_model.pth")
            print("Saved Best Model Checkpoint")

if __name__ == "__main__":
    # Example usage (uncomment and adjust paths to run)
    # main(data_dir="data/Images", csv_path="data/FETAL_PLANES_DB_data.csv")
    pass
