import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import classification_report

from src.dataset import FetalUltrasoundDataset, get_transforms
from src.model import create_fetal_model
from src.utils import get_device, save_checkpoint, set_seed
from src.evaluation import get_predictions

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
    set_seed(42) # Ensure reproducibility
    device = get_device()
    print(f"Using device: {device}")
    
    # Data loaders
    # We load the training split twice to have different transforms for train and val subsets
    train_dataset_for_split = FetalUltrasoundDataset(
        csv_path, data_dir, split='Train', 
        transform=get_transforms(is_train=True)
    )
    val_dataset_for_split = FetalUltrasoundDataset(
        csv_path, data_dir, split='Train', 
        transform=get_transforms(is_train=False)
    )
    
    # 80% train, 20% val
    num_train_total = len(train_dataset_for_split)
    val_size = int(0.2 * num_train_total)
    train_size = num_train_total - val_size
    
    # Generate indices for the split
    indices = list(range(num_train_total))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:train_size], indices[train_size:]
    
    train_subset = torch.utils.data.Subset(train_dataset_for_split, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset_for_split, val_indices)
    
    test_dataset = FetalUltrasoundDataset(
        csv_path, data_dir, split='Test', 
        transform=get_transforms(is_train=False)
    )
    
    # Apply WeightedRandomSampler for training (on the train subset)
    # Get labels for the training subset
    full_train_labels = [train_dataset_for_split.label_map[p] for p in train_dataset_for_split.data['Plane']]
    train_subset_labels = [full_train_labels[i] for i in train_indices]
    
    class_counts = np.bincount(train_subset_labels)
    class_weights = 1. / class_counts
    weights = [class_weights[label] for label in train_subset_labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, 
        sampler=sampler, num_workers=2
    )
    
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Model, Loss, Optimizer
    model = create_fetal_model(model_name=model_name, num_classes=len(train_dataset_for_split.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, "best_model.pth")
            print("Saved Best Model Checkpoint")

    # Final detailed evaluation on TEST SET
    print("\n--- Final Evaluation Report (Test Set) ---")
    model.load_state_dict(torch.load("best_model.pth", weights_only=True)['model_state_dict'])
    y_pred, y_true, _ = get_predictions(model, test_loader, device)
    print(classification_report(y_true, y_pred, target_names=train_dataset_for_split.classes))

if __name__ == "__main__":
    # Default paths matching the README structure
    main(
        data_dir="data/Images", 
        csv_path="data/FETAL_PLANES_DB_data.csv",
        epochs=15,
        batch_size=32,
        lr=1e-4,
        model_name='efficientnet_b0'
    )
