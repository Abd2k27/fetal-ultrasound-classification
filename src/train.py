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
    """
    Standard PyTorch training loop for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        # Move data to GPU (MPS on Mac, CUDA on Linux/Windows)
        images, labels = images.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass (Backpropagation)
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update the progress bar with real-time stats
        pbar.set_postfix(loss=loss.item(), acc=100.*correct/total)
        
    return running_loss / total, 100. * correct / total

def validate(model, loader, criterion, device):
    """
    Standard PyTorch validation loop. No gradients are calculated here.
    """
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
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
    model_name: str = 'efficientnet_b0'
):
    """
    Main orchestration function for the training pipeline.
    """
    set_seed(42) # Ensure reproducibility across different runs
    device = get_device()
    print(f"Using device: {device}")
    
    # --- 1. Dataset & Splitting ---
    # We load the training split twice to apply different transforms (Augmentation for train, Simple for val)
    train_dataset_full = FetalUltrasoundDataset(
        csv_path, data_dir, split='Train', 
        transform=get_transforms(is_train=True)
    )
    val_dataset_full = FetalUltrasoundDataset(
        csv_path, data_dir, split='Train', 
        transform=get_transforms(is_train=False)
    )
    
    # 80% train, 20% validation split
    num_train_total = len(train_dataset_full)
    val_size = int(0.2 * num_train_total)
    train_size = num_train_total - val_size
    
    # Shuffle indices manually to ensure we use the same split for both 'train' and 'val' wrappers
    indices = list(range(num_train_total))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:train_size], indices[train_size:]
    
    train_subset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
    # Test set is kept completely separate (patient-level split from Zenodo)
    test_dataset = FetalUltrasoundDataset(
        csv_path, data_dir, split='Test', 
        transform=get_transforms(is_train=False)
    )
    
    # --- 2. Handling Class Imbalance ---
    # The 'Other' class is much larger than specific organs. 
    # WeightedRandomSampler ensures the model sees each organ equally often during training.
    full_train_labels = [train_dataset_full.label_map[p] for p in train_dataset_full.data['Plane']]
    train_subset_labels = [full_train_labels[i] for i in train_indices]
    
    class_counts = np.bincount(train_subset_labels)
    class_weights = 1. / class_counts
    weights = [class_weights[label] for label in train_subset_labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    # --- 3. Data Loaders ---
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, 
        sampler=sampler, num_workers=2
    )
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # --- 4. Model, Loss, Optimizer ---
    model = create_fetal_model(model_name=model_name, num_classes=len(train_dataset_full.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # AdamW is a robust optimizer with weight decay to prevent overfitting
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # ReduceLROnPlateau lowers the learning rate if validation loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    best_acc = 0.0
    
    # --- 5. Training Loop ---
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save only the best model based on validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, "best_model.pth")
            print("🌟 New Best Model Saved!")

    # --- 6. Final Evaluation ---
    # We load the absolute best weights before testing on the unseen data
    print("\n--- Final Evaluation Report (Test Set) ---")
    model.load_state_dict(torch.load("best_model.pth", weights_only=True)['model_state_dict'])
    y_pred, y_true, _ = get_predictions(model, test_loader, device)
    
    # classification_report gives us Precision, Recall, and F1-Score per class
    print(classification_report(y_true, y_pred, target_names=train_dataset_full.classes))

if __name__ == "__main__":
    # Standard entry point
    main(
        data_dir="data/Images", 
        csv_path="data/FETAL_PLANES_DB_data.csv",
        epochs=15,
        batch_size=32,
        lr=1e-4,
        model_name='efficientnet_b0'
    )
