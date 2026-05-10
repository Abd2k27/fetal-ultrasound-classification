import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from typing import List, Tuple
import os

def get_predictions(model, loader, device):
    """Collects all predictions and true labels from a loader."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, classes, save_path="confusion_matrix.png"):
    """Plots and saves a normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

def find_worst_predictions(y_true, y_pred, y_probs, classes, top_k=5):
    """Finds the most confident wrong predictions."""
    errors = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            # confidence of the wrong prediction
            conf = y_probs[i][y_pred[i]]
            errors.append({
                'index': i,
                'true': classes[y_true[i]],
                'pred': classes[y_pred[i]],
                'conf': conf
            })
            
    # Sort by confidence descending
    errors.sort(key=lambda x: x['conf'], reverse=True)
    return errors[:top_k]

def run_gradcam(model, target_layer, input_tensor, target_category=None):
    """Generates Grad-CAM visualization for a single image tensor."""
    # use_cuda is deprecated, the library detects device from the model
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    if target_category is None:
        targets = None
    else:
        targets = [ClassifierOutputTarget(target_category)]
        
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    return grayscale_cam

def visualize_gradcam(img_tensor, grayscale_cam, save_path="gradcam.png"):
    """Overlays Grad-CAM on the original image."""
    # Reverse normalization for visualization
    img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    plt.imshow(visualization)
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()
