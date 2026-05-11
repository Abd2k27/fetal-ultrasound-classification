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
    """
    Collects all predictions, true labels, and probabilities from a data loader.
    Used for final evaluation and error analysis.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            
            # Apply softmax to get confidence scores (0.0 to 1.0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, classes, save_path="confusion_matrix.png"):
    """
    Generates and saves a normalized confusion matrix.
    
    Normalization ('true') is crucial in medical imaging to see the recall per class 
    independently of class prevalence (the 'Other' class is often much larger).
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(12, 10))
    
    # Format as percentage for better clinical readability
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix (Recall per Class)')
    plt.savefig(save_path)
    plt.close()

def find_worst_predictions(y_true, y_pred, y_probs, classes, top_k=5):
    """
    Identifies the 'worst' predictions: cases where the model was very confident 
    but ultimately wrong. 
    
    In a clinical setting, these are high-risk errors that need manual review 
    to understand if the model is biased or if the image quality was poor.
    """
    errors = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            # Get the confidence score of the incorrect prediction
            conf = y_probs[i][y_pred[i]]
            errors.append({
                'index': i,
                'true': classes[y_true[i]],
                'pred': classes[y_pred[i]],
                'conf': conf
            })
            
    # Sort by confidence descending to find the most 'arrogant' mistakes
    errors.sort(key=lambda x: x['conf'], reverse=True)
    return errors[:top_k]

def run_gradcam(model, target_layer, input_tensor, target_category=None):
    """
    Executes the Grad-CAM (Gradient-weighted Class Activation Mapping) algorithm.
    
    Grad-CAM highlights the pixels in the input image that were most influential 
    in the model's decision for a specific class. This provides visual 
    'evidence' for clinicians.
    """
    # Initialize the Grad-CAM object with the model and target layer (e.g., last conv layer)
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    if target_category is None:
        targets = None
    else:
        # We can target a specific class to see why the model *didn't* choose it
        targets = [ClassifierOutputTarget(target_category)]
        
    # Generate the heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    return grayscale_cam

def visualize_gradcam(img_tensor, grayscale_cam, save_path="gradcam.png"):
    """
    Overlays the Grad-CAM heatmap onto the original image.
    """
    # 1. Reverse the ImageNet normalization to get the original pixels back
    img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    
    # 2. Blend the original image with the heatmap
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(visualization)
    plt.axis('off')
    plt.title('Grad-CAM: Model Focus Area')
    plt.savefig(save_path)
    plt.close()
