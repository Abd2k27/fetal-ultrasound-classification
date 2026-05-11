import pytest
import torch
from src.model import create_fetal_model

def test_model_creation():
    """Verifies that the model can be instantiated with custom classes."""
    num_classes = 6
    model = create_fetal_model(model_name='efficientnet_b0', num_classes=num_classes, pretrained=False)
    assert model is not None
    
def test_model_forward_pass():
    """Tests the model's ability to process a batch of dummy images and return correct shapes."""
    num_classes = 6
    model = create_fetal_model(model_name='efficientnet_b0', num_classes=num_classes, pretrained=False)
    
    # Batch of 2 images, 3 channels, 224x224 resolution
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    
    assert output.shape == (2, num_classes)

def test_model_different_architecture():
    """Confirms that the model creation function is flexible to different architectures."""
    # Test with ResNet18 as an alternative
    model = create_fetal_model(model_name='resnet18', num_classes=3, pretrained=False)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    
    assert output.shape == (1, 3)
