import pytest
import torch
from src.model import create_fetal_model

def test_model_creation():
    num_classes = 6
    model = create_fetal_model(model_name='efficientnet_b0', num_classes=num_classes, pretrained=False)
    assert model is not None
    
def test_model_forward_pass():
    num_classes = 6
    model = create_fetal_model(model_name='efficientnet_b0', num_classes=num_classes, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, num_classes)

def test_model_different_architecture():
    # Test that we can swap architectures easily
    model = create_fetal_model(model_name='resnet18', num_classes=3, pretrained=False)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (1, 3)
