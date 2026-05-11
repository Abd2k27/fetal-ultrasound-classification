import timm
import torch
import torch.nn as nn

def create_fetal_model(model_name: str = 'efficientnet_b0', num_classes: int = 6, pretrained: bool = True):
    """
    Creates a fetal ultrasound classification model using the `timm` (PyTorch Image Models) library.
    
    Why EfficientNet-B0?
    EfficientNet is highly efficient (hence the name), offering a great balance between 
    accuracy and computational cost. For medical imaging on local machines (like Mac M1/M2), 
    it provides fast inference without sacrificing performance on complex anatomical patterns.
    
    Args:
        model_name: Name of the architecture (default: efficientnet_b0).
        num_classes: Number of output classes (anatomical planes).
        pretrained: Whether to load weights pre-trained on ImageNet. Pre-training is 
                    highly recommended as it allows the model to leverage basic 
                    feature detection (edges, textures) learned from millions of images.
        
    Returns:
        nn.Module: The constructed model.
    """
    
    # We use timm.create_model to handle the heavy lifting of architecture setup
    # and automatic replacement of the final fully connected layer to match our num_classes.
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    
    # The 'num_classes' parameter automatically replaces the final linear layer (the 'head')
    # so the model outputs a vector of size 6 (the number of clinical planes we have).
    
    return model

if __name__ == "__main__":
    # Smoke test: verifies model creation and correct output dimensions
    m = create_fetal_model(pretrained=False) # Skip download for quick test
    dummy_input = torch.randn(1, 3, 224, 224)
    output = m(dummy_input)
    
    print(f"Model: efficientnet_b0")
    print(f"Output shape: {output.shape} (Expected: [1, 6])")
    
    assert output.shape == (1, 6)
    print("✅ Model creation successful!")
