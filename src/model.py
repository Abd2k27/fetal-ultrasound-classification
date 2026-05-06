import timm
import torch
import torch.nn as nn

def create_fetal_model(model_name: str = 'efficientnet_b0', num_classes: int = 6, pretrained: bool = True):
    """
    Creates a fetal ultrasound classification model using timm.
    
    Args:
        model_name: Name of the architecture (default: efficientnet_b0).
        num_classes: Number of output classes (default: 6).
        pretrained: Whether to load pre-trained ImageNet weights.
        
    Returns:
        nn.Module: The constructed model.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    
    # Optional: Add clinical context metadata or specific layers if needed
    # For now, a standard fine-tuning setup is robust.
    
    return model

if __name__ == "__main__":
    # Quick test to verify model creation and output shape
    m = create_fetal_model()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = m(dummy_input)
    print(f"Model: efficientnet_b0")
    print(f"Output shape: {output.shape}")
    assert output.shape == (1, 6)
    print("Model creation successful!")
