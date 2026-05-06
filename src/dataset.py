import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, List, Optional, Callable

class FetalUltrasoundDataset(Dataset):
    """
    Custom Dataset for Fetal Ultrasound Plane Classification.
    Parses the FETAL_PLANES_DB_data.csv and loads images.
    """
    CLINICAL_CLASSES = [
        "Abdomen", "Brain", "Femur", "Thorax", "Cervix", "Other"
    ]

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        split: str = "Train",
        transform: Optional[Callable] = None,
        label_map: Optional[dict] = None
    ):
        """
        Args:
            csv_path: Path to the metadata CSV file.
            img_dir: Directory containing the ultrasound images.
            split: "Train" or "Test" to filter the dataset.
            transform: torchvision transforms to apply.
            label_map: Optional mapping from class name to integer.
        """
        self.img_dir = img_dir
        self.transform = transform
        
        # Load metadata
        df = pd.read_csv(csv_path, sep=';') # Original Zenodo CSV uses ';' as delimiter
        
        # Filter by split
        self.data = df[df['Train/Test '] == split].reset_index(drop=True)
        
        # Create label mapping if not provided
        if label_map:
            self.label_map = label_map
        else:
            self.label_map = {name: i for i, name in enumerate(self.CLINICAL_CLASSES)}
            
        self.classes = self.CLINICAL_CLASSES

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        img_name = self.data.iloc[idx]['Image_name']
        plane_name = self.data.iloc[idx]['Plane']
        
        img_path = os.path.join(self.img_dir, f"{img_name}.png")
        
        # Load image and ensure it's RGB (even if grayscale, most pre-trained models expect 3 channels)
        image = Image.open(img_path).convert("RGB")
        
        label = self.label_map[plane_name]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(img_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """
    Returns appropriate transforms for training and validation.
    Includes augmentations for training to improve generalization.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
