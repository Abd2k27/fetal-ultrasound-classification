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
    
    This class handles:
    1. Parsing the Zenodo CSV metadata.
    2. Filtering data into Train/Test splits.
    3. Mapping clinical class names to numeric labels.
    4. Loading images from disk and applying transformations.
    """
    
    # Standard anatomical planes used in prenatal screenings
    CLINICAL_CLASSES = [
        "Fetal abdomen", "Fetal brain", "Fetal femur", "Fetal thorax", "Maternal cervix", "Other"
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
        
        # Load metadata using ';' as the delimiter (standard for this dataset)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
        df = pd.read_csv(csv_path, sep=';')
        
        # Split filtering logic:
        # The original Zenodo dataset uses a column named 'Train ' (with a trailing space)
        # where 1 indicates Training data and 0 indicates Testing data.
        if 'Train ' in df.columns:
            split_val = 1 if split == "Train" else 0
            self.data = df[df['Train '] == split_val].reset_index(drop=True)
        elif 'Train/Test ' in df.columns:
            # Compatibility layer for custom or mocked CSVs
            self.data = df[df['Train/Test '] == split].reset_index(drop=True)
        else:
            available_cols = ", ".join(df.columns)
            raise KeyError(f"Could not find split column ('Train ' or 'Train/Test ') in {csv_path}. Available columns: {available_cols}")
        
        # Initialize label mapping (e.g., {"Fetal brain": 1, ...})
        if label_map:
            self.label_map = label_map
        else:
            self.label_map = {name: i for i, name in enumerate(self.CLINICAL_CLASSES)}
            
        self.classes = self.CLINICAL_CLASSES

    def __len__(self) -> int:
        """Returns the total number of images in the current split."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        """
        Loads and returns the image and its corresponding label at the given index.
        """
        img_name = self.data.iloc[idx]['Image_name']
        plane_name = self.data.iloc[idx]['Plane']
        
        # Images are stored as .png files
        img_path = os.path.join(self.img_dir, f"{img_name}.png")
        
        # We convert to RGB to ensure compatibility with pre-trained models (e.g., EfficientNet)
        # even though ultrasound images are natively grayscale.
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Error loading image {img_path}: {e}")
        
        label = self.label_map[plane_name]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(img_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """
    Defines the image processing pipeline.
    
    For Training: Adds augmentations (rotation, translation, color jitter) to make the 
    model robust to variations in ultrasound probe positioning and machine settings.
    
    For Validation/Test: Only resizes and normalizes the image.
    """
    # Standard ImageNet normalization values
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            # RandomAffine helps simulate different probe angles and positions
            transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),
            # ColorJitter handles variations in ultrasound gain/contrast
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])
