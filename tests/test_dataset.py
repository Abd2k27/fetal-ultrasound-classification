import pytest
import pandas as pd
import os
from src.dataset import FetalUltrasoundDataset

def test_dataset_init(tmp_path):
    """
    Tests the dataset's ability to parse metadata and filter by split.
    Uses a temporary directory to avoid creating junk files.
    """
    # Create a mock CSV matching the expected project structure
    csv_content = "Image_name;Patient_Id;Plane;Brain_plane;Operator;US_Machine;Train/Test \n" \
                  "img1;1;Fetal abdomen;Not-applicable;Op1;Mach1;Train\n" \
                  "img2;2;Fetal brain;Trans-thalamic;Op2;Mach2;Test\n"
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(csv_content)
    
    # Verify Train split loading
    ds = FetalUltrasoundDataset(str(csv_path), img_dir=str(tmp_path), split='Train')
    assert len(ds) == 1
    assert ds.classes[0] == "Fetal abdomen"
    
    # Verify Test split loading and content
    ds_test = FetalUltrasoundDataset(str(csv_path), img_dir=str(tmp_path), split='Test')
    assert len(ds_test) == 1
    assert ds_test.data.iloc[0]['Plane'] == 'Fetal brain'
