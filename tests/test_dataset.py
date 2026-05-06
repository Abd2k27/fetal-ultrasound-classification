import pytest
import pandas as pd
import os
from src.dataset import FetalUltrasoundDataset

def test_dataset_init(tmp_path):
    # Create a mock CSV
    csv_content = "Image_name;Patient_Id;Plane;Brain_plane;Operator;US_Machine;Train/Test \n" \
                  "img1;1;Abdomen;Not-applicable;Op1;Mach1;Train\n" \
                  "img2;2;Brain;Trans-thalamic;Op2;Mach2;Test\n"
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(csv_content)
    
    # We won't test image loading here as it requires real files, 
    # but we can test the metadata parsing.
    ds = FetalUltrasoundDataset(str(csv_path), img_dir=str(tmp_path), split='Train')
    assert len(ds) == 1
    assert ds.classes[0] == "Abdomen"
    
    ds_test = FetalUltrasoundDataset(str(csv_path), img_dir=str(tmp_path), split='Test')
    assert len(ds_test) == 1
    assert ds_test.data.iloc[0]['Plane'] == 'Brain'
