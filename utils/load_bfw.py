import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import logging

logging.basicConfig(level=logging.INFO)

class BFWDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.transform = transform
        self.attributes = pd.read_csv(csv_file)
        self.root_dir = root_dir
        
        # Extract relevant columns
        self.image_paths_1 = self.attributes['p1']
        self.image_paths_2 = self.attributes['p2']
        self.labels = self.attributes['label']
    
    def __len__(self):
        return len(self.attributes)

    def __getitem__(self, idx):
        img_path_1 = os.path.join(self.root_dir, self.image_paths_1[idx])
        img_path_2 = os.path.join(self.root_dir, self.image_paths_2[idx])
        label = self.labels[idx]
        
        try:
            image_1 = Image.open(img_path_1).convert('RGB')
            image_2 = Image.open(img_path_2).convert('RGB')
        except Exception as e:
            logging.error(f"Failed to load images: {img_path_1}, {img_path_2} | Error: {e}")
            return None
        
        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        else:
            # Convert to numpy if no transform is applied
            image_1 = np.array(image_1)
            image_2 = np.array(image_2)
        
        # Extract all attributes for the pair
        attributes = self.attributes.loc[idx].to_dict()
        
        return (image_1, image_2), label, attributes


def collate_fn(batch):
    # Remove failed loads (None)
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def get_bfw_dataloader(csv_file, root_dir, batch_size=32, transform=None, shuffle=True):
    dataset = BFWDataset(csv_file, root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=collate_fn)
    return dataloader