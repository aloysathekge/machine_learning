"""
PyTorch Data Pipeline: Complete Guide
======================================

This guide covers everything about preparing data for PyTorch models:
1. Dataset vs DataLoader
2. Built-in datasets
3. Custom datasets
4. Data transformations
5. Efficient data loading
6. Common patterns and best practices
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
import numpy as np
from PIL import Image
import os

print("="*70)
print("PART 1: DATASET vs DATALOADER - The Basics")
print("="*70)

"""
KEY CONCEPT:
- Dataset: Stores your data (images, labels, etc.) and knows how to get individual samples
- DataLoader: Wraps a Dataset and handles batching, shuffling, parallel loading

Think of it like:
- Dataset = Your photo album (knows where each photo is)
- DataLoader = Your friend who grabs photos in batches and shuffles them
"""

# Example 1: Simple custom dataset
class SimpleDataset(Dataset):
    """
    Every custom Dataset must implement 3 methods:
    - __init__: Initialize your data
    - __len__: Return the total number of samples
    - __getitem__: Return one sample at index idx
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # This is called when you do dataset[idx]
        return self.data[idx], self.labels[idx]

# Create dummy data
dummy_data = torch.randn(100, 3, 32, 32)  # 100 images, 3 channels, 32x32
dummy_labels = torch.randint(0, 10, (100,))  # 100 labels (0-9)

dataset = SimpleDataset(dummy_data, dummy_labels)

print(f"Dataset size: {len(dataset)}")
print(f"First sample shape: {dataset[90][0].shape}")
print(f"First label: {dataset[90][1]}")