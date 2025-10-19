# ----------------------------------------------------------------------
# Author: Adapted by ChatGPT
# CSV-based DataLoader for HiFi-IFDL
# ----------------------------------------------------------------------
import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CSVDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        img_path = os.path.join(self.img_dir, row['Path'])
        label = int(row['Label'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        mask = torch.zeros((256, 256))  # CSV 没有 mask，就用全 0
        return image, mask, label, row['Path']


def train_dataset_loader_init(csv_file, img_dir, batch_size=8, shuffle=True):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = CSVDataset(csv_file, img_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    return loader


def infer_dataset_loader_init(csv_file, img_dir, shuffle=True, bs=8):
    return train_dataset_loader_init(csv_file, img_dir, batch_size=bs, shuffle=shuffle)
