import os
from PIL import Image
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def load_images(root_dir, image_size=(128, 128)):  # Promenjeno na 128x128
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = datasets.DatasetFolder(root=root_dir, loader=datasets.folder.default_loader, extensions=['.jpg', '.png'], transform=transform)
    return dataset

def get_dataloader(batch_size=64, image_size=(128, 128)):  # Promenjeno na 128x128
    dataset = load_images('ProjektniRN/DataSets/Lemon', image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader