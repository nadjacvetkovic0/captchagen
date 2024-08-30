import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loader(image_dir,batch_size, image_size):
    image_dir = 'cCaptcha4d/BananeReal/epoch_90 NAJBOLJA'  # Folder containing your images
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    dataset = datasets.ImageFolder(root=image_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return data_loader
