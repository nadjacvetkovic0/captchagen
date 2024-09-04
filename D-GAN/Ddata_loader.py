import os
import torch
from torchvision import datasets, transforms

def get_data_loader(image_dir, batch_size=64, image_size=100): #loaduje
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]) #radi nes
    
    dataset = datasets.ImageFolder(root=image_dir, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return data_loader
