import torch
import torch.optim as optim
import torch.nn as nn
import os
from Dmodels import DGenerator, DDiscriminator
from Ddata_loader import get_data_loader
from torchvision.utils import save_image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_minimal_noise(images, noise_factor=0.001):
    # Dodavanje vrlo malo šuma
    noisy_images = images + noise_factor * torch.randn_like(images)
    noisy_images = torch.clip(noisy_images, -1.0, 1.0)  # Ograničava vrednosti na opseg [-1, 1]
    return noisy_images

def resize_images(images, size=(100, 100)):
    resize_transform = transforms.Resize(size)
    resized_images = torch.stack([resize_transform(img) for img in images])
    return resized_images

def train_DGAN(image_dir, epochs=50, batch_size=64, lr=0.0002):
    data_loader = get_data_loader(image_dir, batch_size)
    
    generator = DGenerator().to(device)
    discriminator = DDiscriminator().to(device)
    
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(data_loader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # Dodaj minimalan noise (blago zamućenje) na originalne slike
            noisy_real_images = add_minimal_noise(real_images)
            
            # Labels - prilagođavanje veličine labela
            real_labels = torch.ones(batch_size, 1, 12, 12).to(device)
            fake_labels = torch.zeros(batch_size, 1, 12, 12).to(device)
            
            # Train Discriminator
            optimizer_d.zero_grad()
            outputs = discriminator(noisy_real_images)
            d_loss_real = criterion(outputs, real_labels)
            d_loss_real.backward()
            
            noise = torch.randn(batch_size, 3, 100, 100).to(device) * 0.1  # Minimalan šum
            fake_images = generator(noisy_real_images + noise)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            d_loss_fake.backward()
            optimizer_d.step()
            
            # Train Generator
            optimizer_g.zero_grad()
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_g.step()
        
        print(f'Epoch [{epoch+1}/{epochs}], d_loss: {d_loss_real.item()+d_loss_fake.item():.4f}, g_loss: {g_loss.item():.4f}')
        
        # Čuvanje slika svaku 5. epohu
        if (epoch+1) % 5 == 0:
            epoch_folder = f'D-CAPTCHA/BANANE/epoch_{epoch+1}'
            os.makedirs(epoch_folder, exist_ok=True)
            resized_fake_images = resize_images(fake_images)  # Promeni veličinu na 100x100 piksela
            for idx, img in enumerate(resized_fake_images):
                save_image(img, os.path.join(epoch_folder, f'fake_image_{idx+1}.png'))

train_DGAN(image_dir='cCaptcha4d/BananeReal/epoch_90 NAJBOLJA', epochs=250, batch_size=64, lr=0.0002)
