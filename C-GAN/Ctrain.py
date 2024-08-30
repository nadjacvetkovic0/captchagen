import os
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from Cmodels import Generator, Discriminator
from Cdata_loader import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_gan(num_epochs=300, batch_size=64, lr=0.0001, betas=(0.5, 0.999), save_images_interval=5):
    dataloader = get_dataloader(batch_size=batch_size, image_size=(128, 128))  # 128x128 veličina slike
    
    # Initialize networks and move them to the GPU if available
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=betas)
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=betas)

    # Create a folder to save generated images
    if not os.path.exists('Training/TrainingMandarine2'):
        os.makedirs('Training/TrainingMandarine2')

    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # Real labels with smoothing
            real_labels = torch.full((batch_size,), 0.9, dtype=torch.float, device=device)  # Label smoothing 0.9
            fake_labels = torch.full((batch_size,), 0.1, dtype=torch.float, device=device)  # Label smoothing 0.1

            # --- Update Discriminator ---
            netD.zero_grad()
            output_real = netD(real_images)
            errD_real = criterion(output_real, real_labels)
            errD_real.backward()

            noise = torch.randn(batch_size, 100, 1, 1, device=device) * 0.8  # Smanjen šum
            fake_images = netG(noise)
            output_fake = netD(fake_images.detach())
            errD_fake = criterion(output_fake, fake_labels)
            errD_fake.backward()

            optimizerD.step()

            # --- Update Generator ---
            netG.zero_grad()
            fake_labels.fill_(0.9)  # Smoothed labels
            output = netD(fake_images)
            errG = criterion(output, fake_labels)
            errG.backward()

            optimizerG.step()

        print(f'Epoch [{epoch+1}/{num_epochs}] Loss D: {errD_real.item() + errD_fake.item()}, Loss G: {errG.item()}')

        # Save generated images at intervals
        if (epoch + 1) % save_images_interval == 0:
            save_generated_images(netG, epoch + 1)

def save_generated_images(generator, epoch, num_images=64):
    folder_name = f'Training/TrainingMandarine2/epoch_{epoch}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    noise = torch.randn(num_images, 100, 1, 1, device=device) * 0.8  # Smanjen šum
    with torch.no_grad():
        generated_images = generator(noise).detach().cpu()

    for i in range(num_images):
        image = generated_images[i]
        # Normalize the image to the range [0, 1]
        image = (image + 1) / 2  # Since Tanh output range is [-1, 1]
        image = torch.clamp(image, 0, 1)  # Ensure the image values are between 0 and 1
        plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
        plt.axis('off')
        plt.savefig(os.path.join(folder_name, f'banana_{i+1}.png'))
        plt.close()

if __name__ == "__main__":
    train_gan()
