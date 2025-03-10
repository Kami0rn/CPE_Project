import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import torchvision

import torch.nn as nn
import torch.optim as optim

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Generator
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 3*64*64),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 3, 64, 64)
        return img

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3*64*64, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 1)
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = [os.path.join(img_dir, img_name) for img_name in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, img_name))]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # Return a dummy label

# Hyperparameters
latent_dim = 100
batch_size = 64
n_epochs = 200
lr = 0.0001  # ปรับค่า learning rate
n_critic = 5
lambda_gp = 10  # ค่า lambda สำหรับ gradient penalty

# Data loading
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# อัปเดตเส้นทางไดเรกทอรีของภาพที่นี่
dataset = CustomImageDataset(img_dir='E:/CPE/1-2567/Project/Finale/Anime/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models and optimizers
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)
optimizer_G = optim.RMSprop(generator.parameters(), lr=lr)
optimizer_D = optim.RMSprop(discriminator.parameters(), lr=lr)

# Function to save generated images
def save_image(img_tensor, epoch, nrow=8, ncol=8, filename='generated_images.png'):
    img_tensor = img_tensor * 0.5 + 0.5  # Unnormalize
    img_grid = torchvision.utils.make_grid(img_tensor, nrow=nrow)
    plt.figure(figsize=(ncol, nrow))
    plt.imshow(img_grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.savefig(f'epoch_{epoch}_{filename}')
    plt.close()

# Gradient penalty function
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.randn((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Training
for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        
        # ย้ายภาพไปยังอุปกรณ์
        real_imgs = imgs.to(device)
        
        # ฝึก Discriminator
        optimizer_D.zero_grad()
        z = torch.randn(imgs.size(0), latent_dim).to(device)
        fake_imgs = generator(z).detach()
        real_validity = discriminator(real_imgs)
        fake_validity = discriminator(fake_imgs)
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        loss_D = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        loss_D.backward()
        optimizer_D.step()

        # ฝึก Generator ทุกๆ n_critic ขั้นตอน
        if i % n_critic == 0:
            optimizer_G.zero_grad()
            gen_imgs = generator(z)
            loss_G = -torch.mean(discriminator(gen_imgs))
            loss_G.backward()
            optimizer_G.step()

    print(f"[Epoch {epoch}/{n_epochs}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}]")
    
    # บันทึกภาพที่สร้างขึ้น
    save_image(gen_imgs, epoch)