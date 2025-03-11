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

# -------------------------------
#   DCGAN Generator
# -------------------------------
class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, feature_g=64):
        super(DCGANGenerator, self).__init__()
        # latent_dim -> feature_g*8 x 4 x 4 -> ... -> 3 x 64 x 64
        self.net = nn.Sequential(
            # Input Z: (N, latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, feature_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),
            # (feature_g*8) x 4 x 4
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            # (feature_g*4) x 8 x 8
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            # (feature_g*2) x 16 x 16
            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),
            # (feature_g) x 32 x 32
            nn.ConvTranspose2d(feature_g, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # (img_channels) x 64 x 64
        )

    def forward(self, z):
        return self.net(z)

# -------------------------------
#   DCGAN Discriminator
# -------------------------------
class DCGANDiscriminator(nn.Module):
    def __init__(self, img_channels=3, feature_d=64):
        super(DCGANDiscriminator, self).__init__()
        # input: (N, 3, 64, 64)
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feature_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_d) x 32 x 32
            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_d*2) x 16 x 16
            nn.Conv2d(feature_d * 2, feature_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_d*4) x 8 x 8
            nn.Conv2d(feature_d * 4, feature_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_d*8) x 4 x 4
            nn.Conv2d(feature_d * 8, 1, 4, 1, 0, bias=False),
            # output shape: (1) x 1 x 1
        )

    def forward(self, x):
        # Flatten ให้ได้สเกล scalar สุดท้ายสำหรับ WGAN
        # (N, 1, 1, 1) -> (N)
        return self.net(x).view(-1)

# -------------------------------
#   Custom Dataset
# -------------------------------
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = [os.path.join(img_dir, img_name) 
                          for img_name in os.listdir(img_dir) 
                          if os.path.isfile(os.path.join(img_dir, img_name))]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # Return a dummy label

# -------------------------------
#   Hyperparameters
# -------------------------------
latent_dim = 100
batch_size = 64
n_epochs = 5001
lr = 0.00005  # สามารถลองปรับเป็น 0.0001 หรือ 0.0002
n_critic = 5
lambda_gp = 10  # Gradient Penalty lambda

# -------------------------------
#   Data Loading
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # สำหรับ 3 แชนแนล
])

dataset = CustomImageDataset(img_dir='E:/CPE/1-2567/Project/Finale/Anime/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -------------------------------
#   Initialize models + Optimizers
# -------------------------------
generator = DCGANGenerator(latent_dim=latent_dim, img_channels=3, feature_g=64).to(device)
discriminator = DCGANDiscriminator(img_channels=3, feature_d=64).to(device)

optimizer_G = optim.RMSprop(generator.parameters(), lr=lr)
optimizer_D = optim.RMSprop(discriminator.parameters(), lr=lr)

# -------------------------------
#   Function: Save Generated Images
# -------------------------------
def save_image(img_tensor, epoch, nrow=8, ncol=8, filename='generated_images.png'):
    # Unnormalize: scale back to [0,1]
    img_tensor = img_tensor * 0.5 + 0.5  
    img_grid = torchvision.utils.make_grid(img_tensor, nrow=nrow)
    plt.figure(figsize=(ncol, nrow))
    plt.imshow(img_grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    
    # Ensure the directory exists
    save_dir = 'set4'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the image
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch}_{filename}'))
    plt.close()

# -------------------------------
#   Function: Gradient Penalty
# -------------------------------
def compute_gradient_penalty(D, real_samples, fake_samples):
    # ถ้าต้องการสุ่ม alpha แบบ Uniform ให้เปลี่ยนเป็น torch.rand(...)
    alpha = torch.randn((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
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

# -------------------------------
#   Training Loop
# -------------------------------
for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        
        # Move real images to device
        real_imgs = imgs.to(device)
        batch_size_now = real_imgs.size(0)

        # -------------------------
        #  Train Discriminator
        # -------------------------
        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = torch.randn(batch_size_now, latent_dim, 1, 1, device=device)
        
        # Generate a batch of images
        fake_imgs = generator(z).detach()  # .detach() to avoid gradient to G
        real_validity = discriminator(real_imgs)
        fake_validity = discriminator(fake_imgs)

        # Compute gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)

        # WGAN-GP loss for D
        loss_D = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        loss_D.backward()
        optimizer_D.step()

        # -------------------------
        #  Train Generator every n_critic steps
        # -------------------------
        if i % n_critic == 0:
            optimizer_G.zero_grad()
            gen_imgs = generator(z)  # regenerate from same z or new z
            loss_G = -torch.mean(discriminator(gen_imgs))
            loss_G.backward()
            optimizer_G.step()

    # Logging
    print(f"[Epoch {epoch}/{n_epochs}] [D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

    # Save sample generated images
    save_image(gen_imgs, epoch)
