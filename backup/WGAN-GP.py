import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import spectral_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
#   DCGAN Generator (Modified)
# -------------------------------
class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, feature_g=64):
        super(DCGANGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# -------------------------------
#   WGAN-GP Discriminator (Critic) with Spectral Normalization
# -------------------------------
class WGANDiscriminator(nn.Module):
    def __init__(self, img_channels=3, feature_d=64):
        super(WGANDiscriminator, self).__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(img_channels, feature_d, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(feature_d * 2, feature_d * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(feature_d * 4, feature_d * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(feature_d * 8, 1, 4, 1, 0, bias=False)),
        )

    def forward(self, x):
        return self.net(x).view(-1)

# -------------------------------
#   Weight Initialization
# -------------------------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# -------------------------------
#   Custom Dataset (No changes)
# -------------------------------
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = [os.path.join(img_dir, img_name) for img_name in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, img_name))]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0

# -------------------------------
#   Hyperparameters (Adjusted)
# -------------------------------
latent_dim = 100
batch_size = 128
n_epochs = 5001
lr = 0.00005  # Adjusted learning rate
n_critic = 3
lambda_gp = 10

# -------------------------------
#   Data Loading (No changes)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = CustomImageDataset(img_dir='C:/Program Files/Anime/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -------------------------------
#   Initialize Models
# -------------------------------
generator = DCGANGenerator(latent_dim=latent_dim).to(device)
discriminator = WGANDiscriminator().to(device)

# Apply weight initialization
generator.apply(weights_init)
discriminator.apply(weights_init)

# -------------------------------
#   Optimizers (Changed to Adam)
# -------------------------------
optimizer_G = optim.Adam(generator.parameters(), lr=0.00001, betas=(0.5, 0.9))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.9))

# -------------------------------
#   Directories for saving
# -------------------------------
save_dir_images = 'set6'
save_dir_models = 'model_checkpoints6'
save_dir_plots = 'plots6'  # New directory for plots
os.makedirs(save_dir_images, exist_ok=True)
os.makedirs(save_dir_models, exist_ok=True)
os.makedirs(save_dir_plots, exist_ok=True)  # Create plots directory

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
    plt.savefig(os.path.join(save_dir_images, f'epoch_{epoch}_{filename}'))
    plt.close()

# -------------------------------
#   Gradient Penalty (Fixed)
# -------------------------------
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device, requires_grad=True)  # Fixed: use rand instead of randn
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),  # Correct grad_outputs
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# -------------------------------
#   Training Loop (Improved)
# -------------------------------
# Lists to store losses
G_losses = []
D_losses = []

for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        real_imgs = imgs.to(device)
        batch_size_now = real_imgs.size(0)

        # Train Discriminator
        optimizer_D.zero_grad()
        
        # Generate new noise
        z = torch.randn(batch_size_now, latent_dim, 1, 1, device=device)
        fake_imgs = generator(z).detach()
        
        # Compute losses
        real_validity = discriminator(real_imgs)
        fake_validity = discriminator(fake_imgs)
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        loss_D = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        
        # Gradient clipping for stability
        loss_D.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        optimizer_D.step()

        # Train Generator
        if i % n_critic == 0:
            optimizer_G.zero_grad()
            # Generate new noise for generator
            z = torch.randn(batch_size_now, latent_dim, 1, 1, device=device)
            gen_imgs = generator(z)
            loss_G = -torch.mean(discriminator(gen_imgs))
            loss_G.backward()
            optimizer_G.step()
    
    # Save losses
    G_losses.append(loss_G.item())
    D_losses.append(loss_D.item())

    # Save generated images
    save_image(gen_imgs, epoch)

    # Logging and saving
    print(f"[Epoch {epoch}/{n_epochs}] [D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")
    if epoch % 10 == 0:
        # Save models
        torch.save(generator.state_dict(), os.path.join(save_dir_models, f"generator_epoch_{epoch}.pth"))
        torch.save(discriminator.state_dict(), os.path.join(save_dir_models, f"discriminator_epoch_{epoch}.pth"))

        # Plot and save loss graph
        plt.figure(figsize=(10, 5))
        plt.plot(G_losses, label="Generator Loss")
        plt.plot(D_losses, label="Discriminator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Generator and Discriminator Loss")
        plt.legend()
        plt.savefig(os.path.join(save_dir_plots, f"loss_plot_epoch_{epoch}.png"))  # Save in plots directory
        plt.close()