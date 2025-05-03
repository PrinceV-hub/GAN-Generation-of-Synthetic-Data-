import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

def train_gan(generator, discriminator, data_scaled, latent_dim, epochs=5000, batch_size=64, lr=0.0002):
    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

    g_losses = []
    d_losses = []
    real_label = 1.
    fake_label = 0.

    for epoch in range(epochs):
        idx = np.random.randint(0, data_scaled.shape[0], batch_size)
        real_data = torch.tensor(data_scaled[idx], dtype=torch.float)
        noise = torch.randn(batch_size, latent_dim)
        fake_data = generator(noise)

        # Discriminator
        optimizer_D.zero_grad()
        loss_real = criterion(discriminator(real_data), torch.full((batch_size, 1), real_label))
        loss_fake = criterion(discriminator(fake_data.detach()), torch.full((batch_size, 1), fake_label))
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # Generator
        optimizer_G.zero_grad()
        loss_G = criterion(discriminator(fake_data), torch.full((batch_size, 1), real_label))
        loss_G.backward()
        optimizer_G.step()

        g_losses.append(loss_G.item())
        d_losses.append(loss_D.item())

        if epoch % 500 == 0:
            print(f"Epoch {epoch} | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

    # Save loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/gan_loss_plot.png")
    return generator
