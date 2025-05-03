# === Import Required Libraries ===
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# === Load and preprocess data ===
df = pd.read_excel("data.xlsx", sheet_name=0)
data = df.select_dtypes(include=[np.number])
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

df.head()

row, columns = df.shape
print(f"Number of rows: {row} and number of columns: {columns}")

# === Define Generator and Discriminator ===
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# === Initialize Models and Parameters ===
latent_dim = 20
data_dim = data_scaled.shape[1]
generator = Generator(latent_dim, data_dim)
discriminator = Discriminator(data_dim)

criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

# === Training Loop ===
epochs = 5000
batch_size = 64
real_label = 1.
fake_label = 0.
g_losses = []
d_losses = []

for epoch in range(epochs):
    idx = np.random.randint(0, data_scaled.shape[0], batch_size)
    real_data = torch.tensor(data_scaled[idx], dtype=torch.float)
    noise = torch.randn(batch_size, latent_dim)
    fake_data = generator(noise)

    # Train Discriminator
    optimizer_D.zero_grad()
    loss_real = criterion(discriminator(real_data), torch.full((batch_size, 1), real_label))
    loss_fake = criterion(discriminator(fake_data.detach()), torch.full((batch_size, 1), fake_label))
    loss_D = loss_real + loss_fake
    loss_D.backward()
    optimizer_D.step()

    # Train Generator
    optimizer_G.zero_grad()
    loss_G = criterion(discriminator(fake_data), torch.full((batch_size, 1), real_label))
    loss_G.backward()
    optimizer_G.step()

    g_losses.append(loss_G.item())
    d_losses.append(loss_D.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

# === Plot Loss Curves ===
plt.figure(figsize=(10, 5))
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GAN Training Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("gan_loss_plot.png")
plt.show()

# === Generate and Save Synthetic Data ===
noise = torch.randn(data_scaled.shape[0], latent_dim)
synthetic_data = generator(noise).detach().numpy()
synthetic_data = scaler.inverse_transform(synthetic_data)
synthetic_df = pd.DataFrame(synthetic_data, columns=data.columns)
synthetic_df.to_csv("synthetic_data.csv", index=False)
print("Synthetic data saved to synthetic_data.csv")

print("\nSynthetic Data Shape:", synthetic_df.shape)
print("\nColumn Names:\n", synthetic_df.columns.tolist())
print("\nFirst 5 Rows:\n", synthetic_df.head())

# === Compare Real vs Synthetic Distributions ===
real_df = pd.DataFrame(data, columns=data.columns)
synthetic_df = pd.read_csv("synthetic_data.csv")

for column in real_df.columns:
    plt.figure(figsize=(8, 4))
    sns.kdeplot(real_df[column], label='Real', fill=True)
    sns.kdeplot(synthetic_df[column], label='Synthetic', fill=True)
    plt.title(f"Distribution: {column}")
    plt.xlabel(column)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Pearson Correlation Matrices and Heatmaps ===
real_corr = real_df.corr(method='pearson')
synthetic_corr = synthetic_df.corr(method='pearson')

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(real_corr, annot=True, cmap='coolwarm', square=True)
plt.title("Real Data Correlation")

plt.subplot(1, 2, 2)
sns.heatmap(synthetic_corr, annot=True, cmap='coolwarm', square=True)
plt.title("Synthetic Data Correlation")

plt.tight_layout()
plt.show()
