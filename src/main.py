from src.data_loader import load_and_preprocess
from src.models import Generator, Discriminator
from src.train import train_gan
from src.generate import generate_synthetic_data
from src.visualize import compare_distributions, correlation_heatmaps

if __name__ == "__main__":
    df, real_df, data_scaled, scaler = load_and_preprocess("data/data.xlsx")
    latent_dim = 20
    generator = Generator(latent_dim, data_scaled.shape[1])
    discriminator = Discriminator(data_scaled.shape[1])

    generator = train_gan(generator, discriminator, data_scaled, latent_dim)
    synthetic_df = generate_synthetic_data(generator, data_scaled.shape, latent_dim, scaler, real_df.columns)
    compare_distributions(real_df, synthetic_df)
    correlation_heatmaps(real_df, synthetic_df)
