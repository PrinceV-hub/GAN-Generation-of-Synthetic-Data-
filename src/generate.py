import torch
import pandas as pd

def generate_synthetic_data(generator, data_shape, latent_dim, scaler, columns):
    noise = torch.randn(data_shape[0], latent_dim)
    synthetic_data = generator(noise).detach().numpy()
    synthetic_data = scaler.inverse_transform(synthetic_data)
    df_synthetic = pd.DataFrame(synthetic_data, columns=columns)
    df_synthetic.to_csv("synthetic_data.csv", index=False)
    return df_synthetic
