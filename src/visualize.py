import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compare_distributions(real_df, synthetic_df):
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

def correlation_heatmaps(real_df, synthetic_df):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(real_df.corr(), annot=True, cmap='coolwarm', square=True)
    plt.title("Real Data Correlation")

    plt.subplot(1, 2, 2)
    sns.heatmap(synthetic_df.corr(), annot=True, cmap='coolwarm', square=True)
    plt.title("Synthetic Data Correlation")
    plt.tight_layout()
    plt.show()
