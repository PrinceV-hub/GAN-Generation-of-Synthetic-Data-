# 🧬 Synthetic Tabular Data Generator using GAN

## 📌 Objective

This project aims to generate **synthetic tabular data** using a **Generative Adversarial Network (GAN)** and evaluate how closely it mimics the **real dataset**. The comparison is done via **feature-wise distribution plots** and **pairwise Pearson correlation heatmaps**.

---

## 📂 Dataset Description

- The dataset is loaded from an Excel file named `data.xlsx`.
- Only **numerical columns** are used for GAN training.
- Data is **normalized** using Min-Max Scaling for efficient GAN training.
- Synthetic data is saved as `synthetic_data.csv`.

---

## ⚙️ Methodology

### 🧠 GAN Architecture

#### Generator
- Input: Random noise vector (latent space)
- Output: Synthetic data vector with same feature dimensions as real data

#### Discriminator
- Input: A data vector (real or synthetic)
- Output: Probability indicating if the input is real

> The Generator learns to **fool** the Discriminator, while the Discriminator learns to **detect fakes**. They improve together in a zero-sum game.

---

## 🔁 Training Process

- **Epochs**: 5000  
- **Batch Size**: 64  
- **Loss Function**: Binary Cross-Entropy  
- **Optimizer**: Adam (`lr = 0.0002`)  
- **Monitoring**: Generator and Discriminator losses plotted over epochs

---

## 📊 Evaluation Metrics

### 1. Feature Distribution Comparison
- KDE plots for each feature show the distribution of:
  - Real data (Blue)
  - Synthetic data (Orange)
- Highlights how well synthetic features mimic real ones.

### 2. Pearson Correlation Heatmaps
- Correlation matrices are computed for:
  - Real dataset
  - Synthetic dataset
- Heatmaps visualize inter-feature relationships and structural similarity.

---

## 🖼️ Visual Outputs

- ✅ KDE Distribution Plots for real vs synthetic features  
- ✅ Pearson Correlation Heatmaps  
- ✅ Generator vs Discriminator Loss Curve

You can find the generated visuals in the `outputs/` folder.

---

## 📁 Project Structure



---

## 🚀 How to Run

### 1. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```
``` Run the code
python main.py
