import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(file_path="data/data.xlsx"):
    df = pd.read_excel(file_path, sheet_name=0)
    numeric_data = df.select_dtypes(include=[np.number])
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    return df, numeric_data, scaled_data, scaler
