# Build train/val/test numpy arrays from the combined sentiment parquet.


import numpy as np
import pandas as pd


df = pd.read_parquet("combined-sentiment-3class.parquet")


print("Unique splits:", df["split"].unique())


splits = {
    split_name: df[df["split"] == split_name].reset_index(drop=True)
    for split_name in df["split"].unique()
}


X_train = splits["train"]["input"].to_numpy()
y_train = splits["train"]["label"].to_numpy()


X_val = splits["val"]["input"].to_numpy()
y_val = splits["val"]["label"].to_numpy()


X_test = splits["test"]["input"].to_numpy()
y_test = splits["test"]["label"].to_numpy()


