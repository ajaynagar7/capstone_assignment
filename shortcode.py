import pandas as pd
import numpy as np


# df = pd.read_csv("data/raw_shipments.csv")
# print(df.columns.to_list())

# from src.shipment_prep import clean_shipments, build_features
# features, delivery_hours = build_features(clean_shipments(df)[0], pd.read_csv("data/hubs.csv"))
# print(features.columns)

df1 = pd.read_csv("data/new_shipments.csv")
print(df1.info())