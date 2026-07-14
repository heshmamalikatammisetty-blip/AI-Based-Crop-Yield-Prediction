import pandas as pd
df = pd.read_csv("data/crop_yield.csv")
print("Missing Values:")
print(df.isnull().sum())