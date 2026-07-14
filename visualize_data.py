import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/crop_yield.csv")

top_crops = df.groupby("Crop")["Yield"].mean().sort_values(ascending=False).head(10)

top_crops.plot(kind="bar")

plt.title("Top 10 Crops by Yield")
plt.xlabel("Crop")
plt.ylabel("Average Yield")

plt.show()