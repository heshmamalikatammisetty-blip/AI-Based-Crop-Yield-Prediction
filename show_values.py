import pandas as pd

df = pd.read_csv("data/crop_yield.csv")

print("Crops:")
print(sorted(df["Crop"].unique()))

print("\nSeasons:")
print(sorted(df["Season"].unique()))

print("\nStates:")
print(sorted(df["State"].unique()))