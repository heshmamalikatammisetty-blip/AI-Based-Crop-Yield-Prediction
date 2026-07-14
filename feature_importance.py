import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/crop_yield.csv")

crop_encoder = LabelEncoder()
season_encoder = LabelEncoder()
state_encoder = LabelEncoder()

df["Crop"] = crop_encoder.fit_transform(df["Crop"])
df["Season"] = season_encoder.fit_transform(df["Season"])
df["State"] = state_encoder.fit_transform(df["State"])

X = df.drop("Yield", axis=1)
y = df["Yield"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance.sort_values().plot(kind="barh")

plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.show()