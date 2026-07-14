import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("data/crop_yield.csv")

# Convert text columns to numbers
crop_encoder = LabelEncoder()
season_encoder = LabelEncoder()
state_encoder = LabelEncoder()

df["Crop"] = crop_encoder.fit_transform(df["Crop"])
df["Season"] = season_encoder.fit_transform(df["Season"])
df["State"] = state_encoder.fit_transform(df["State"])

# Features
X = df.drop("Yield", axis=1)

# Target
y = df["Yield"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
score = model.score(X_test, y_test)

print("Model Score:", score)

#Save model
joblib.dump(model,"models/crop_yield_model.pkl")
joblib.dump(crop_encoder, "models/crop_encoder.pkl")
joblib.dump(season_encoder, "models/season_encoder.pkl")
joblib.dump(state_encoder, "models/state_encoder.pkl")
print("Model saved successfully!")