import joblib
import pandas as pd

# Load model and encoders
model = joblib.load("models/crop_yield_model.pkl")
crop_encoder = joblib.load("models/crop_encoder.pkl")
season_encoder = joblib.load("models/season_encoder.pkl")
state_encoder = joblib.load("models/state_encoder.pkl")

# User input
crop = input("Enter Crop: ").strip()
season = input("Enter Season: ").strip()
state = input("Enter State: ").strip()

crop_year = int(input("Enter Crop Year: "))
area = float(input("Enter Area: "))
production = float(input("Enter Production: "))
annual_rainfall = float(input("Enter Annual Rainfall: "))
fertilizer = float(input("Enter Fertilizer: "))
pesticide = float(input("Enter Pesticide: "))

# Encode text values
crop_encoded = crop_encoder.transform([crop])[0]
season = season.strip()
for s in season_encoder.classes_:
    if s.strip().lower() == season.lower():
        season=s
        break
season_encoded= season_encoder.transform([season])[0]
state_encoded = state_encoder.transform([state])[0]

# Create input data
input_data = pd.DataFrame([[
    crop_encoded,
    crop_year,
    season_encoded,
    state_encoded,
    area,
    production,
    annual_rainfall,
    fertilizer,
    pesticide
]], columns=[
    "Crop",
    "Crop_Year",
    "Season",
    "State",
    "Area",
    "Production",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
])

# Predict
prediction = model.predict(input_data)

print("\nPredicted Crop Yield:", prediction[0])