import joblib
season_encoder = joblib.load("models/season_encoder.pkl")
print(season_encoder.classes_)