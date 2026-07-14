import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("models/crop_yield_model.pkl")

st.title("🌾 AI-Based Crop Yield Prediction")
st.subheader("Project Information")
df = pd.read_csv("data/crop_yield.csv")

top_crops = (
    df.groupby("Crop")["Yield"]
      .mean()
      .sort_values(ascending=False)
      .head(10))
st.subheader("📈 Model Insights")
st.sidebar.success("Model Accuracy:90.62%")
st.write("""
The Random Forest model identified the most influential factors affecting crop yield:
- Area
- Production
- Annual Rainfall
- Fertilizer Usage
- Pesticide Usage
""")


st.subheader("📊 Top 10 Crops by Average Yield")

st.bar_chart(top_crops)
st.write("📊 Dataset Records: 19,689")
st.write("🤖 Best Model: Random Forest")
st.write("🎯 Accuracy: 90.62%")

crop_list=["Rice","Wheat","Maize"]
crop=st.selectbox("Select crop",crop_list)

state_list=["Andhra Pradesh","Tamil Nadu","Punjab"]
state=st.selectbox("Select State",state_list)

season_list=["Kharif","Rabi","Whole Year"]
season=st.selectbox("Select Season",season_list)

crop_year = st.number_input("Crop Year", value=2020)
area = st.number_input("Area", value=100.0)
production = st.number_input("Production", value=500.0)
annual_rainfall = st.number_input("Annual Rainfall", value=800.0)
fertilizer = st.number_input("Fertilizer", value=200.0)
pesticide = st.number_input("Pesticide", value=50.0)

if st.button("Predict Yield"):

    input_data = pd.DataFrame([[
        0,  # Crop
        crop_year,
        0,  # Season
        0,  # State
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

    prediction = model.predict(input_data)

    st.success(f"Predicted Crop Yield: {prediction[0]:.2f}")
    
st.markdown("---")
st.caption("AI-Based Crop Yield Prediction System | Built using Python, Scikit-Learn, Streamlit|Developed By                                                                                          " \
"Tammisetty Heshma Malika ")