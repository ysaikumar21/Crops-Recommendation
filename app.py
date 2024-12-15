#################### Files Required For Deployment ################################

# UI : implemented by python streamlit library
# Trained Model Files: Saved pkl files (crops.pkl)
# Logic Code connecting UI & Pkl files

################# UI ##########################
import streamlit as st  # pip install streamlit
import pandas as pd
import sklearn
import pickle
import warnings
warnings.filterwarnings("ignore")

# App Header
st.header("🌱 Crop Recommendation 🌾")

st.write("📊 **Predictive Model Built on Below Sample Data**")

# Load dataset for display and range extraction
try:
    df = pd.read_csv("Crop_recommendation.csv")
    st.dataframe(df.head())
except FileNotFoundError:
    st.error("❌ Dataset file 'Modified_Crop_recommendation.csv' not found. Please upload the file.")
    st.stop()

# Input fields
st.write("🌟 **Enter the following input values:**")
col1, col2 = st.columns(2)

with col1:
    n = st.number_input(f"Enter Nitrogen Value (Min: {df.N.min()} to Max: {df.N.max()}) 🧪", min_value=df.N.min(), max_value=df.N.max())

with col2:
    p = st.number_input(f"Enter Phosphorus Value (Min: {df.P.min()} to Max: {df.P.max()}) 🧪", min_value=df.P.min(), max_value=df.P.max())

col3, col4 = st.columns(2)

with col3:
    k = st.number_input(f"Enter Potassium (K) Value (Min: {df.K.min()} to Max: {df.K.max()}) 🪴", min_value=df.K.min(), max_value=df.K.max())

with col4:
    t = st.number_input(f"Enter Temperature (T) (Min: {df.temperature.min()} to Max: {df.temperature.max()}) 🌡️", 
                        min_value=df.temperature.min(), max_value=df.temperature.max())

col5, col6 = st.columns(2)
with col5:
    h = st.number_input(f"Enter Humidity (H) (Min: {df.humidity.min()} to Max: {df.humidity.max()}) 💧", 
                        min_value=df.humidity.min(), max_value=df.humidity.max())

with col6:
    ph = st.number_input(f"Enter pH Value (Min: {df.ph.min()} to Max: {df.ph.max()}) ⚗️", 
                        min_value=df.ph.min(), max_value=df.ph.max())

r = st.number_input(f"Enter Rainfall Value (Min: {df.rainfall.min()} to Max: {df.rainfall.max()}) 🌧️", 
                    min_value=df.rainfall.min(), max_value=df.rainfall.max())

# Prepare input data
xdata = [n, p, k, t, h, ph, r]

###################### Writing Prediction Logic ######################

# Load model
try:
    with open('crops.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"❌ File not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

# Map predictions to crop names
crop_mapping = {
    0: '🌾 rice', 1: '🥥 coconut', 2: '🌽 maize', 3: '🫘 chickpea', 4: '🫘 kidneybeans',
    5: '🫘 pigeonpeas', 6: '🫘 mothbeans', 7: '🫘 mungbean', 8: '🫘 blackgram', 9: '🫘 lentil',
    10: '🍎 pomegranate', 11: '🍌 banana', 12: '🥭 mango', 13: '🍇 grapes', 14: '🍉 watermelon',
    15: '🍈 muskmelon', 16: '🍎 apple', 17: '🍊 orange', 18: '🍍 papaya', 19: '🧥 cotton',
    20: '🧥 jute', 21: '☕ coffee'
}

# Validate inputs
if any(pd.isna(xdata)):
    st.error("❌ All inputs are required. Please provide valid values.")
    st.stop()

# Convert input data to DataFrame
try:
    x = pd.DataFrame([xdata], columns=df.columns[:7])
except Exception as e:
    st.error(f"❌ Error preparing input data: {e}")
    st.stop()

# Show input data
st.write("💾 **Given Input:**")
st.dataframe(x)

# Predict and display result
if st.button("🔍 Predict"):
    try:
        prediction = model.predict(x)
        crop_name = crop_mapping.get(prediction[0], '❓ Unknown label')
        st.write(f"🌟 **Recommended Crop:** {crop_name}")
    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
