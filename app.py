import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Demand Forecasting App", page_icon="📈", layout="centered")

@st.cache_resource
def load_artifacts():
    with open("xgboost_demand_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, label_encoders = load_artifacts()

banner = """<svg viewBox="0 0 800 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;border-radius:14px;">
<defs>
<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
<stop offset="0%" stop-color="#0f2027"/>
<stop offset="50%" stop-color="#203a43"/>
<stop offset="100%" stop-color="#2c5364"/>
</linearGradient>
</defs>
<rect width="800" height="220" fill="url(#bg)"/>
<g fill="#4fc3f7" opacity="0.85">
<rect x="430" y="140" width="40" height="50" rx="4"/>
<rect x="490" y="115" width="40" height="75" rx="4"/>
<rect x="550" y="90" width="40" height="100" rx="4"/>
<rect x="610" y="105" width="40" height="85" rx="4"/>
<rect x="670" y="60" width="40" height="130" rx="4"/>
<rect x="730" y="35" width="40" height="155" rx="4"/>
</g>
<polyline points="450,125 510,100 570,75 630,90 690,45 750,22" fill="none" stroke="#ffd54f" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
<g fill="#ffd54f">
<circle cx="450" cy="125" r="5"/>
<circle cx="510" cy="100" r="5"/>
<circle cx="570" cy="75" r="5"/>
<circle cx="630" cy="90" r="5"/>
<circle cx="690" cy="45" r="5"/>
<circle cx="750" cy="22" r="5"/>
</g>
<line x1="410" y1="190" x2="780" y2="190" stroke="#ffffff" stroke-opacity="0.4" stroke-width="2"/>
<text x="40" y="95" fill="#ffffff" font-family="Arial, sans-serif" font-size="34" font-weight="bold">Demand Forecasting</text>
<text x="40" y="132" fill="#b3e5fc" font-family="Arial, sans-serif" font-size="18">Predict product demand with XGBoost</text>
<text x="40" y="165" fill="#ffffff" fill-opacity="0.7" font-family="Arial, sans-serif" font-size="14">Price | Discount | Inventory | Promotion | Competitor</text>
</svg>"""

st.markdown(banner, unsafe_allow_html=True)
st.title("Demand Forecasting App")
st.divider()
st.header("Input Features")
price = st.number_input("Price", min_value=0.0, value=50.0)
discount = st.number_input("Discount (%)", min_value=0, max_value=100, value=10)
inventory_level = st.number_input("Inventory Level", min_value=0, value=100)
promotion = st.selectbox("Promotion", [0, 1])
competitor_pricing = st.number_input("Competitor Price", min_value=0.0, value=50.0)
category = st.selectbox(
    "Category",
    label_encoders["Category"].classes_.tolist()
)

input_data = pd.DataFrame({
    "Price": [price],
    "Discount": [discount],
    "Inventory Level": [inventory_level],
    "Promotion": [promotion],
    "Competitor Pricing": [competitor_pricing],
    "Category": [category]
})

for col, encoder in label_encoders.items():
    if col in input_data.columns:
        input_data[col] = encoder.transform(input_data[col])

st.divider()
if st.button("Predicted Demand"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Demand: {int(prediction)}")