import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("sample_drilling_data.csv")

X = df[['WOB', 'RPM', 'Torque']]
y = df['Problem']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

st.title("🛢️ AI-Powered Drilling Problem Detection")
st.write("Upload drilling data and get predictions.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview", data.head())
    
    # Make predictions
    preds = model.predict(data[['WOB','RPM','Torque']])
    data['Prediction'] = preds
    
    st.write("### Predictions", data.head())
    
    # Plot WOB vs ROP
    fig, ax = plt.subplots()
    scatter = ax.scatter(data['WOB'], data['ROP'], c=data['Prediction'], cmap='coolwarm')
    ax.set_xlabel("WOB")
    ax.set_ylabel("ROP")
    ax.set_title("WOB vs ROP (Predictions)")
    st.pyplot(fig)
