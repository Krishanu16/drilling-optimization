import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title='Drilling Optimization Extended', layout='wide')
st.title('AI-Powered Drilling Optimization — Extended Demo')

uploaded = st.file_uploader('Upload drilling CSV', type=['csv'])
if uploaded is None:
    st.info('Using sample dataset shipped with app. You can upload your own CSV with columns: timestamp,WOB_klbf,RPM,Torque_klbf_ft,ROP_ftph,Flow_gpm,MW_ppg')
    df = pd.read_csv('sample_drilling_extended.csv', parse_dates=['timestamp'])
else:
    df = pd.read_csv(uploaded, parse_dates=['timestamp'])

# Compute MSE
def compute_mse(df, wob_col='WOB_klbf', torque_col='Torque_klbf_ft', rpm_col='RPM', rop_col='ROP_ftph', bit_diam_in=8.5):
    df = df.copy()
    WOB_lbf = df[wob_col] * 1000.0
    Torque_lbf_ft = df[torque_col] * 1000.0
    v_ft_per_min = (df[rop_col] / 60.0).replace(0, np.nan)
    D_in = float(bit_diam_in)
    A_in2 = math.pi * (D_in**2) / 4.0
    C = 120.0
    term1 = WOB_lbf / A_in2
    term2 = (C * Torque_lbf_ft * df[rpm_col]) / (A_in2 * v_ft_per_min)
    mse = (term1 + term2).replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    return mse

df['MSE'] = compute_mse(df)

# Bit wear index heuristic
df['ft_per_min'] = df['ROP_ftph'] / 60.0
df['ft_per_min'] = df['ft_per_min'].replace(0, 1e-6)
df['cumulative_ft'] = df['ft_per_min'].cumsum()
df['mse_times_df'] = df['MSE'] * df['ft_per_min']
df['cum_mse_energy'] = df['mse_times_df'].cumsum()
df['Bit_Wear_Index'] = df['cum_mse_energy'] / (df['cumulative_ft'] + 1e-6) * 1000.0

st.subheader('Data Preview')
st.dataframe(df.head())

st.subheader('Computed Metrics')
st.write('MSE (Mechanical Specific Energy) and Bit Wear Index')

col1, col2 = st.columns(2)
with col1:
    st.line_chart(df.set_index('timestamp')['MSE'])
with col2:
    st.line_chart(df.set_index('timestamp')['Bit_Wear_Index'])

# Simple model demo
st.subheader('Simple RandomForest Demo (Bit Wear based label)')
df['Problem'] = (df['Bit_Wear_Index'] > df['Bit_Wear_Index'].quantile(0.95)).astype(int)
features = ['WOB_klbf','RPM','Torque_klbf_ft','ROP_ftph','MSE','Bit_Wear_Index']
X = df[features].fillna(method='ffill').fillna(0)
y = df['Problem']

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)
df['Pred'] = model.predict(X)

st.write('Prediction sample:')
st.dataframe(df[['timestamp','WOB_klbf','RPM','Torque_klbf_ft','ROP_ftph','MSE','Bit_Wear_Index','Pred']].tail(20))

st.download_button('Download computed CSV', df.to_csv(index=False), file_name='drilling_with_mse_bwi.csv', mime='text/csv')

st.info('Placeholder functions for Burgoyne & Young, Warren, Maurer etc. are in the code comments — implement with calibrated coefficients from literature for production use.')