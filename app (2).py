import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title='Drilling Optimization Full App', layout='wide')
st.title('AI-Powered Drilling Optimization — Full App')

uploaded = st.file_uploader('Upload drilling CSV', type=['csv'])
if uploaded is None:
    st.info('Using sample dataset shipped with app. You can upload your own CSV with columns: timestamp,WOB_klbf,RPM,Torque_klbf_ft,ROP_ftph,Flow_gpm,MW_ppg,Viscosity_cP')
    df = pd.read_csv('sample_drilling_full.csv', parse_dates=['timestamp'])
else:
    df = pd.read_csv(uploaded, parse_dates=['timestamp'])

# Model functions
import math

def compute_mse_row(WOB, Torque, RPM, ROP, bit_diam_in=8.5):
    WOB_lbf = WOB * 1000.0
    Torque_lbf_ft = Torque * 1000.0
    v_ft_per_min = (ROP / 60.0) if ROP>0 else 1e-6
    A_in2 = math.pi * (bit_diam_in**2) / 4.0
    C = 120.0
    term1 = WOB_lbf / A_in2
    term2 = (C * Torque_lbf_ft * RPM) / (A_in2 * v_ft_per_min)
    return term1 + term2

# simplified models
def burgoyne_young_rop(WOB, RPM, formation_factor=1.0, bit_wear=1.0, a=0.6, b=0.4, k=0.1):
    return k * (WOB**a) * (RPM**b) * formation_factor * bit_wear

def warren_rop(WOB, RPM, K=0.05, alpha=0.5, beta=0.7):
    return K * (WOB**alpha) * (RPM**beta)

def modified_warren_rop(WOB, RPM, K=0.05, alpha=0.5, beta=0.7, wear_factor=1.0, hydraulic_factor=1.0):
    return warren_rop(WOB, RPM, K, alpha, beta) * wear_factor * hydraulic_factor

def hareland_rampersad_rop(WOB, RPM, a=0.45, b=0.6, K=0.02):
    return K * (WOB**a) * (RPM**b)

def maurer_hydraulic_factor(flow_gpm, annulus_area=100.0, k=1e-3):
    return 1.0 + k * flow_gpm / (math.sqrt(annulus_area)+1e-6)

# Compute metrics for dataframe
if 'MSE' not in df.columns:
    df['MSE'] = df.apply(lambda r: compute_mse_row(r['WOB_klbf'], r['Torque_klbf_ft'], r['RPM'], r['ROP_ftph']), axis=1)

# compute bit wear index (rolling)
df['ft_per_min'] = df['ROP_ftph'] / 60.0
df['ft_per_min'] = df['ft_per_min'].replace(0,1e-6)
df['mse_times_df'] = df['MSE'] * df['ft_per_min']
df['cum_mse'] = df['mse_times_df'].cumsum()
df['cum_ft'] = df['ft_per_min'].cumsum()
df['Bit_Wear_Index'] = df['cum_mse'] / (df['cum_ft']+1e-6) * 1000.0

st.subheader('Data Preview')
st.dataframe(df.head())

# Show computed plots
st.subheader('Computed Metrics Over Time')
col1, col2 = st.columns(2)
with col1:
    st.line_chart(df.set_index('timestamp')['MSE'])
with col2:
    st.line_chart(df.set_index('timestamp')['Bit_Wear_Index'])

st.subheader('Interactive Input for Single-Point Prediction')
with st.form('input_form'):
    WOB = st.number_input('WOB (klbf)', value=float(df['WOB_klbf'].median()))
    RPM = st.number_input('RPM', value=float(df['RPM'].median()))
    Torque = st.number_input('Torque (klbf-ft)', value=float(df['Torque_klbf_ft'].median()))
    ROP = st.number_input('ROP (ftph)', value=float(df['ROP_ftph'].median()))
    Flow = st.number_input('Flow (gpm)', value=float(df['Flow_gpm'].median()))
    MW = st.number_input('Mud Weight (ppg)', value=float(df['MW_ppg'].median()))
    Visc = st.number_input('Viscosity (cP)', value=float(df['Viscosity_cP'].median()))
    submit = st.form_submit_button('Compute')

if submit:
    mse_val = compute_mse_row(WOB, Torque, RPM, ROP)
    bwi_val = ( (df['mse_times_df'].iloc[-59:].sum() + mse_val*(ROP/60.0)) / (df['ft_per_min'].iloc[-59:].sum() + ROP/60.0 + 1e-6) ) * 1000.0
    by = burgoyne_young_rop(WOB, RPM)
    warr = warren_rop(WOB, RPM)
    mod_w = modified_warren_rop(WOB, RPM, wear_factor=1.0, hydraulic_factor=maurer_hydraulic_factor(Flow))
    hr = hareland_rampersad_rop(WOB, RPM)
    st.write('### Results Summary')
    st.write({'MSE':mse_val, 'Bit_Wear_Index':bwi_val, 'Bourgoyne_Young_ROP_ftph':by, 'Warren_ROP_ftph':warr, 'Modified_Warren_ROP_ftph':mod_w, 'Hareland_ROP_ftph':hr})
    # show a comparative bar chart of ROP predictions
    preds = pd.DataFrame({'Model':['Bourgoyne_Young','Warren','Modified_Warren','Hareland'],'ROP_ftph':[by,warr,mod_w,hr]})
    st.bar_chart(preds.set_index('Model'))

# Simple RF model trained on df for quick classification
st.subheader('Quick ML Classifier Demo')
from sklearn.ensemble import RandomForestClassifier
features = ['WOB_klbf','RPM','Torque_klbf_ft','ROP_ftph','MSE','Bit_Wear_Index']
X = df[features].fillna(method='ffill').fillna(0)
df['Problem'] = ((df['Bit_Wear_Index'] > df['Bit_Wear_Index'].quantile(0.97)) | (df['ROP_ftph'] < df['ROP_ftph'].quantile(0.03))).astype(int)
y = df['Problem']
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)
df['Pred'] = model.predict(X)
st.write('Model trained on demo synthetic data. Use field-calibrated data for production.')
st.write('Sample predictions:')
st.dataframe(df[['timestamp','WOB_klbf','RPM','ROP_ftph','MSE','Bit_Wear_Index','Pred']].tail(10))

# Download computed CSV
st.download_button('Download computed CSV', df.to_csv(index=False), file_name='drilling_full_computed.csv', mime='text/csv')
