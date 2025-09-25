
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Drilling Optimization App", layout="wide")
st.title("AI-powered Drilling Problem Detection & WOB Optimization")

st.markdown("Upload a CSV or use the sample dataset. The app fits the Bourgoyne & Young model coefficients using a simple genetic algorithm, predicts ROP, and searches for optimal WOB.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    df = pd.read_csv("sample_drilling_data.csv")
    st.info("Using sample dataset extracted from Salem et al. (2021).")
else:
    df = pd.read_csv(uploaded)

st.dataframe(df.head())

# Feature engineering: compute simple BY model parameters per the paper (we use simplified forms)
def prepare_features(df):
    # For clarity we use the input columns directly and add normalized W (W_over_d) and log transforms
    X = df.copy()
    X["ln_W_over_d"] = np.log(X["W_over_d_1000lb_per_in"].replace(0.0001, 0.0001))
    X["ln_RPM"] = np.log(X["RPM"].replace(0.1, 0.1))
    return X

X = prepare_features(df)

st.subheader("Model fitting (Genetic Algorithm)")

# Bourgoyne & Young model: ROP = a1*f1 + a2*f2 + ... + a8*f8 (we implement simple terms)
def by_predict(params, row):
    # params: array-like of 8 coefficients
    a1,a2,a3,a4,a5,a6,a7,a8 = params
    # approximate the terms used in the paper (simplified)
    f1 = 1.0  # formation strength (treated as constant baseline scaled by a1)
    f2 = (10000 - row["Depth_ft"])/10000.0
    f3 = max(0.0, 0.69*(row["Pore_gradient_ppg"] - 9.0))
    f4 = (row["Depth_ft"]*(row["ECD_ppg"] - row["Pore_gradient_ppg"])) / (row["Depth_ft"]+1.0)
    f5 = row["W_over_d_1000lb_per_in"]
    f6 = row["RPM"]
    f7 = -row["H_tooth_dull"]  # negative effect for wear
    f8 = row["Impact_force_lbf"]
    # Combine (some terms use logs in the original; we use a linear combination)
    pred = (a1*f1 + a2*f2 + a3*f3 + a4*f4 + a5*f5 + a6*f6 + a7*f7 + a8*f8)
    return pred

def mse_for_params(params, df):
    preds = df.apply(lambda r: by_predict(params, r), axis=1)
    return mean_squared_error(df["ROP_ft_hr"], preds)

# Simple GA implementation
import random
def simple_ga(df, pop_size=80, gens=200, mut_rate=0.2):
    # initialize population around small random values
    pop = [np.random.normal(loc=0.0, scale=1.0, size=8) for _ in range(pop_size)]
    scores = [mse_for_params(p, df) for p in pop]
    for g in range(gens):
        # selection (tournament)
        new_pop = []
        for _ in range(pop_size//2):
            i,j = random.sample(range(pop_size), 2)
            parent = pop[i] if scores[i]<scores[j] else pop[j]
            k,l = random.sample(range(pop_size), 2)
            parent2 = pop[k] if scores[k]<scores[l] else pop[l]
            # crossover
            cut = random.randint(1,7)
            child1 = np.concatenate([parent[:cut], parent2[cut:]])
            child2 = np.concatenate([parent2[:cut], parent[cut:]])
            # mutation
            for ch in (child1, child2):
                if random.random() < mut_rate:
                    idx = random.randint(0,7)
                    ch[idx] += np.random.normal(0, 0.5)
                new_pop.append(ch)
        pop = new_pop
        scores = [mse_for_params(p, df) for p in pop]
    best_idx = int(np.argmin(scores))
    return pop[best_idx], scores[best_idx]

if st.button("Fit model with GA (this may take ~20s)"):
    with st.spinner("Running simple GA..."):
        best_params, best_score = simple_ga(df)
    st.success(f"GA completed. Best MSE: {best_score:.3f}")
    st.write("Best parameters (a1..a8):")
    st.write(best_params.tolist())
    df["pred_ROP"] = df.apply(lambda r: by_predict(best_params, r), axis=1)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.scatter(df["ROP_ft_hr"], df["pred_ROP"])
    ax.plot([df["ROP_ft_hr"].min(), df["ROP_ft_hr"].max()],
            [df["ROP_ft_hr"].min(), df["ROP_ft_hr"].max()], linestyle="--")
    ax.set_xlabel("Actual ROP (ft/hr)"); ax.set_ylabel("Predicted ROP (ft/hr)")
    st.pyplot(fig)
    st.subheader("Data with Predictions")
    st.dataframe(df[["Depth_ft","ROP_ft_hr","pred_ROP","W_over_d_1000lb_per_in","RPM"]])

    # WOB optimization: for a selected depth, optimize W over a grid to maximize predicted ROP
    st.subheader("Optimize WOB for a depth")
    depth_sel = st.number_input("Depth (ft) for optimization", value=float(df["Depth_ft"].median()))
    rpm_sel = st.number_input("RPM", value=float(df["RPM"].median()))
    pore_sel = st.number_input("Pore gradient (ppg)", value=float(df["Pore_gradient_ppg"].median()))
    ecd_sel = st.number_input("ECD (ppg)", value=float(df["ECD_ppg"].median()))
    impact_sel = st.number_input("Jet impact force (lbf)", value=float(df["Impact_force_lbf"].median()))
    tooth_sel = st.number_input("Tooth dullness (fraction)", value=0.125, step=0.001)
    # search grid of W_over_d (1000 lb/in)
    W_grid = np.linspace(0.1, 30.0, 300)
    preds = []
    for w in W_grid:
        row = {"Depth_ft": depth_sel, "RPM": rpm_sel, "Pore_gradient_ppg": pore_sel, "ECD_ppg": ecd_sel,
               "W_over_d_1000lb_per_in": w, "H_tooth_dull": tooth_sel, "Impact_force_lbf": impact_sel}
        preds.append(by_predict(best_params, pd.Series(row)))
    best_idx = np.argmax(preds)
    st.write(f"Estimated optimal W/d (1000 lb/in): {W_grid[best_idx]:.3f} --> Predicted ROP: {preds[best_idx]:.2f} ft/hr")
    fig2, ax2 = plt.subplots(figsize=(8,3))
    ax2.plot(W_grid, preds)
    ax2.set_xlabel("W/d (1000 lb/in)"); ax2.set_ylabel("Predicted ROP (ft/hr)")
    st.pyplot(fig2)
else:
    st.info("Press the button to fit model and run optimization.")

st.markdown("---")
st.caption("Model and sample data adapted from: Salem et al. (2021), 'Estimation of Bourgoyne and Young Model Coefficients to Predict Optimum Drilling Rates and Bit Weights using Genetic Algorithms' (IOP Conf. Series)")
