import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# ========== Core functions ==========

def detect_variable_type(series):
    unique_vals = series.dropna().unique()
    if series.dtype == 'bool' or set(unique_vals).issubset({0, 1}):
        return "Binary"
    elif series.dtype in ['int64', 'int32'] and (series >= 0).all():
        mean = series.mean()
        var = series.var()
        if var > mean * 1.5:
            return "Count (Overdispersed)"
        else:
            return "Count"
    elif series.dtype in ['float64', 'float32']:
        if ((series >= 0) & (series <= 1)).all():
            return "Proportion"
        elif stats.skew(series.dropna()) > 1:
            return "Continuous (Skewed)"
        else:
            return "Continuous"
    else:
        return "Unknown"

def suggest_likelihood(var_type):
    return {
        "Binary": "Bernoulli",
        "Count": "Poisson",
        "Count (Overdispersed)": "Negative Binomial",
        "Proportion": "Beta",
        "Continuous": "Normal",
        "Continuous (Skewed)": "Log-normal or Gamma",
        "Unknown": "Manual review needed"
    }.get(var_type, "Unknown")

def suggest_prior(var_type):
    return {
        "Binary": "Beta(1,1)",
        "Count": "Gamma(2,2)",
        "Count (Overdispersed)": "Gamma or prior on overdispersion",
        "Proportion": "Beta(2,2)",
        "Continuous": "Normal(μ, σ²)",
        "Continuous (Skewed)": "Log-normal(μ, σ²)",
        "Unknown": "Consult domain expert"
    }.get(var_type, "Unknown")

def show_visuals(data, column):
    st.subheader("📊 Visualizations")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data[column], kde=True, ax=ax[0])
    ax[0].set_title(f'Histogram of {column}')
    sns.boxplot(x=data[column], ax=ax[1])
    ax[1].set_title(f'Boxplot of {column}')
    st.pyplot(fig)

# ========== Streamlit UI ==========

st.set_page_config(page_title="Bayesian Likelihood Suggester", layout="wide")
st.title("🔍 Bayesian Likelihood Suggester")
st.write("Upload a CSV file or use the default dataset to identify variable types and recommended likelihood models for Bayesian inference.")

# Sidebar: File uploader
st.sidebar.header("Step 1: Upload or Use Default Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Use default dataset if none uploaded
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
else:
    iris = load_iris(as_frame=True)
    df = iris.frame
    df['target'] = iris.target
    st.info("📌 No file uploaded. Using default Iris dataset.")

# Column selector
column = st.selectbox("Step 2: Select a column to analyze", df.columns)

# Process selected column
if column:
    data = df[column].dropna()

    st.subheader("📈 Summary Statistics")
    st.write(data.describe())

    var_type = detect_variable_type(data)
    likelihood = suggest_likelihood(var_type)
    prior = suggest_prior(var_type)

    st.markdown(f"### 🔍 Detected Variable Type: **{var_type}**")
    st.markdown(f"### 🧠 Suggested Likelihood Model: **{likelihood}**")
    st.markdown(f"### 📐 Suggested Prior: **{prior}**")

    st.caption("These suggestions are heuristic. Always verify with context and expert insight.")

    show_visuals(df, column)
