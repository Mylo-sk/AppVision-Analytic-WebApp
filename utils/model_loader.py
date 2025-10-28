import streamlit as st
import sys
from pathlib import Path
import os
import urllib.request
import numpy as np
import pandas as pd
import joblib
import faiss

#Importing models
def maybe_download_from_gdrive(file_id, file_path):
    if not os.path.exists(file_path):
        print(f"Downloading {file_path} from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        urllib.request.urlretrieve(url, file_path)

# Repeat for each model (replace the IDs)
maybe_download_from_gdrive("1UUYrKXjWKM4aUYSeB52CXLtVerJGf-2b", "models/gstore_rfr_model.pkl")
maybe_download_from_gdrive("1SoSZtcgKwrS7mVIhLJ7pTClNp6Q51XiG", "models/xgb_model.pkl")
maybe_download_from_gdrive("16gcOixrPgBOT8jQXGg4ewvyjQ_4ohrge", "models/faiss_recommender.idx")
maybe_download_from_gdrive("1HbfVwrOvvymDAuKOaw22PXq2K8nFmgLs", "models/scaler.pkl")
maybe_download_from_gdrive("160jtBbD3GxNysem6QXc_IVPcvSiJ4GVM", "models/cat_encoder.pkl")
maybe_download_from_gdrive("1qnJD-H_6SN2HNDOuwwLRM_DdsRPIyOT0", "models/feature_matrix.npz")
maybe_download_from_gdrive("1sc6Z1D_vjadV4-RAniEFVYQwHdTQ9mWh", "data/gs_df_clean.csv")  # Add this

# Load models (cached)
@st.cache_resource
def load_models():
    regressor = joblib.load("models/gstore_rfr_model.pkl")
    classifier = joblib.load("models/xgb_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoder = joblib.load("models/cat_encoder.pkl")
    faiss_index = faiss.read_index("models/faiss_recommender.idx")
    feature_matrix = np.load("models/feature_matrix.npz")
    gs_df = pd.read_csv("data/gs_df_clean.csv")

    return regressor, classifier, scaler, encoder, faiss_index, feature_matrix, gs_df

regressor, classifier, scaler, encoder, faiss_index, feature_matrix, gs_df = load_models()
