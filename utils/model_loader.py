import streamlit as st
import sys
from pathlib import Path
import os
import urllib.request
import numpy as np
import pandas as pd
import joblib
import faiss


def maybe_download_from_gdrive(file_id, file_path):
    """Download file from Google Drive if it doesn't exist"""
    
    # Create directory if it doesn't exist
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    # Download if file doesn't exist
    if not os.path.exists(file_path):
        try:
            print(f"Downloading {file_path} from Google Drive...")
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            urllib.request.urlretrieve(url, file_path)
            print(f"✅ Downloaded {file_path}")
        except Exception as e:
            print(f"❌ Error downloading {file_path}: {e}")
            raise

# Load models (cached)
@st.cache_resource
def load_models():
    try:
        maybe_download_from_gdrive("1UUYrKXjWKM4aUYSeB52CXLtVerJGf-2b", "models/gstore_rfr_model.pkl")
        maybe_download_from_gdrive("1SoSZtcgKwrS7mVIhLJ7pTClNp6Q51XiG", "models/xgboost_model.pkl")
        maybe_download_from_gdrive("16gcOixrPgBOT8jQXGg4ewvyjQ_4ohrge", "models/faiss_recommender.idx")
        maybe_download_from_gdrive("1HbfVwrOvvymDAuKOaw22PXq2K8nFmgLs", "models/scaler.pkl")
        maybe_download_from_gdrive("160jtBbD3GxNysem6QXc_IVPcvSiJ4GVM", "models/cat_encoder.pkl")
        maybe_download_from_gdrive("1qnJD-H_6SN2HNDOuwwLRM_DdsRPIyOT0", "models/feature_matrix.npz")
        maybe_download_from_gdrive("1sc6Z1D_vjadV4-RAniEFVYQwHdTQ9mWh", "data/gs_df_clean.csv")


        regressor = joblib.load("models/gstore_rfr_model.pkl")
        classifier = joblib.load("models/xgboost_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        encoder = joblib.load("models/cat_encoder.pkl")
        faiss_index = faiss.read_index("models/faiss_recommender.idx")
        feature_matrix = np.load("models/feature_matrix.npz")
        gs_df = pd.read_csv("data/gs_df_clean.csv")

        return regressor, classifier, scaler, encoder, faiss_index, feature_matrix, gs_df
    
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        st.info("Please ensure all model files are accessible from Google Drive")
        return None, None, None, None, None, None, None

regressor, classifier, scaler, encoder, faiss_index, feature_matrix, gs_df = load_models()
