import streamlit as st
import sys
from pathlib import Path
import os
import urllib.request
import numpy as np
import pandas as pd
import joblib



def maybe_download_from_gdrive(file_id, file_path):
    """Download file from Google Drive if it doesn't exist"""
    
    # Create directory if it doesn't exist
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    # Download if file doesn't exist
    if not os.path.exists(file_path):
        try:
            print(f"⏳ Downloading {file_path}...")
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            urllib.request.urlretrieve(url, file_path)
            print(f"✅ Downloaded {file_path}")
            return True
        except Exception as e:
            print(f"❌ Error downloading {file_path}: {e}")
            print(f"   File ID: {file_id}")
            return False
    else:
        print(f"✓ Using cached: {file_path}")
        return True


@st.cache_resource
def load_models():
    """Load all models and data from Google Drive - cached"""
    
    print("\n" + "="*60)
    print("LOADING MODELS FROM GOOGLE DRIVE")
    print("="*60)
    
    # Download all files
    all_downloaded = True
    
    all_downloaded &= maybe_download_from_gdrive(
        "1UUYrKXjWKM4aUYSeB52CXLtVerJGf-2b", 
        "models/gstore_rfr_model.pkl"
    )
    
    all_downloaded &= maybe_download_from_gdrive(
        "1SoSZtcgKwrS7mVIhLJ7pTClNp6Q51XiG", 
        "models/xgboost_model.pkl"
    )
    
    all_downloaded &= maybe_download_from_gdrive(
        "16gcOixrPgBOT8jQXGg4ewvyjQ_4ohrge", 
        "models/faiss_recommender.idx"
    )
    
    all_downloaded &= maybe_download_from_gdrive(
        "1HbfVwrOvvymDAuKOaw22PXq2K8nFmgLs", 
        "models/scaler.pkl"
    )
    
    all_downloaded &= maybe_download_from_gdrive(
        "160jtBbD3GxNysem6QXc_IVPcvSiJ4GVM", 
        "models/cat_encoder.pkl"
    )
    
    all_downloaded &= maybe_download_from_gdrive(
        "1qnJD-H_6SN2HNDOuwwLRM_DdsRPIyOT0", 
        "models/feature_matrix.npz"
    )
    
    all_downloaded &= maybe_download_from_gdrive(
        "1sc6Z1D_vjadV4-RAniEFVYQwHdTQ9mWh", 
        "data/gs_df_clean.csv"
    )
    
    if not all_downloaded:
        st.error("❌ Some files failed to download!")
        return None, None, None, None, None, None, None
    
    print("\n" + "="*60)
    print("LOADING FILES INTO MEMORY")
    print("="*60)
    
    try:
        regressor = joblib.load("models/gstore_rfr_model.pkl")
        print("✅ Regressor loaded")
        
        classifier = joblib.load("models/xgboost_model.pkl")
        print("✅ Classifier loaded")
        
        scaler = joblib.load("models/scaler.pkl")
        print("✅ Scaler loaded")
        
        encoder = joblib.load("models/cat_encoder.pkl")
        print("✅ Encoder loaded")
        
        faiss_index = faiss.read_index("models/faiss_recommender.idx")
        print("✅ FAISS index loaded")
        
        feature_matrix = np.load("models/feature_matrix.npz")
        print("✅ Feature matrix loaded")
        
        gs_df = pd.read_csv("data/gs_df_clean.csv")
        print(f"✅ Dataset loaded ({len(gs_df)} rows)")
        
        print("="*60)
        print("✅ ALL MODELS LOADED SUCCESSFULLY!")
        print("="*60 + "\n")
        
        return regressor, classifier, scaler, encoder, faiss_index, feature_matrix, gs_df
        
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        st.info("""
        **Troubleshooting:**
        1. Check all files are shared on Google Drive
        2. Verify file IDs are correct
        3. Ensure files haven't been deleted
        4. Check Google Drive storage quota
        """)
        print(f"❌ Error: {e}")
        return None, None, None, None, None, None, None
