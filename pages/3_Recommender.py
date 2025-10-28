# pages/3_📱_Recommender.py
"""
Recommender Page - Content-Based App Recommendation System
FAISS Uses Facebook AI Similarity Search for fast, accurate app recommendations
"""

import streamlit as st
import sys
from pathlib import Path
from utils.model_loader import load_models
import pandas as pd
import numpy as np
import joblib
import faiss
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from scipy.sparse import hstack

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

from utils.styling import load_css, create_header, create_section_header
from utils.config import APP_TITLE, APP_ICON, PAGE_LAYOUT, CLASSIFIER_FEATURES, MODEL_METRICS, COLORS
from utils.helpers import initialize_session_state, format_number, convert_days_to_readable, calculate_rating_density

# ============================================================================
# LOAD MODELS AND DATA FROM GOOGLE DRIVE
# ============================================================================
regressor, classifier, scaler, encoder, faiss_index, feature_matrix, gs_df = load_models()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_input_features(form_data, cat_encoder, scaler):
    """
    Prepare input features for similarity search
    """
    # Extract features
    category = form_data['Category']
    rating_quality = form_data['Rating_Quality_Score']
    rating_count = form_data['Rating_Count']
    app_age = form_data['App_Age_Days']
    size_mb = form_data['Size_MB']
    
    # Calculate rating density
    rating_density = rating_count / (app_age + 1)
    
    # Encode category
    cat_features = cat_encoder.transform([[category]])
    
    # Scale numerical features
    num_features = np.array([[
        rating_quality,
        rating_count,
        app_age,
        size_mb,
        rating_density
    ]])
    num_features_scaled = scaler.transform(num_features)
    
    # Combine features
    X = hstack([cat_features, num_features_scaled])
    X_dense = X.toarray().astype('float32')
    
    # Normalize for cosine similarity
    faiss.normalize_L2(X_dense)
    
    return X_dense


def recommend_apps_faiss(form_data, faiss_index, encoder, scaler, n=5, min_rating=3.5):
    """
    Recommend similar apps using FAISS similarity search
    """
    try:
        # Prepare input vector
        input_vector = prepare_input_features(form_data, encoder, scaler)
        
        # Search for similar apps (get more than needed to filter)
        D, I = faiss_index.search(input_vector, n * 3)
        
        # Get recommended apps
        similar_indices = I[0]
        
        # Filter by minimum rating and return top n
        recommendations = gs_df.iloc[similar_indices].copy()
        recommendations = recommendations[recommendations['Rating'] >= min_rating]
        recommendations = recommendations.head(n)
        
        # Add similarity scores
        recommendations['Similarity'] = D[0][:len(recommendations)]
        
        return recommendations[['App_Name', 'Category', 'Rating', 'Rating_Count', 
                               'Installs_Central', 'Size_MB', 'Free', 'Ad_Supported', 
                               'In_App_Purchases', 'Editors_Choice', 'Similarity']]
        
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return pd.DataFrame()


def search_similar_apps(app_name, faiss_index, feature_matrix, n=5):
    """
    Find apps similar to a specific app by name
    """
    # Find matching apps (case-insensitive, partial match)
    matches = gs_df[gs_df['App_Name'].str.contains(app_name, case=False, na=False)]
    
    if matches.empty:
        return None, f"No apps found matching '{app_name}'. Try another search term."
    
    # Get the first match
    idx = matches.index[0]
    matched_app = gs_df.iloc[idx]
    
    try:
        # Load feature vector from .npz
        X_dense = feature_matrix['arr_0'].astype('float32')
        
        if X_dense is None:
            return None, "Feature matrix not available for app-based search."
        
        # Get app vector and normalize
        app_vector = X_dense[idx].reshape(1, -1)
        faiss.normalize_L2(app_vector)
        
        # Search for similar apps
        D, I = faiss_index.search(app_vector, n + 1)
        
        # Exclude the app itself (first result)
        similar_indices = I[0][1:]
        
        # Get recommendations
        recommendations = gs_df.iloc[similar_indices].copy()
        recommendations['Similarity'] = D[0][1:len(recommendations)+1]
        
        return recommendations[['App_Name', 'Category', 'Rating', 'Rating_Count', 
                               'Installs_Central', 'Size_MB', 'Similarity']], None
        
    except Exception as e:
        return None, f"Error finding similar apps: {e}"


# ============================================================================
# PAGE SETUP
# ============================================================================

st.set_page_config(
    page_title=f"Recommender | {APP_TITLE}",
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded"
)

initialize_session_state()
load_css()

# Check if models loaded
if faiss_index is None or gs_df is None:
    st.error("⚠️ Could not load recommender components. Please check model files.")
    st.stop()

# ============================================================================
# MAIN UI
# ============================================================================

create_header("Smart App Recommender", "Discover similar successful apps using AI-powered content-based filtering")

# Sidebar metrics
with st.sidebar:
    st.markdown("### Recommender Info")
    rec_metrics = MODEL_METRICS.get("recommender", {})
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Precision@10", f"{rec_metrics.get('precision_at_10', 0.85):.1%}")
    with col2:
        st.metric("Recall@10", f"{rec_metrics.get('recall_at_10', 0.72):.1%}")
    st.metric("NDCG Score", f"{rec_metrics.get('ndcg', 0.89):.3f}", 
              help="Normalized Discounted Cumulative Gain")
    
    st.markdown("---")
    st.markdown("### How It Works")
    st.markdown("""
    **FAISS-Based Similarity Search**
    1. Describe Your App - Enter key features
    2. AI Analysis - FAISS finds similar apps
    3. Smart Ranking - Apps ranked by similarity
    4. Learn Strategies - Discover success patterns
    
    **Technology:**
    - Facebook AI Similarity Search (FAISS)
    - Content-based filtering
    - Cosine similarity matching
    """)
    
    if st.button("Back to Home", use_container_width=True):
        st.switch_page("app.py")

# Choose recommendation mode
st.markdown("### Choose Recommendation Mode")
mode = st.radio(
    "",
    options=["Feature-Based Recommendations", "Find Apps Similar to..."],
    help="Choose how you want to discover new apps"
)

st.markdown("---")

# ============================================================================
# MODE 1: SEARCH BY APP NAME
# ============================================================================

if mode == "Find Apps Similar to...":
    st.markdown("### Search for an App")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Enter app name",
            placeholder="e.g., WhatsApp, Instagram, Candy Crush...",
            help="Type any part of the app name"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        n_results = st.number_input("Results", min_value=3, max_value=20, value=8, step=1)
    
    if st.button("Find Similar Apps", use_container_width=True, type="primary"):
        if not search_query:
            st.warning("Please enter an app name to search.")
        else:
            with st.spinner(f"Searching for apps similar to {search_query}..."):
                recommendations, error = search_similar_apps(
                    search_query,
                    faiss_index,
                    feature_matrix,
                    n=n_results
                )
                
                if error:
                    st.error(error)
                elif recommendations is not None and not recommendations.empty:
                    st.session_state.recommendations = recommendations
                    st.session_state.search_query = search_query
                    st.session_state.recommendations_made = True
                else:
                    st.warning("No recommendations found.")

# ============================================================================
# MODE 2: FEATURE-BASED RECOMMENDATIONS
# ============================================================================

else:
    st.markdown("### Describe Your App")
    
    with st.container():
        col1, col2 = st.columns(2, gap="large")
        form_data = {}
        
        with col1:
            create_section_header("App Performance", "📊")
            
            # Rating Quality
            config = CLASSIFIER_FEATURES.get("Rating_Quality_Score", {})
            rating = st.slider(
                f"{config.get('icon', '')} {config.get('display_name', 'Rating Quality')}",
                min_value=config.get("min", 0),
                max_value=config.get("max", 5),
                value=config.get("default", 3),
                step=config.get("step", 0.5),
                help=config.get("help_text", ""),
                key="rec_rating"
            )
            form_data['Rating_Quality_Score'] = rating
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Rating Count
            config = CLASSIFIER_FEATURES.get("Rating_Count", {})
            rating_count = st.number_input(
                f"{config.get('icon', '')} {config.get('display_name', 'Rating Count')}",
                min_value=config.get("min", 0),
                max_value=config.get("max", 10000000),
                value=config.get("default", 10000),
                step=config.get("step", 1000),
                help=config.get("help_text", ""),
                key="rec_rating_count"
            )
            form_data['Rating_Count'] = rating_count
            st.markdown("<br>", unsafe_allow_html=True)
            
            # App Age
            config = CLASSIFIER_FEATURES.get("App_Age_Days", {})
            app_age = st.number_input(
                f"{config.get('icon', '')} {config.get('display_name', 'App Age (Days)')}",
                min_value=config.get("min", 0),
                max_value=config.get("max", 5000),
                value=config.get("default", 365),
                step=config.get("step", 30),
                help=config.get("help_text", ""),
                key="rec_app_age"
            )
            form_data['App_Age_Days'] = app_age
            readable_age = convert_days_to_readable(app_age)
            st.caption(f"Age: {readable_age}")
            rating_density = calculate_rating_density(rating_count, app_age)
            st.caption(f"Rating Density: {rating_density:.2f} ratings/day")
        
        with col2:
            create_section_header("App Details", "📱")
            
            # Size
            config = CLASSIFIER_FEATURES.get("Size_MB", {})
            size = st.slider(
                f"{config.get('icon', '')} {config.get('display_name', 'Size (MB)')}",
                min_value=config.get("min", 0.0),
                max_value=config.get("max", 500.0),
                value=config.get("default", 50.0),
                step=config.get("step", 5.0),
                help=config.get("help_text", ""),
                key="rec_size"
            )
            form_data['Size_MB'] = size
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Category
            config = CLASSIFIER_FEATURES.get("Category", {})
            category = st.selectbox(
                f"{config.get('icon', '')} {config.get('display_name', 'Category')}",
                options=config.get("options", []),
                index=config.get("options", []).index(config.get("default", config.get("options", [""])[0])) if config.get("options") else 0,
                help=config.get("help_text", ""),
                key="rec_category"
            )
            form_data['Category'] = category
    
    st.markdown("---")
    
    # Recommendation Settings
    st.markdown("### Recommendation Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        n_recommendations = st.slider(
            "Number of Recommendations",
            min_value=3,
            max_value=15,
            value=8,
            step=1,
            help="How many similar apps to recommend"
        )
    
    with col2:
        min_rating = st.slider(
            "Minimum Rating Filter",
            min_value=1.0,
            max_value=5.0,
            value=3.5,
            step=0.5,
            help="Only show apps with this rating or higher"
        )
    
    # Generate Recommendations Button
    if st.button("Find Similar Apps", use_container_width=True, type="primary"):
        with st.spinner("Analyzing and finding similar apps..."):
            recommendations = recommend_apps_faiss(
                form_data,
                faiss_index,
                encoder,
                scaler,
                n=n_recommendations,
                min_rating=min_rating
            )
            
            if not recommendations.empty:
                st.session_state.recommendations = recommendations
                st.session_state.form_data = form_data
                st.session_state.recommendations_made = True
            else:
                st.warning("No apps found matching your criteria. Try adjusting the filters.")

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

st.markdown("---")

if st.session_state.get("recommendations_made", False):
    recommendations = st.session_state.get("recommendations")
    
    if recommendations is not None and not recommendations.empty:
        st.markdown("### Recommended Apps")
        
        if "search_query" in st.session_state:
            st.markdown(f"**Apps similar to:** {st.session_state.search_query}")
        
        st.markdown(f"**Found {len(recommendations)} similar apps**")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Reset button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("New Search", use_container_width=True):
                st.session_state.recommendations_made = False
                st.session_state.recommendations = None
                if "search_query" in st.session_state:
                    del st.session_state.search_query
                if "form_data" in st.session_state:
                    del st.session_state.form_data
                st.rerun()
        
        # Display results in tabs
        tab1, tab2, tab3 = st.tabs(["📋 Detailed List", "📊 Comparison Table", "💡 Key Insights"])
        
        # Format dataframe
        display_df = recommendations.copy()
        display_df['Rating_Count'] = display_df['Rating_Count'].apply(format_number)
        display_df['Installs_Central'] = display_df['Installs_Central'].apply(format_number)
        display_df['Size_MB'] = display_df['Size_MB'].apply(lambda x: f"{x:.0f}MB")
        display_df['Similarity'] = display_df['Similarity'].apply(lambda x: f"{x*100:.1f}%")
        
        with tab1:
            st.markdown("#### Top Recommendation")
            if len(recommendations) > 0:
                top_app = recommendations.iloc[0]
                top_similarity = top_app.get('Similarity', 0) * 100
                
                # Color based on similarity
                if top_similarity >= 90:
                    color = COLORS.get("success", "#10b981")
                elif top_similarity >= 75:
                    color = COLORS.get("accent", "#8b5cf6")
                elif top_similarity >= 60:
                    color = COLORS.get("warning", "#f59e0b")
                else:
                    color = COLORS.get("secondary", "#6b7280")
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color}, {COLORS.get('primary', '#3b82f6')}); 
                            padding: 2rem; border-radius: 12px; color: white; margin-bottom: 1rem;">
                    <h3 style="margin: 0 0 1rem 0;">{top_app['App_Name']}</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
                        <div>
                            <div style="font-size: 0.9rem; opacity: 0.8;">Similarity</div>
                            <div style="font-size: 2rem; font-weight: bold;">{top_similarity:.0f}%</div>
                        </div>
                        <div>
                            <div style="font-size: 0.9rem; opacity: 0.8;">Rating</div>
                            <div style="font-size: 2rem; font-weight: bold;">{top_app['Rating']:.1f}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.9rem; opacity: 0.8;">Installs</div>
                            <div style="font-size: 2rem; font-weight: bold;">{format_number(top_app['Installs_Central'])}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("#### All Recommendations")
            display_columns = ['App_Name', 'Category', 'Rating', 'Rating_Count', 'Installs_Central', 'Size_MB', 'Similarity']
            st.dataframe(
                display_df[display_columns],
                use_container_width=True,
                hide_index=True
            )
        
        with tab2:
            st.markdown("#### Quick Comparison")
            st.dataframe(
                display_df[['App_Name', 'Category', 'Rating', 'Similarity']],
                use_container_width=True,
                hide_index=True
            )
        
        with tab3:
            st.markdown("#### Key Success Patterns")
            
            top3 = recommendations.head(3)
            avg_rating = top3['Rating'].mean()
            avg_reviews = top3['Rating_Count'].mean()
            avg_installs = top3['Installs_Central'].mean()
            avg_size = top3['Size_MB'].mean()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Rating", f"{avg_rating:.2f}")
            with col2:
                st.metric("Avg Reviews", format_number(avg_reviews))
            with col3:
                st.metric("Avg Installs", format_number(avg_installs))
            with col4:
                st.metric("Avg Size", f"{avg_size:.0f}MB")
            
            st.markdown("---")
            
            # Monetization insights
            st.markdown("#### Monetization Strategies")
            if all(col in recommendations.columns for col in ['Free', 'Ad_Supported', 'In_App_Purchases']):
                free_pct = recommendations['Free'].sum() / len(recommendations) * 100
                ads_pct = recommendations['Ad_Supported'].sum() / len(recommendations) * 100
                iapp_pct = recommendations['In_App_Purchases'].sum() / len(recommendations) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Free Apps", f"{free_pct:.0f}%")
                with col2:
                    st.metric("Ad-Supported", f"{ads_pct:.0f}%")
                with col3:
                    st.metric("In-App Purchases", f"{iapp_pct:.0f}%")
            
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; color: #64748B; font-size: 0.9rem;">
                <p><strong>Powered by Facebook AI Similarity Search (FAISS)</strong></p>
                <p>Content-based recommendations using cosine similarity and normalized features</p>
                <p>Results based on app features: category, ratings, engagement, and performance metrics</p>
            </div>
            """, unsafe_allow_html=True)
