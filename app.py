# === MODERN DIABETES PREDICTION STREAMLIT APP ===

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pickle
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# Page configuration
st.set_page_config(
    page_title="DiabetesAI - Smart Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main styles */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
    }
    
    .section-header {
        font-size: 1.6rem;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        font-weight: 700;
        border-left: 4px solid #667eea;
        padding-left: 1rem;
    }
    
    /* Modern cards */
    .modern-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #e1e8ed;
        margin: 0.8rem 0;
        transition: transform 0.3s ease;
    }
    
    .modern-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        margin: 0.5rem;
    }
    
    .metric-card h3 {
        font-size: 0.9rem;
        margin: 0 0 0.5rem 0;
        opacity: 0.9;
        font-weight: 500;
    }
    
    .metric-card h2 {
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
    }
    
    /* Prediction cards */
    .prediction-card {
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        text-align: center;
        border: none;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        color: white;
    }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border: none;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 10px 10px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
    }
    
    /* Custom slider styling */
    .stSlider {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and resources
@st.cache_resource
def load_model():
    try:
        with open('diabetes_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_features():
    try:
        with open('feature_names.pkl', 'rb') as file:
            features = pickle.load(file)
        return features
    except Exception as e:
        st.error(f"Error loading feature names: {e}")
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('diabetes_cleaned.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def load_metrics():
    try:
        with open('model_metrics.pkl', 'rb') as file:
            metrics = pickle.load(file)
        return metrics
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return None

# Load all resources
model = load_model()
feature_names = load_features()
df = load_data()
metrics = load_metrics()

# Show loading status
if model and feature_names and df is not None:
    st.sidebar.success("🎯 All systems ready!")
else:
    st.error("❌ System initialization failed. Please check resource files.")
    st.stop()

# Modern sidebar navigation
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin-bottom: 2rem;'>
    <h2 style='margin: 0;'>🩺 DiabetesAI</h2>
    <p style='margin: 0; opacity: 0.9;'>Smart Prediction System</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("", [
    "📊 Dashboard", 
    "🔍 Data Explorer", 
    "📈 Analytics", 
    "🎯 Prediction", 
    "🤖 Model Info"
], label_visibility="collapsed")

# Dashboard Page
if page == "📊 Dashboard":
    st.markdown('<h1 class="main-header">🩺 DiabetesAI Dashboard</h1>', unsafe_allow_html=True)
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='modern-card'>
            <h3 style='color: #2c3e50; margin-top: 0;'>Welcome to DiabetesAI</h3>
            <p style='color: #7f8c8d; line-height: 1.6;'>
            Advanced machine learning system for diabetes risk assessment. 
            Our AI analyzes multiple health parameters to provide accurate 
            predictions and valuable insights for proactive health management.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); border-radius: 15px;'>
            <h3 style='color: #2c3e50;'>Quick Assessment</h3>
            <p style='color: #7f8c8d;'>Get instant diabetes risk prediction</p>
            <button style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 0.8rem 1.5rem; border-radius: 25px; font-weight: 600; cursor: pointer;' onclick="window.location.href='#prediction'">Start Assessment</button>
        </div>
        """, unsafe_allow_html=True)
    
    # Key metrics
    st.markdown('<div class="section-header">📈 System Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Patients</h3>
            <h2>{len(df):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Diabetes Cases</h3>
            <h2>{df['Outcome'].sum():,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        diabetes_rate = (df['Outcome'].sum() / len(df)) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>Prevalence Rate</h3>
            <h2>{diabetes_rate:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if metrics:
            st.markdown(f"""
            <div class="metric-card">
                <h3>AI Accuracy</h3>
                <h2>{metrics['accuracy']:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Features and quick actions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">🔬 Health Parameters</div>', unsafe_allow_html=True)
        features_list = [
            {"icon": "🩸", "name": "Glucose Level", "desc": "Blood sugar concentration"},
            {"icon": "💓", "name": "Blood Pressure", "desc": "Diastolic pressure"},
            {"icon": "📏", "name": "BMI", "desc": "Body Mass Index"},
            {"icon": "🕰️", "name": "Age", "desc": "Patient age"},
            {"icon": "🧬", "name": "Genetics", "desc": "Diabetes pedigree"},
            {"icon": "💉", "name": "Insulin", "desc": "Serum insulin level"}
        ]
        
        for feature in features_list:
            st.markdown(f"""
            <div class='feature-card'>
                <div style='display: flex; align-items: center; gap: 1rem;'>
                    <span style='font-size: 1.5rem;'>{feature['icon']}</span>
                    <div>
                        <h4 style='margin: 0; color: #2c3e50;'>{feature['name']}</h4>
                        <p style='margin: 0; color: #7f8c8d; font-size: 0.9rem;'>{feature['desc']}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-header">🚀 Quick Actions</div>', unsafe_allow_html=True)
        
        action_col1, action_col2 = st.columns(2)
        
        with action_col1:
            if st.button("🎯 Risk Assessment", use_container_width=True):
                st.session_state.page = "Prediction"
                st.rerun()
            
            if st.button("📊 View Data", use_container_width=True):
                st.session_state.page = "Data Explorer"
                st.rerun()
        
        with action_col2:
            if st.button("📈 Analytics", use_container_width=True):
                st.session_state.page = "Analytics"
                st.rerun()
            
            if st.button("🤖 Model Info", use_container_width=True):
                st.session_state.page = "Model Info"
                st.rerun()
        
        # Model performance card
        if metrics:
            st.markdown(f"""
            <div class='modern-card'>
                <h4 style='color: #2c3e50; margin-top: 0;'>AI Performance</h4>
                <div style='background: linear-gradient(90deg, #4ecdc4 {metrics['accuracy']*100}%, #f0f0f0 {metrics['accuracy']*100}%); height: 10px; border-radius: 5px; margin: 1rem 0;'></div>
                <p style='color: #7f8c8d; margin: 0;'><strong>Algorithm:</strong> {metrics['best_model']}</p>
                <p style='color: #7f8c8d; margin: 0;'><strong>Accuracy:</strong> {metrics['accuracy']:.1%}</p>
                <p style='color: #7f8c8d; margin: 0;'><strong>Trained:</strong> {metrics['model_timestamp'].split()[0]}</p>
            </div>
            """, unsafe_allow_html=True)

# Data Explorer Page
elif page == "🔍 Data Explorer":
    st.markdown('<h1 class="main-header">🔍 Data Explorer</h1>', unsafe_allow_html=True)
    
    # Quick stats
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("""
        <div class='modern-card'>
            <h4 style='color: #2c3e50; margin-top: 0;'>Dataset Overview</h4>
            <p style='color: #7f8c8d;'>Explore the diabetes dataset with interactive filtering and analysis tools.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("📥 Export Data", use_container_width=True):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="diabetes_dataset.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "📊 Statistics", "🎛️ Smart Filter"])
    
    with tab1:
        st.markdown("#### Dataset Sample")
        st.dataframe(df.head(12), use_container_width=True)
        st.caption(f"Displaying first 12 of {len(df):,} records")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.markdown("#### Data Types")
            dtype_info = pd.DataFrame({
                'Feature': df.columns,
                'Type': df.dtypes,
                'Missing': df.isnull().sum(),
                'Unique': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(dtype_info, use_container_width=True)
    
    with tab3:
        st.markdown("#### Smart Data Filtering")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age_range = st.slider("👤 Age Range", 
                                int(df['Age'].min()), 
                                int(df['Age'].max()), 
                                (25, 50))
            glucose_range = st.slider("🩸 Glucose Range", 
                                    int(df['Glucose'].min()), 
                                    int(df['Glucose'].max()), 
                                    (80, 140))
        
        with col2:
            bmi_range = st.slider("⚖️ BMI Range", 
                                float(df['BMI'].min()), 
                                float(df['BMI'].max()), 
                                (20.0, 35.0))
            outcome_filter = st.selectbox("🎯 Health Status", 
                                        ["All Records", "Healthy", "Diabetic"])
        
        # Apply filters
        filtered_df = df[
            (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) &
            (df['Glucose'] >= glucose_range[0]) & (df['Glucose'] <= glucose_range[1]) &
            (df['BMI'] >= bmi_range[0]) & (df['BMI'] <= bmi_range[1])
        ]
        
        if outcome_filter == "Healthy":
            filtered_df = filtered_df[filtered_df['Outcome'] == 0]
        elif outcome_filter == "Diabetic":
            filtered_df = filtered_df[filtered_df['Outcome'] == 1]
        
        # Results
        st.markdown(f"**🎯 Filtered Results: {len(filtered_df):,} patients**")
        
        if len(filtered_df) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("👥 Patients", len(filtered_df))
            with col2:
                diabetes_count = filtered_df['Outcome'].sum()
                st.metric("🩺 Diabetic", diabetes_count)
            with col3:
                diabetes_rate = (diabetes_count / len(filtered_df)) * 100
                st.metric("📈 Rate", f"{diabetes_rate:.1f}%")
            
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.warning("No records match the selected criteria.")

# Analytics Page
elif page == "📈 Analytics":
    st.markdown('<h1 class="main-header">📈 Advanced Analytics</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔗 Correlations", "🎯 Insights"])
    
    with tab1:
        st.markdown("#### Feature Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature = st.selectbox("Select Feature", df.columns[:-1])
            plot_type = st.selectbox("Visualization Type", ["Histogram", "Box Plot", "Density Plot"])
        
        with col2:
            show_stats = st.checkbox("Show Statistics", value=True)
            compare_outcome = st.checkbox("Compare by Outcome", value=True)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if plot_type == "Histogram":
            if compare_outcome:
                df[df['Outcome'] == 0][feature].hist(alpha=0.6, label='Healthy', ax=ax, color='#4ecdc4')
                df[df['Outcome'] == 1][feature].hist(alpha=0.6, label='Diabetic', ax=ax, color='#ff6b6b')
                ax.legend()
            else:
                df[feature].hist(alpha=0.7, ax=ax, color='#667eea')
            ax.set_title(f'Distribution of {feature}')
        
        elif plot_type == "Box Plot":
            if compare_outcome:
                df.boxplot(column=feature, by='Outcome', ax=ax)
            else:
                df.boxplot(column=feature, ax=ax)
        
        else:  # Density Plot
            if compare_outcome:
                df[df['Outcome'] == 0][feature].plot.density(ax=ax, label='Healthy', color='#4ecdc4')
                df[df['Outcome'] == 1][feature].plot.density(ax=ax, label='Diabetic', color='#ff6b6b')
                ax.legend()
            else:
                df[feature].plot.density(ax=ax, color='#667eea')
            ax.set_title(f'Density of {feature}')
        
        st.pyplot(fig)
        
        if show_stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{df[feature].mean():.2f}")
            with col2:
                st.metric("Median", f"{df[feature].median():.2f}")
            with col3:
                st.metric("Std Dev", f"{df[feature].std():.2f}")
            with col4:
                st.metric("IQR", f"{df[feature].quantile(0.75) - df[feature].quantile(0.25):.2f}")
    
    with tab2:
        st.markdown("#### Feature Correlation Analysis")
        
        # Correlation heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        corr_matrix = df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8}, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)
        
        # Top correlations
        st.markdown("#### Top Correlations with Diabetes")
        outcome_corr = df.corr()['Outcome'].sort_values(ascending=False)[1:6]
        
        cols = st.columns(5)
        for i, (feature, corr) in enumerate(outcome_corr.items()):
            with cols[i]:
                color = "#ff6b6b" if corr > 0 else "#4ecdc4"
                st.markdown(f"""
                <div style='text-align: center; padding: 1rem; background: white; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-top: 4px solid {color};'>
                    <div style='font-size: 0.8rem; color: #7f8c8d;'>{feature}</div>
                    <div style='font-size: 1.4rem; font-weight: bold; color: {color};'>{corr:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("#### Key Insights & Patterns")
        
        insights = [
            {
                "title": "Glucose Dominance",
                "content": "Blood glucose levels show the strongest correlation with diabetes outcomes",
                "impact": "Critical",
                "icon": "🩸"
            },
            {
                "title": "BMI Influence",
                "content": "Higher BMI significantly increases diabetes risk probability",
                "impact": "High",
                "icon": "⚖️"
            },
            {
                "title": "Age Factor",
                "content": "Diabetes prevalence increases steadily with patient age",
                "impact": "Medium",
                "icon": "👴"
            },
            {
                "title": "Genetic Markers",
                "content": "Diabetes pedigree function shows moderate correlation",
                "impact": "Medium",
                "icon": "🧬"
            }
        ]
        
        for insight in insights:
            impact_color = "#ff6b6b" if insight["impact"] == "Critical" else "#ffa726" if insight["impact"] == "High" else "#4ecdc4"
            
            st.markdown(f"""
            <div class='modern-card'>
                <div style='display: flex; align-items: start; gap: 1rem;'>
                    <span style='font-size: 2rem;'>{insight['icon']}</span>
                    <div style='flex: 1;'>
                        <h4 style='margin: 0 0 0.5rem 0; color: #2c3e50;'>{insight['title']}</h4>
                        <p style='margin: 0; color: #7f8c8d; line-height: 1.5;'>{insight['content']}</p>
                    </div>
                    <div style='background: {impact_color}; color: white; padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>
                        {insight['impact']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Prediction Page (Modern Design)
elif page == "🎯 Prediction":
    st.markdown('<h1 class="main-header">🎯 Diabetes Risk Assessment</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class='modern-card'>
        <h4 style='color: #2c3e50; margin-top: 0;'>AI-Powered Health Assessment</h4>
        <p style='color: #7f8c8d; margin-bottom: 0;'>
        Enter the patient's health parameters below for an instant diabetes risk assessment. 
        Our advanced AI analyzes multiple factors to provide accurate predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input sections in modern cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='modern-card'>
            <h4 style='color: #2c3e50; margin-top: 0;'>👤 Basic Information</h4>
        """, unsafe_allow_html=True)
        pregnancies = st.slider("Pregnancies", 0, 17, 3, key="preg")
        glucose = st.slider("Glucose Level", 0, 200, 120, key="glucose")
        blood_pressure = st.slider("Blood Pressure", 0, 122, 70, key="bp")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='modern-card'>
            <h4 style='color: #2c3e50; margin-top: 0;'>⚖️ Body Metrics</h4>
        """, unsafe_allow_html=True)
        skin_thickness = st.slider("Skin Thickness", 0, 99, 20, key="skin")
        insulin = st.slider("Insulin Level", 0, 846, 79, key="insulin")
        bmi = st.slider("Body Mass Index", 0.0, 67.1, 32.0, key="bmi")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='modern-card'>
            <h4 style='color: #2c3e50; margin-top: 0;'>🧬 Additional Factors</h4>
        """, unsafe_allow_html=True)
        diabetes_pedigree = st.slider("Genetic Factor", 0.08, 2.42, 0.47, key="dpf")
        age = st.slider("Age", 21, 81, 33, key="age")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Analyze Diabetes Risk", use_container_width=True, type="primary"):
            with st.spinner("🤖 AI is analyzing health parameters..."):
                # Create input array
                input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                     insulin, bmi, diabetes_pedigree, age]])
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0]
                
                # Display results
                st.markdown("### 📊 Assessment Results")
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class='prediction-card risk-high'>
                        <h1 style='margin: 0; font-size: 2.5rem;'>🟥 HIGH RISK</h1>
                        <p style='font-size: 1.2rem; margin: 1rem 0;'>Diabetes Probability: <strong>{probability[1]:.1%}</strong></p>
                        <div style='background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                            <p style='margin: 0;'>🩺 <strong>Recommendation:</strong> Consult healthcare professional immediately</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='prediction-card risk-low'>
                        <h1 style='margin: 0; font-size: 2.5rem;'>🟩 LOW RISK</h1>
                        <p style='font-size: 1.2rem; margin: 1rem 0;'>Diabetes Probability: <strong>{probability[1]:.1%}</strong></p>
                        <div style='background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                            <p style='margin: 0;'>💡 <strong>Recommendation:</strong> Maintain healthy lifestyle with regular checkups</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = probability[1] * 100,
                        title = {'text': "Diabetes Risk Score"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "red"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Feature importance
                    if hasattr(model, 'feature_importances_'):
                        feature_imp = pd.DataFrame({
                            'feature': feature_names,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=True)
                        
                        fig = px.bar(feature_imp, x='importance', y='feature', 
                                    title='Key Influencing Factors',
                                    orientation='h',
                                    color='importance',
                                    color_continuous_scale='Viridis')
                        st.plotly_chart(fig, use_container_width=True)

# Model Info Page
elif page == "🤖 Model Info":
    st.markdown('<h1 class="main-header">🤖 AI Model Information</h1>', unsafe_allow_html=True)
    
    if metrics:
        # Model overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>AI Algorithm</h3>
                <h2>{metrics['best_model']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Prediction Accuracy</h3>
                <h2>{metrics['accuracy']:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Training Date</h3>
                <h2>{metrics['model_timestamp'].split()[0]}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Performance metrics
    st.markdown('<div class="section-header">📊 Performance Metrics</div>', unsafe_allow_html=True)
    
    # Generate predictions
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    predictions = model.predict(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)
    
    # Metrics in modern layout
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Accuracy", f"{accuracy:.1%}")
    with col2:
        st.metric("📏 Precision", f"{precision:.1%}")
    with col3:
        st.metric("🔍 Recall", f"{recall:.1%}")
    with col4:
        st.metric("⚖️ F1-Score", f"{f1:.1%}")
    
    # Confusion Matrix
    st.markdown("#### 🎯 Confusion Matrix")
    cm = confusion_matrix(y, predictions)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Diabetic'],
                yticklabels=['Healthy', 'Diabetic'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Model Performance Matrix')
    st.pyplot(fig)
    
    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        st.markdown("#### 🔍 Feature Importance Analysis")
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(feature_importance, x='importance', y='feature', 
                    title='Feature Impact on Predictions',
                    orientation='h',
                    color='importance',
                    color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: white; padding: 1rem;'>
    <h4 style='margin: 0;'>🩺 DiabetesAI</h4>
    <p style='margin: 0; font-size: 0.8rem; opacity: 0.8;'>Advanced Prediction System</p>
    <p style='margin: 0; font-size: 0.7rem; opacity: 0.6;'>For educational use only</p>
</div>
""", unsafe_allow_html=True)

print("🚀 Modern DiabetesAI app is ready!")