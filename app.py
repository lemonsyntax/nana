#!/usr/bin/env python3
"""
Student Performance Prediction - Web Application

A comprehensive Streamlit app with:
1. Interactive prediction interface
2. Data analysis and visualization
3. Model performance insights
4. Feature importance analysis
5. Model comparison tools
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import joblib
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_collection import DataCollector
from data_preprocessing import DataPreprocessor
from neural_network_model import StudentPerformanceModel

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .feature-input {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .grade-a { background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important; }
    .grade-b { background: linear-gradient(135deg, #17a2b8 0%, #6f42c1 100%) !important; }
    .grade-c { background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%) !important; }
    .grade-d { background: linear-gradient(135deg, #fd7e14 0%, #e83e8c 100%) !important; }
    .grade-f { background: linear-gradient(135deg, #dc3545 0%, #6f42c1 100%) !important; }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: black;
    }
    
    .success-box {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_preprocessor():
    """Load the trained model and preprocessor (always use 11 features)"""
    try:
        models_dir = 'models'
        if not os.path.exists(models_dir):
            return None, None
        
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
        
        if not model_files:
            return None, None
        
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(models_dir, latest_model)
        
        # Always use 11 features
        input_dim = 11
        model = StudentPerformanceModel(input_dim=input_dim)
        model.load_model(model_path)
        
        # Use enhanced preprocessor with all features
        preprocessor = DataPreprocessor()
        collector = DataCollector()
        df = collector.load_csv()
        if df is not None:
            X_train, X_test, y_train, y_test = preprocessor.preprocess_data(df)
            st.success(f"✅ Enhanced preprocessor loaded! Using {input_dim} features.")
        else:
            st.error("❌ Could not load dataset")
            return None, None
        
        return model, preprocessor
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

@st.cache_data
def load_data():
    """Load the dataset"""
    try:
        collector = DataCollector()
        df = collector.load_csv()
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def main():
    """Main application function"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Student Performance Predictor</h1>
        <p>Predict student academic performance using advanced machine learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and data
    model, preprocessor = load_model_and_preprocessor()
    df = load_data()
    
    if model is None or preprocessor is None:
        st.error("⚠️ Model not found! Please run the training script first.")
        st.info("💡 Run: `python train_model.py` to train the model")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.markdown("## 📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Home", "📈 Predict Performance", "📊 Data Insights", "ℹ️ About"]
    )
    
    if page == "🏠 Home":
        show_home_page(df)
    elif page == "📈 Predict Performance":
        show_prediction_page(model, preprocessor)
    elif page == "📊 Data Insights":
        show_analysis_page(df)
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page(df):
    """Display the home page"""
    st.markdown("""
    ## 🎯 Welcome to the Student Performance Prediction System!
    
    This intelligent system helps predict student exam scores based on various factors that influence learning outcomes using our comprehensive dataset.
    
    ### ✨ What You Can Do:
    
    - **📈 Make Predictions**: Input student information and get accurate exam score predictions
    - **📊 Explore Insights**: Understand what factors most influence academic success
    - **🎯 Get Recommendations**: Receive personalized insights for improvement
    
    ### 📋 Key Factors We Consider:
    
    | Category | Examples |
    |----------|----------|
    | **Study Habits** | Hours studied, attendance rate |
    | **Academic History** | Previous test scores |
    | **Family Background** | Parental education level |
    | **Activities** | Extracurricular participation |
    | **Support Systems** | Access to resources, tutoring |
    
    ### 🎯 Prediction Target:
    - **Exam Score**: Predicted exam score on a 0-100 scale
    """)
    
    if df is not None:
        # Quick stats with better styling
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📚 Total Students", len(df), help="Number of students in our dataset")
        with col2:
            st.metric("📊 Average Exam Score", f"{df['Exam_Score'].mean():.1f}", help="Average exam score across all students")
        with col3:
            st.metric("🏆 Highest Score", f"{df['Exam_Score'].max():.1f}", help="Best performing student's score")
        with col4:
            st.metric("📈 Lowest Score", f"{df['Exam_Score'].min():.1f}", help="Lowest performing student's score")
        
        # Quick visualization
        st.markdown("### 📊 Exam Score Distribution")
        fig = px.histogram(df, x='Exam_Score', nbins=20, 
                          title="Distribution of Exam Scores",
                          labels={'Exam_Score': 'Exam Score', 'count': 'Number of Students'},
                          color_discrete_sequence=['#667eea'])
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show dataset info
        st.markdown("### 📋 Dataset Information")
        st.info(f"""
        **Dataset**: StudentPerformanceFactors.csv
        **Features**: {len(df.columns)} columns including study habits, academic history, and demographic factors
        **Target**: Exam_Score (0-100 scale)
        **Data Quality**: Clean dataset with comprehensive student information
        """)

def show_prediction_page(model, preprocessor):
    """Display the prediction page (updated UI, always 11 features)"""
    st.markdown("## 📈 Student Performance Prediction")
    st.markdown("""
    <div class="info-box">
        💡 <strong>How to use:</strong> Fill in the student information below and click "Predict" to get an accurate prediction of their exam score.
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown("### Enter Student Data to Predict Performance")
        
        # Core academic features
        col1, col2 = st.columns(2)
        with col1:
            hours_studied = st.number_input("Hours Studied", min_value=0.0, max_value=100.0, value=5.0, help="Enter the number of hours the student studied for the exam.")
            previous_score = st.number_input("Previous Test Score", min_value=0.0, max_value=100.0, value=75.0, help="Enter the score from the student's previous test (0-100).")
            attendance = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, value=90.0, help="Enter the attendance rate in percentage (0-100%).")
            sleep_hours = st.number_input("Sleep Hours per Night", min_value=0.0, max_value=24.0, value=8.0, help="Average hours of sleep per night")
            tutoring = st.number_input("Tutoring Sessions", min_value=0, max_value=50, value=0, help="Number of tutoring sessions attended")
        
        with col2:
            extracurricular = st.selectbox("Extra-Curricular Activities", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No", help="Does the student participate in extra-curricular activities?")
            parent_edu = st.selectbox("Parent's Education Level", options=[1, 2, 3, 4], format_func=lambda x: {1: 'High School', 2: 'College/Undergraduate', 3: 'Graduate', 4: 'Postgraduate'}[x], help="Highest education level of the student's parents")
            family_income = st.selectbox("Family Income Level", options=[1, 2, 3], format_func=lambda x: {1: 'Low', 2: 'Medium', 3: 'High'}[x], help="Family income level")
            teacher_quality = st.selectbox("Teacher Quality", options=[1, 2, 3], format_func=lambda x: {1: 'Low', 2: 'Medium', 3: 'High'}[x], help="Perceived quality of teaching")
            peer_influence = st.selectbox("Peer Influence", options=[1, 2, 3], format_func=lambda x: {1: 'Negative', 2: 'Neutral', 3: 'Positive'}[x], help="Influence of peers on academic performance")
            internet_access = st.selectbox("Internet Access", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No", help="Does the student have reliable internet access?")
        
        submitted = st.form_submit_button("🎯 Predict Performance")

        if submitted:
            try:
                # Always use all 11 features in the correct order
                input_data = {
                    'Hours_Studied': hours_studied,
                    'Previous_Scores': previous_score,
                    'Attendance': attendance,
                    'Extracurricular_Activities': extracurricular,
                    'Parental_Education_Level': parent_edu,
                    'Sleep_Hours': sleep_hours,
                    'Tutoring_Sessions': tutoring,
                    'Family_Income': family_income,
                    'Teacher_Quality': teacher_quality,
                    'Peer_Influence': peer_influence,
                    'Internet_Access': internet_access
                }
                input_df = pd.DataFrame([input_data])
                X_transformed = preprocessor.transform_new_data(input_df)
                prediction = model.predict(X_transformed)[0][0]
                # Display results with better formatting
                st.markdown("### 🎯 Prediction Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📊 Predicted Exam Score", f"{prediction:.1f}")
                with col2:
                    # Grade classification
                    if prediction >= 90:
                        grade = "A"
                        grade_color = "grade-a"
                    elif prediction >= 80:
                        grade = "B"
                        grade_color = "grade-b"
                    elif prediction >= 70:
                        grade = "C"
                        grade_color = "grade-c"
                    elif prediction >= 60:
                        grade = "D"
                        grade_color = "grade-d"
                    else:
                        grade = "F"
                        grade_color = "grade-f"
                    st.markdown(f"""
                    <div class="prediction-card {grade_color}">
                        <h3>Grade: {grade}</h3>
                        <p>Performance Level</p>
                    </div>
                    """, unsafe_allow_html=True)
                # Show insights
                st.markdown("### 💡 Insights")
                if prediction >= 80:
                    st.success("🎉 Excellent performance predicted! This student shows strong academic potential.")
                elif prediction >= 70:
                    st.info("👍 Good performance predicted. There's room for improvement with focused effort.")
                elif prediction >= 60:
                    st.warning("⚠️ Average performance predicted. Consider additional support and study strategies.")
                else:
                    st.error("📚 Below average performance predicted. Intensive support and intervention recommended.")
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.info("💡 This might be due to missing features. Please ensure all required data is provided.")

def show_analysis_page(df):
    """Display the analysis page"""
    st.markdown("## 📊 Data Insights & Analysis")
    
    if df is None:
        st.error("❌ Data not available")
        return
    
    # Overview with better styling
    st.markdown("### 📈 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📚 Total Students", len(df))
        st.metric("📊 Features", len(df.columns))
    with col2:
        st.metric("📈 Average Exam Score", f"{df['Exam_Score'].mean():.1f}")
        st.metric("📊 Score Std Dev", f"{df['Exam_Score'].std():.1f}")
    with col3:
        st.metric("👨 Male Students", f"{len(df[df['Gender'] == 'Male'])} ({len(df[df['Gender'] == 'Male'])/len(df)*100:.1f}%)")
        st.metric("👩 Female Students", f"{len(df[df['Gender'] == 'Female'])} ({len(df[df['Gender'] == 'Female'])/len(df)*100:.1f}%)")
    
    # Exam score distribution
    st.markdown("### 📊 Exam Score Distribution")
    fig = px.histogram(df, x='Exam_Score', nbins=20, 
                      title="Distribution of Exam Scores",
                      labels={'Exam_Score': 'Exam Score', 'count': 'Number of Students'},
                      color_discrete_sequence=['#667eea'])
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.markdown("### 🔗 Feature Correlations")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()['Exam_Score'].sort_values(ascending=False)
    correlations = correlations[correlations.index != 'Exam_Score']
    
    fig = px.bar(x=correlations.values, y=correlations.index, 
                 orientation='h',
                 title="Feature Correlations with Exam Score",
                 labels={'x': 'Correlation Coefficient', 'y': 'Features'},
                 color=correlations.values,
                 color_continuous_scale='RdBu')
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed analysis
    st.markdown("### 📋 Detailed Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📚 Academic Factors", "👥 Demographics", "🏃 Lifestyle"])
    
    with tab1:
        st.markdown("#### Academic Performance Factors")
        
        # Previous scores vs Exam score
        fig = px.scatter(df, x='Previous_Scores', y='Exam_Score', color='Gender',
                        title="Previous Scores vs Exam Score",
                        labels={'Previous_Scores': 'Previous Scores', 'Exam_Score': 'Exam Score'},
                        color_discrete_map={'Male': '#667eea', 'Female': '#764ba2'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Hours studied analysis
        fig = px.scatter(df, x='Hours_Studied', y='Exam_Score',
                        title="Hours Studied vs Exam Score",
                        labels={'Hours_Studied': 'Hours Studied', 'Exam_Score': 'Exam Score'},
                        color_discrete_sequence=['#667eea'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Attendance analysis
        attendance_stats = df.groupby('Attendance')['Exam_Score'].agg(['mean', 'count']).reset_index()
        fig = px.bar(attendance_stats, x='Attendance', y='mean',
                    title="Average Exam Score by Attendance",
                    labels={'Attendance': 'Attendance', 'mean': 'Average Exam Score'},
                    color_discrete_sequence=['#667eea'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Demographic Analysis")
        
        # Gender analysis
        gender_stats = df.groupby('Gender')['Exam_Score'].agg(['mean', 'count']).reset_index()
        fig = px.bar(gender_stats, x='Gender', y='mean',
                    title="Average Exam Score by Gender",
                    labels={'Gender': 'Gender', 'mean': 'Average Exam Score'},
                    color_discrete_sequence=['#667eea', '#764ba2'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Parental education analysis
        edu_stats = df.groupby('Parental_Education_Level')['Exam_Score'].agg(['mean', 'count']).reset_index()
        fig = px.bar(edu_stats, x='Parental_Education_Level', y='mean',
                    title="Average Exam Score by Parental Education Level",
                    labels={'Parental_Education_Level': 'Parental Education Level', 'mean': 'Average Exam Score'},
                    color_discrete_sequence=['#667eea'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("#### Lifestyle Factors")
        
        # Extracurricular activities
        extra_stats = df.groupby('Extracurricular_Activities')['Exam_Score'].agg(['mean', 'count']).reset_index()
        fig = px.bar(extra_stats, x='Extracurricular_Activities', y='mean',
                    title="Average Exam Score by Extracurricular Participation",
                    labels={'Extracurricular_Activities': 'Extracurricular Activities', 'mean': 'Average Exam Score'},
                    color_discrete_sequence=['#667eea'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Sleep hours vs Exam score
        if 'Sleep_Hours' in df.columns:
            fig = px.scatter(df, x='Sleep_Hours', y='Exam_Score',
                            title="Sleep Hours vs Exam Score",
                            labels={'Sleep_Hours': 'Sleep Hours', 'Exam_Score': 'Exam Score'},
                            color_discrete_sequence=['#667eea'])
            st.plotly_chart(fig, use_container_width=True)

def show_about_page():
    """Display the about page"""
    st.markdown("## ℹ️ About This System")
    
    st.markdown("""
    ### 🎓 Student Performance Prediction System
    
    This intelligent system helps predict student academic performance using advanced machine learning techniques.
    
    ### 🎯 What We Do
    
    We analyze various factors that influence student success and provide accurate predictions to help:
    - **Students** understand their potential performance
    - **Educators** identify students who might need additional support
    - **Parents** understand factors affecting their child's academic success
    
    ### 📊 What We Consider
    
    Our system analyzes multiple factors including:
    
    #### Academic Factors
    - Previous grades and performance history
    - Study time and learning habits
    - Attendance and participation
    
    #### Personal Factors
    - Age and maturity level
    - Health and well-being
    - Motivation and engagement
    
    #### Family Background
    - Parental education levels
    - Family support and relationships
    - Socioeconomic factors
    
    #### Lifestyle Factors
    - Free time activities
    - Social interactions
    - Health habits
    
    ### 🎯 Our Predictions
    
    - **Target**: Exam Score on a 0-100 scale
    - **Accuracy**: High precision predictions based on comprehensive data analysis
    - **Interpretation**: Clear grade levels (A-F) with actionable insights
    
    ### 🔒 Privacy & Ethics
    
    - This system uses synthetic data for demonstration purposes
    - No real student information is collected or stored
    - Educational AI should be used responsibly and transparently
    - Consider potential biases and fairness implications
    
    ### 💡 How to Use
    
    1. Navigate to "Predict Performance"
    2. Input student information using the form
    3. Click "Predict" to get results
    4. Review the prediction and recommendations
    5. Explore data insights for deeper understanding
    
    ---
    
    **Built with ❤️ for educational advancement**
    
    *Empowering students, educators, and parents with data-driven insights*
    """)

if __name__ == "__main__":
    main() 