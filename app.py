import streamlit as st
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
import findspark

# Import our custom modules
from models.preprocessing import initialize_spark, load_data, extract_features, prepare_input_name, extract_first_name
from models.ml_models import train_logistic_regression, train_naive_bayes, predict_gender, evaluate_model, save_model, load_model
from utils.helpers import get_data_summary, create_gender_distribution_chart, create_name_length_chart, show_prediction_results, compare_prediction_results

# Initialize findspark to locate Spark
findspark.init()

# Set page configuration
st.set_page_config(
    page_title="Name Gender Predictor",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define paths
DATA_PATH = "data/names-mr.csv"
LR_MODEL_PATH = "models/saved/logistic_regression"
NB_MODEL_PATH = "models/saved/naive_bayes"

# Create directory for saved models if it doesn't exist
os.makedirs("models/saved", exist_ok=True)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #7f8c8d;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        font-size: 1rem;
        font-weight: 600;
        background-color: #f8f9fa;
        border-radius: 0.5rem 0.5rem 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for storing Spark session and models
if 'spark' not in st.session_state:
    st.session_state.spark = None
if 'lr_model' not in st.session_state:
    st.session_state.lr_model = None
if 'nb_model' not in st.session_state:
    st.session_state.nb_model = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Function to initialize Spark and load data
@st.cache_resource(show_spinner=False)
def initialize_app():
    with st.spinner('Initializing Spark session...'):
        spark = initialize_spark()
    return spark

# Function to load and process data
def load_and_process_data(spark, data_path):
    with st.spinner('Loading and processing data...'):
        # Load and preprocess data
        data = load_data(spark, data_path)
        
        # Extract features
        processed_data = extract_features(data)
        
        # Get data summary for visualization
        data_summary = get_data_summary(data)
        
        # Split data into training and testing sets
        train_data, test_data = processed_data.randomSplit([0.8, 0.2], seed=42)
        
        return data, processed_data, train_data, test_data, data_summary

# Function to train or load models
def get_models(train_data):
    # Try to load models if they exist, otherwise train new ones
    try:
        lr_model = load_model(st.session_state.spark, "logistic_regression", LR_MODEL_PATH)
        nb_model = load_model(st.session_state.spark, "naive_bayes", NB_MODEL_PATH)
        st.success("Loaded pre-trained models successfully!")
    except:
        with st.spinner('Training Logistic Regression model...'):
            lr_model = train_logistic_regression(train_data)
            save_model(lr_model, LR_MODEL_PATH)
        
        with st.spinner('Training Naive Bayes model...'):
            nb_model = train_naive_bayes(train_data)
            save_model(nb_model, NB_MODEL_PATH)
        
        st.success("Models trained and saved successfully!")
    
    return lr_model, nb_model

# Main app function
def main():
    # Display header
    st.markdown('<h1 class="main-header">Name Gender Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Predict the gender associated with a name using machine learning models</p>', unsafe_allow_html=True)
    
    # Initialize Spark session
    if st.session_state.spark is None:
        st.session_state.spark = initialize_app()
    
    # Sidebar for app controls
    with st.sidebar:
        st.markdown('<h2 class="sub-header">Controls</h2>', unsafe_allow_html=True)
        
        # Load data button
        if not st.session_state.data_loaded:
            if st.button("Load Data & Train Models"):
                try:
                    # Load and process data
                    data, processed_data, train_data, test_data, data_summary = load_and_process_data(
                        st.session_state.spark, DATA_PATH
                    )
                    
                    # Store in session state
                    st.session_state.data = data
                    st.session_state.processed_data = processed_data
                    st.session_state.train_data = train_data
                    st.session_state.test_data = test_data
                    st.session_state.data_summary = data_summary
                    
                    # Get or train models
                    lr_model, nb_model = get_models(train_data)
                    
                    # Store models in session state
                    st.session_state.lr_model = lr_model
                    st.session_state.nb_model = nb_model
                    
                    # Set data loaded flag
                    st.session_state.data_loaded = True
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.success("Data loaded and models ready!")
            
            if st.button("Reset App"):
                # Clear session state
                for key in list(st.session_state.keys()):
                    if key != 'spark':
                        del st.session_state[key]
                st.session_state.data_loaded = False
                st.experimental_rerun()
        
        # About section
        st.markdown("---")
        st.markdown('<h3>About</h3>', unsafe_allow_html=True)
        st.markdown(
            "This app predicts the gender associated with a name using machine learning models. "
            "It uses PySpark ML to train Logistic Regression and Naive Bayes models on Mauritanian names data."
        )
    
    # Main content area
    if not st.session_state.data_loaded:
        # Display placeholder content when data is not loaded
        st.info("Click the 'Load Data & Train Models' button in the sidebar to get started.")
        
        # Show loading spinner
        if st.sidebar.button("Show Demo"):
            with st.spinner("Loading demo..."):
                # Simulate loading process
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                st.success("Demo mode activated!")
                time.sleep(1)
                st.experimental_rerun()
    else:
        # Create tabs for different models
        tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Logistic Regression", "🧮 Naive Bayes"])
        
        with tab1:
            st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
            
            # Display data summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Names", st.session_state.data_summary["total_names"])
            with col2:
                st.metric("Male Names", st.session_state.data_summary["male_count"])
            with col3:
                st.metric("Female Names", st.session_state.data_summary["female_count"])
            
            # Display charts
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(create_gender_distribution_chart(st.session_state.data_summary))
            with col2:
                st.pyplot(create_name_length_chart(st.session_state.data_summary["pandas_df"]))
            
            # Display sample data
            st.markdown('<h3>Sample Data</h3>', unsafe_allow_html=True)
            sample_df = st.session_state.data.limit(10).toPandas()
            st.dataframe(sample_df, use_container_width=True)
        
        # Function for prediction interface (reused in both model tabs)
        def prediction_interface(model, model_type):
            st.markdown(f'<h2 class="sub-header">{model_type} Model</h2>', unsafe_allow_html=True)
            
            # Input form for prediction
            name_input = st.text_input("Enter a name to predict:", key=f"{model_type}_input")
            
            if name_input:
                # Extract first name if multiple names are provided
                first_name = extract_first_name(name_input)
                
                # Prepare input for prediction
                input_data = prepare_input_name(st.session_state.spark, first_name)
                
                # Make prediction
                prediction = predict_gender(model, input_data)
                
                # Display prediction results
                show_prediction_results(prediction, model_type)
                
                return prediction
            return None
        
        with tab2:
            # Logistic Regression model tab
            lr_prediction = prediction_interface(st.session_state.lr_model, "Logistic Regression")
        
        with tab3:
            # Naive Bayes model tab
            nb_prediction = prediction_interface(st.session_state.nb_model, "Naive Bayes")
        
        # Compare predictions if both are available
        if lr_prediction and nb_prediction:
            st.markdown("---")
            st.markdown('<h2 class="sub-header">Model Comparison</h2>', unsafe_allow_html=True)
            compare_prediction_results(lr_prediction, nb_prediction)

# Run the app
if __name__ == "__main__":
    main()