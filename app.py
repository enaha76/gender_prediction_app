import streamlit as st
import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Display Python and package version information
st.set_page_config(
    page_title="Name Gender Predictor",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Check Python version
python_version = sys.version
st.write(f"Python Version: {python_version}")

try:
    # Try to import PySpark
    import pyspark
    from pyspark.sql import SparkSession
    import findspark
    
    # Initialize findspark to locate Spark
    findspark.init()
    
    # Import our custom modules
    from models.preprocessing import initialize_spark, load_data, extract_features, prepare_input_name, extract_first_name
    from models.ml_models import train_logistic_regression, train_naive_bayes, predict_gender, evaluate_model, save_model, load_model
    from utils.helpers import get_data_summary, create_gender_distribution_chart, create_name_length_chart, show_prediction_results, compare_prediction_results
    
    # The rest of your app code follows...
    # ...
    
    # Main app function definition
    def main():
        # Display header
        st.markdown('<h1 class="main-header">Name Gender Predictor</h1>', unsafe_allow_html=True)
        st.markdown('<p class="info-text">Predict the gender associated with a name using machine learning models</p>', unsafe_allow_html=True)
        
        # Initialize session state for storing Spark session and models
        if 'spark' not in st.session_state:
            st.session_state.spark = None
        if 'lr_model' not in st.session_state:
            st.session_state.lr_model = None
        if 'nb_model' not in st.session_state:
            st.session_state.nb_model = None
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        
        # Initialize Spark session
        if st.session_state.spark is None:
            try:
                with st.spinner('Initializing Spark session...'):
                    st.session_state.spark = initialize_spark()
                st.success("Spark initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing Spark: {str(e)}")
                st.info("Please check your Java installation and try again.")
                return
        
        # Sidebar for app controls
        with st.sidebar:
            st.markdown('<h2 class="sub-header">Controls</h2>', unsafe_allow_html=True)
            
            # Load data button
            if not st.session_state.data_loaded:
                if st.button("Load Data & Train Models"):
                    try:
                        # Check if data file exists
                        DATA_PATH = "data/names-mr.csv"
                        if not os.path.exists(DATA_PATH):
                            st.error(f"Data file not found: {DATA_PATH}")
                            st.info("Please upload the names-mr.csv file to the data directory.")
                            return
                        
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
        
        # Rest of your app code...
        # ...
    
    # Run the app
    if __name__ == "__main__":
        main()

except ImportError as e:
    st.error(f"Error importing required packages: {str(e)}")
    st.info("Please check your dependencies and make sure they're correctly installed.")
    st.write("Required packages:")
    st.code("""
    streamlit>=1.30.0
    pyspark>=3.5.0
    findspark>=1.4.0
    matplotlib>=3.8.0
    seaborn>=0.13.0
    pandas>=2.1.0
    numpy>=1.26.0
    scikit-learn>=1.3.0
    joblib>=1.3.1
    """)
    
except Exception as e:
    st.error(f"Unexpected error: {str(e)}")
    st.info("Please check your installation and try again.")