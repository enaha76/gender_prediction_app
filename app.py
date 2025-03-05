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
    
    # Define function to load and process data
    def load_and_process_data(spark, data_path):
        """Load and process the name data."""
        # Load raw data
        data = load_data(spark, data_path)
        
        # Extract features for machine learning
        processed_data = extract_features(data)
        
        # Split data into training and testing sets (80% train, 20% test)
        train_data, test_data = processed_data.randomSplit([0.8, 0.2], seed=42)
        
        # Get data summary for visualization
        data_summary = get_data_summary(data)
        
        return data, processed_data, train_data, test_data, data_summary
    
    # Define function to get or train models
    def get_models(train_data):
        """Get pre-trained models or train new ones."""
        # Define paths for saved models
        lr_model_path = "models/saved/logistic_regression_model"
        nb_model_path = "models/saved/naive_bayes_model"
        
        try:
            # Try to load pre-trained models
            with st.spinner('Loading pre-trained models...'):
                lr_model = load_model(st.session_state.spark, "logistic_regression", lr_model_path)
                nb_model = load_model(st.session_state.spark, "naive_bayes", nb_model_path)
            st.success("Pre-trained models loaded successfully!")
        except:
            # Train new models if pre-trained models are not available
            with st.spinner('Training Logistic Regression model...'):
                lr_model = train_logistic_regression(train_data)
                save_model(lr_model, lr_model_path)
            
            with st.spinner('Training Naive Bayes model...'):
                nb_model = train_naive_bayes(train_data)
                save_model(nb_model, nb_model_path)
            
            st.success("Models trained and saved successfully!")
        
        return lr_model, nb_model
    
    # Main app function definition
    def main():
        # Add custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #2c3e50;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.8rem;
            color: #2c3e50;
        }
        .info-text {
            font-size: 1.2rem;
            color: #7f8c8d;
            margin-bottom: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)
        
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
                        
                        # Evaluate models on test data
                        with st.spinner('Evaluating models...'):
                            lr_metrics = evaluate_model(lr_model, test_data)
                            nb_metrics = evaluate_model(nb_model, test_data)
                            
                        # Store models and metrics in session state
                        st.session_state.lr_model = lr_model
                        st.session_state.nb_model = nb_model
                        st.session_state.lr_metrics = lr_metrics
                        st.session_state.nb_metrics = nb_metrics
                        
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
        
        # Main content
        if st.session_state.data_loaded:
            # Create tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Logistic Regression", "Naive Bayes", "Model Comparison"])
            
            with tab1:
                st.markdown("## Data Overview")
                col1, col2, col3 = st.columns(3)
                
                # Display summary statistics
                with col1:
                    st.metric("Total Names", st.session_state.data_summary["total_names"])
                
                with col2:
                    st.metric("Male Names", st.session_state.data_summary["male_count"])
                
                with col3:
                    st.metric("Female Names", st.session_state.data_summary["female_count"])
                
                # Display gender distribution chart
                st.markdown("### Gender Distribution")
                gender_chart = create_gender_distribution_chart(st.session_state.data_summary)
                st.pyplot(gender_chart)
                
                # Display name length distribution chart
                st.markdown("### Name Length Distribution")
                length_chart = create_name_length_chart(st.session_state.data_summary["pandas_df"])
                st.pyplot(length_chart)
                
                # Display sample data
                st.markdown("### Sample Data")
                st.dataframe(st.session_state.data_summary["pandas_df"].head(10))
            
            with tab2:
                st.markdown("## Logistic Regression Model")
                
                # Display model metrics
                st.markdown("### Model Performance")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{st.session_state.lr_metrics['accuracy']:.3f}")
                
                with col2:
                    st.metric("Precision", f"{st.session_state.lr_metrics['precision']:.3f}")
                
                with col3:
                    st.metric("Recall", f"{st.session_state.lr_metrics['recall']:.3f}")
                
                with col4:
                    st.metric("F1 Score", f"{st.session_state.lr_metrics['f1']:.3f}")
                
                # Name input for prediction
                st.markdown("### Predict Gender")
                input_name = st.text_input("Enter a name to predict gender:", "Mohamed", key="lr_input")
                
                if st.button("Predict", key="lr_predict_button"):
                    if input_name:
                        with st.spinner('Predicting...'):
                            # Extract first name if full name is provided
                            first_name = extract_first_name(input_name)
                            
                            # Prepare name for prediction
                            input_data = prepare_input_name(st.session_state.spark, first_name)
                            
                            # Predict gender
                            prediction = predict_gender(st.session_state.lr_model, input_data)
                            
                            # Show prediction results
                            show_prediction_results(prediction, "Logistic Regression")
                    else:
                        st.warning("Please enter a name.")
            
            with tab3:
                st.markdown("## Naive Bayes Model")
                
                # Display model metrics
                st.markdown("### Model Performance")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{st.session_state.nb_metrics['accuracy']:.3f}")
                
                with col2:
                    st.metric("Precision", f"{st.session_state.nb_metrics['precision']:.3f}")
                
                with col3:
                    st.metric("Recall", f"{st.session_state.nb_metrics['recall']:.3f}")
                
                with col4:
                    st.metric("F1 Score", f"{st.session_state.nb_metrics['f1']:.3f}")
                
                # Name input for prediction
                st.markdown("### Predict Gender")
                input_name = st.text_input("Enter a name to predict gender:", "Fatima", key="nb_input")
                
                if st.button("Predict", key="nb_predict_button"):
                    if input_name:
                        with st.spinner('Predicting...'):
                            # Extract first name if full name is provided
                            first_name = extract_first_name(input_name)
                            
                            # Prepare name for prediction
                            input_data = prepare_input_name(st.session_state.spark, first_name)
                            
                            # Predict gender
                            prediction = predict_gender(st.session_state.nb_model, input_data)
                            
                            # Show prediction results
                            show_prediction_results(prediction, "Naive Bayes")
                    else:
                        st.warning("Please enter a name.")
            
            with tab4:
                st.markdown("## Model Comparison")
                
                # Compare model metrics
                st.markdown("### Performance Comparison")
                metrics_df = pd.DataFrame({
                    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                    "Logistic Regression": [
                        st.session_state.lr_metrics["accuracy"],
                        st.session_state.lr_metrics["precision"],
                        st.session_state.lr_metrics["recall"],
                        st.session_state.lr_metrics["f1"]
                    ],
                    "Naive Bayes": [
                        st.session_state.nb_metrics["accuracy"],
                        st.session_state.nb_metrics["precision"],
                        st.session_state.nb_metrics["recall"],
                        st.session_state.nb_metrics["f1"]
                    ]
                })
                
                # Display metrics table
                st.dataframe(metrics_df.style.format({
                    "Logistic Regression": "{:.3f}",
                    "Naive Bayes": "{:.3f}"
                }))
                
                # Compare predictions for a name
                st.markdown("### Compare Predictions")
                input_name = st.text_input("Enter a name to compare predictions:", "Ahmed", key="compare_input")
                
                if st.button("Compare", key="compare_button"):
                    if input_name:
                        with st.spinner('Comparing predictions...'):
                            # Extract first name if full name is provided
                            first_name = extract_first_name(input_name)
                            
                            # Prepare name for prediction
                            input_data = prepare_input_name(st.session_state.spark, first_name)
                            
                            # Predict gender with both models
                            lr_prediction = predict_gender(st.session_state.lr_model, input_data)
                            nb_prediction = predict_gender(st.session_state.nb_model, input_data)
                            
                            # Compare prediction results
                            compare_prediction_results(lr_prediction, nb_prediction)
                    else:
                        st.warning("Please enter a name.")
                        
        else:
            # Display instructions when data is not loaded
            st.info("👈 Please click 'Load Data & Train Models' in the sidebar to start.")
            
            # Add placeholders for app visualization
            st.markdown("## App Preview")
            st.image("https://placeholder.pics/svg/800x400/DEDEDE/555555/Name%20Gender%20Predictor%20Preview", use_column_width=True)
    
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