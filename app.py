"""
Main Streamlit application for Gender Prediction App
"""
import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession

# Import utility modules
from utils.preprocess import clean_names, handle_missing_values, validate_data
from utils.features import extract_name_features, get_feature_names, get_feature_descriptions
from utils.models import (
    train_logistic_regression, 
    train_naive_bayes,
    predict_gender,
    save_model,
    load_model,
    get_model_info
)

# Configure page
st.set_page_config(
    page_title="Gender Prediction from Names",
    page_icon="👤",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #3498db;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .result-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .prediction-male {
        color: #2980b9;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .prediction-female {
        color: #e74c3c;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .probability {
        font-size: 1.1rem;
        color: #7f8c8d;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        background-color: #f1f1f1;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db !important;
        color: white !important;
    }
    .feature-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 4px solid #3498db;
    }
    .model-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        border: 1px solid #e0e0e0;
    }
    .model-advantage {
        color: #27ae60;
        margin-left: 20px;
    }
    .model-disadvantage {
        color: #e74c3c;
        margin-left: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Spark session
@st.cache_resource
def get_spark():
    return SparkSession.builder \
        .appName("Gender Prediction") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

def main():
    st.markdown('<h1 class="main-header">Gender Prediction from Names</h1>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Data & Training", "🧪 Logistic Regression", "🔍 Naive Bayes"])
    
    # Initialize Spark session
    spark = get_spark()
    
    # Create necessary directories if they don't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("models/lr_model", exist_ok=True)
    os.makedirs("models/nb_model", exist_ok=True)
    
    # Tab 1: Data & Training
    with tab1:
        st.markdown('<h2 class="sub-header">Upload Training Data</h2>', unsafe_allow_html=True)
        
        # File uploader for training data
        training_file = st.file_uploader("Upload training dataset (CSV)", type=["csv"])
        
        # Session state to track if models are trained
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
        
        if training_file is not None:
            # Save uploaded file to data directory
            with open(os.path.join("data", "names-mr.csv"), "wb") as f:
                f.write(training_file.getbuffer())
            
            # Load training data
            try:
                train_df = pd.read_csv(os.path.join("data", "names-mr.csv"), sep=';')
                
                # Display training data preview
                st.markdown('<h3 class="sub-header">Training Data Preview</h3>', unsafe_allow_html=True)
                st.write(train_df.head())
                
                # Check if required columns exist
                if "NOMPL" not in train_df.columns or "SEXE" not in train_df.columns:
                    st.error("Training data must contain 'NOMPL' and 'SEXE' columns!")
                else:
                    # Convert to Spark DataFrame
                    df_train = spark.createDataFrame(train_df)
                    
                    # Data exploration
                    st.markdown('<h3 class="sub-header">Data Exploration</h3>', unsafe_allow_html=True)
                    st.write(f"Number of samples: {df_train.count()}")
                    
                    # Clean and preprocess data
                    df_train = clean_names(df_train)
                    df_train = handle_missing_values(df_train)
                    
                    # Check data validity
                    if not validate_data(df_train):
                        st.error("Invalid data format! Please make sure gender values are 'M' or 'F'.")
                    else:
                        # Gender distribution
                        gender_counts = df_train.groupBy("SEXE").count().toPandas()
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            # Plot gender distribution
                            fig, ax = plt.subplots(figsize=(10, 5))
                            sns.barplot(x="SEXE", y="count", data=gender_counts, palette=["#e74c3c", "#2980b9"])
                            ax.set_title("Gender Distribution in Training Data")
                            ax.set_xlabel("Gender")
                            ax.set_ylabel("Count")
                            st.pyplot(fig)
                        
                        with col2:
                            # Display gender counts
                            st.write("Gender Distribution:")
                            for idx, row in gender_counts.iterrows():
                                gender = row['SEXE']
                                count = row['count']
                                st.markdown(f"**{gender}**: {count} names ({count/sum(gender_counts['count'])*100:.1f}%)")
                        
                        # Feature information
                        st.markdown('<h3 class="sub-header">Features Used for Prediction</h3>', unsafe_allow_html=True)
                        
                        feature_names = get_feature_names()
                        feature_descriptions = get_feature_descriptions()
                        
                        feature_cols = st.columns(3)
                        for i, feature in enumerate(feature_names):
                            with feature_cols[i % 3]:
                                st.markdown(f"""
                                <div class="feature-card">
                                    <h4>{feature}</h4>
                                    <p>{feature_descriptions[feature]}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Extract features
                        with st.spinner("Extracting features..."):
                            df_train = extract_name_features(df_train)
                        
                        # Train models button
                        train_col1, train_col2 = st.columns([1, 1])
                        with train_col1:
                            if st.button("Train Models", type="primary", key="train_button"):
                                with st.spinner("Training models..."):
                                    # Train Logistic Regression model
                                    lr_model, lr_accuracy, lr_label_indexer = train_logistic_regression(df_train)
                                    
                                    # Train Naive Bayes model
                                    nb_model, nb_accuracy, nb_label_indexer = train_naive_bayes(df_train)
                                    
                                    # Save models
                                    save_model(lr_model, "models/lr_model")
                                    save_model(nb_model, "models/nb_model")
                                    
                                    # Save in session state
                                    st.session_state.lr_model = lr_model
                                    st.session_state.nb_model = nb_model
                                    st.session_state.lr_label_indexer = lr_label_indexer
                                    st.session_state.nb_label_indexer = nb_label_indexer
                                    st.session_state.lr_accuracy = lr_accuracy
                                    st.session_state.nb_accuracy = nb_accuracy
                                    st.session_state.models_trained = True
                                    
                                    st.success("Models trained successfully! You can now make predictions in the model tabs.")
                        
                        with train_col2:
                            if st.session_state.models_trained:
                                # Show accuracy metrics
                                st.markdown("### Model Performance")
                                metric_col1, metric_col2 = st.columns(2)
                                with metric_col1:
                                    st.metric(
                                        "Logistic Regression Accuracy", 
                                        f"{st.session_state.lr_accuracy:.4f}",
                                        f"{st.session_state.lr_accuracy - st.session_state.nb_accuracy:.4f}"
                                    )
                                with metric_col2:
                                    st.metric(
                                        "Naive Bayes Accuracy", 
                                        f"{st.session_state.nb_accuracy:.4f}",
                                        f"{st.session_state.nb_accuracy - st.session_state.lr_accuracy:.4f}"
                                    )
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        else:
            st.info("Please upload a CSV file with names and gender labels.")
            
            # Demo data option
            if st.button("Use Demo Data"):
                # Create a sample of the data you provided
                demo_data = """NOMPL;SEXE
Fatimetou Ahmed Mbareck;F
Mariem Tah Mohamed Elmoktar Essaghir;F
Aicha Ahmed Beyhime;F
Nanna El Mounir Hame;F
Vatme Mohamed Magha;F
Fatimtou Zahra Ely Mohamed Lmin;F
El Hadj Samba Abou Diop;M
Zeinebou Ahmedou Akembi;F
Mama Mohamed Sidi M'Hamed;F
Bouye Ahmed Ahmed Djoume;M
Bouna Itawol Amrou Abdoullah;M
Lamine Amadou Ba;M
Halima Esghaier Mbarek;M
Vatma El Hasniya Saad Bouh Hamady;F
Aichete Mahfoudh Khlil;F
Mouadh Mahfoudh Ekhlil;M
Zeinabou Mohamed El Mostapha Abdallahi;F
Ayoub Mahfoudh Ekhlil;M
Mariem Mohamed Mahmoud Taleb Sid'Ahmed;F
Abderrahmane Mohamed Ahmed Sghair;M
Mohamed Vadel Mohamed Boide;M
Mariem Badi Bady;F"""
                
                # Save demo data
                os.makedirs("data", exist_ok=True)
                with open(os.path.join("data", "names-mr.csv"), "w") as f:
                    f.write(demo_data)
                
                # Load demo data
                train_df = pd.read_csv(os.path.join("data", "names-mr.csv"), sep=';')
                df_train = spark.createDataFrame(train_df)
                
                st.markdown('<h3 class="sub-header">Demo Data Preview</h3>', unsafe_allow_html=True)
                st.write(train_df.head())
                
                # Data exploration
                st.markdown('<h3 class="sub-header">Data Exploration</h3>', unsafe_allow_html=True)
                st.write(f"Number of samples: {df_train.count()}")
                
                # Clean and preprocess data
                df_train = clean_names(df_train)
                df_train = handle_missing_values(df_train)
                
                # Gender distribution
                gender_counts = df_train.groupBy("SEXE").count().toPandas()
                
                # Plot gender distribution
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(x="SEXE", y="count", data=gender_counts, palette=["#e74c3c", "#2980b9"])
                ax.set_title("Gender Distribution in Demo Data")
                ax.set_xlabel("Gender")
                ax.set_ylabel("Count")
                st.pyplot(fig)
                
                # Extract features
                with st.spinner("Extracting features..."):
                    df_train = extract_name_features(df_train)
                
                # Train models button
                if st.button("Train Models on Demo Data", type="primary"):
                    with st.spinner("Training models on demo data..."):
                        # Train Logistic Regression model
                        lr_model, lr_accuracy, lr_label_indexer = train_logistic_regression(df_train)
                        
                        # Train Naive Bayes model
                        nb_model, nb_accuracy, nb_label_indexer = train_naive_bayes(df_train)
                        
                        # Save models
                        save_model(lr_model, "models/lr_model")
                        save_model(nb_model, "models/nb_model")
                        
                        # Save in session state
                        st.session_state.lr_model = lr_model
                        st.session_state.nb_model = nb_model
                        st.session_state.lr_label_indexer = lr_label_indexer
                        st.session_state.nb_label_indexer = nb_label_indexer
                        st.session_state.lr_accuracy = lr_accuracy
                        st.session_state.nb_accuracy = nb_accuracy
                        st.session_state.models_trained = True
                        
                        st.success("Models trained successfully! You can now make predictions in the model tabs.")
    
    # Tab 2: Logistic Regression
    with tab2:
        st.markdown('<h2 class="sub-header">Logistic Regression Prediction</h2>', unsafe_allow_html=True)
        
        # Model information
        lr_info = get_model_info('lr')
        st.markdown(f"""
        <div class="model-card">
            <h3>{lr_info['name']}</h3>
            <p>{lr_info['description']}</p>
            <h4>Advantages:</h4>
            {''.join([f'<p class="model-advantage">✓ {adv}</p>' for adv in lr_info['advantages']])}
            <h4>Disadvantages:</h4>
            {''.join([f'<p class="model-disadvantage">✗ {dis}</p>' for dis in lr_info['disadvantages']])}
        </div>
        """, unsafe_allow_html=True)
        
        # Check if model is trained
        model_exists = os.path.exists("models/lr_model") or st.session_state.models_trained
        
        if not model_exists:
            st.warning("Please train the models in the 'Data & Training' tab first!")
        else:
            # Load model if not in session state
            if not st.session_state.models_trained and 'lr_model' not in st.session_state:
                try:
                    st.session_state.lr_model = load_model("models/lr_model")
                    st.session_state.models_trained = True
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    st.warning("Please train the models in the 'Data & Training' tab first!")
            
            # Input for prediction
            input_name = st.text_input("Enter a name:", key="lr_input")
            
            if st.button("Predict Gender", key="lr_predict"):
                if input_name:
                    with st.spinner("Predicting..."):
                        # Create temporary DataFrame
                        input_df = spark.createDataFrame([(input_name,)], ["NOMPL"])
                        
                        # Clean and extract features
                        input_df = clean_names(input_df)
                        input_df = extract_name_features(input_df)
                        
                        # Make prediction
                        gender, probability = predict_gender(
                            st.session_state.lr_model, 
                            input_name, 
                            st.session_state.lr_label_indexer, 
                            spark
                        )
                        
                        # Display result with animation
                        with st.empty():
                            for i in range(5):
                                if i < 4:
                                    st.markdown(f"""
                                    <div class="result-container">
                                        <h3>Analyzing name: {input_name}</h3>
                                        <p>Processing{'.' * (i + 1)}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    time.sleep(0.3)
                                else:
                                    gender_class = "prediction-male" if gender == "M" else "prediction-female"
                                    gender_text = "Male" if gender == "M" else "Female"
                                    
                                    st.markdown(f"""
                                    <div class="result-container">
                                        <h3>Prediction Result</h3>
                                        <p>The name <b>{input_name}</b> is predicted to be:</p>
                                        <p class="{gender_class}">{gender_text}</p>
                                        <p class="probability">Confidence: {probability:.2%}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Create visualization for probability
                                    fig, ax = plt.subplots(figsize=(10, 2))
                                    ax.barh(["Confidence"], [probability], color="#3498db" if gender == "M" else "#e74c3c")
                                    ax.barh(["Confidence"], [1-probability], left=[probability], color="#ecf0f1")
                                    ax.set_xlim(0, 1)
                                    ax.set_xlabel("Probability")
                                    ax.set_title(f"Prediction Confidence: {probability:.2%}")
                                    for s in ["top", "right"]:
                                        ax.spines[s].set_visible(False)
                                    st.pyplot(fig)
                else:
                    st.error("Please enter a name to predict.")
    
    # Tab 3: Naive Bayes
    with tab3:
        st.markdown('<h2 class="sub-header">Naive Bayes Prediction</h2>', unsafe_allow_html=True)
        
        # Model information
        nb_info = get_model_info('nb')
        st.markdown(f"""
        <div class="model-card">
            <h3>{nb_info['name']}</h3>
            <p>{nb_info['description']}</p>
            <h4>Advantages:</h4>
            {''.join([f'<p class="model-advantage">✓ {adv}</p>' for adv in nb_info['advantages']])}
            <h4>Disadvantages:</h4>
            {''.join([f'<p class="model-disadvantage">✗ {dis}</p>' for dis in nb_info['disadvantages']])}
        </div>
        """, unsafe_allow_html=True)
        
        # Check if model is trained
        model_exists = os.path.exists("models/nb_model") or st.session_state.models_trained
        
        if not model_exists:
            st.warning("Please train the models in the 'Data & Training' tab first!")
        else:
            # Load model if not in session state
            if not st.session_state.models_trained and 'nb_model' not in st.session_state:
                try:
                    st.session_state.nb_model = load_model("models/nb_model")
                    st.session_state.models_trained = True
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    st.warning("Please train the models in the 'Data & Training' tab first!")
            
            # Input for prediction
            input_name = st.text_input("Enter a name:", key="nb_input")
            
            if st.button("Predict Gender", key="nb_predict"):
                if input_name:
                    with st.spinner("Predicting..."):
                        # Create temporary DataFrame
                        input_df = spark.createDataFrame([(input_name,)], ["NOMPL"])
                        
                        # Clean and extract features
                        input_df = clean_names(input_df)
                        input_df = extract_name_features(input_df)
                        
                        # Make prediction
                        gender, probability = predict_gender(
                            st.session_state.nb_model, 
                            input_name, 
                            st.session_state.nb_label_indexer, 
                            spark
                        )
                        
                        # Display result with animation
                        with st.empty():
                            for i in range(5):
                                if i < 4:
                                    st.markdown(f"""
                                    <div class="result-container">
                                        <h3>Analyzing name: {input_name}</h3>
                                        <p>Processing{'.' * (i + 1)}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    time.sleep(0.3)
                                else:
                                    gender_class = "prediction-male" if gender == "M" else "prediction-female"
                                    gender_text = "Male" if gender == "M" else "Female"
                                    
                                    st.markdown(f"""
                                    <div class="result-container">
                                        <h3>Prediction Result</h3>
                                        <p>The name <b>{input_name}</b> is predicted to be:</p>
                                        <p class="{gender_class}">{gender_text}</p>
                                        <p class="probability">Confidence: {probability:.2%}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Create visualization for probability
                                    fig, ax = plt.subplots(figsize=(10, 2))
                                    ax.barh(["Confidence"], [probability], color="#3498db" if gender == "M" else "#e74c3c")
                                    ax.barh(["Confidence"], [1-probability], left=[probability], color="#ecf0f1")
                                    ax.set_xlim(0, 1)
                                    ax.set_xlabel("Probability")
                                    ax.set_title(f"Prediction Confidence: {probability:.2%}")
                                    for s in ["top", "right"]:
                                        ax.spines[s].set_visible(False)
                                    st.pyplot(fig)
                else:
                    st.error("Please enter a name to predict.")

if __name__ == "__main__":
    main()