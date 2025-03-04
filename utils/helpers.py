import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def get_data_summary(spark_df):
    """Get summary statistics from a Spark DataFrame."""
    # Convert to Pandas for easier visualization
    pandas_df = spark_df.toPandas()
    
    # Gender distribution
    gender_counts = pandas_df["gender"].value_counts()
    
    # Name length statistics
    pandas_df["name_length"] = pandas_df["name"].apply(lambda x: len(x))
    
    return {
        "total_names": len(pandas_df),
        "male_count": gender_counts.get("M", 0),
        "female_count": gender_counts.get("F", 0),
        "avg_name_length": pandas_df["name_length"].mean(),
        "min_name_length": pandas_df["name_length"].min(),
        "max_name_length": pandas_df["name_length"].max(),
        "pandas_df": pandas_df
    }

def create_gender_distribution_chart(data_summary):
    """Create a chart showing gender distribution."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Gender distribution
    gender_data = [data_summary["male_count"], data_summary["female_count"]]
    labels = ["Male", "Female"]
    colors = ["#3498db", "#e74c3c"]
    
    ax.pie(gender_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    plt.title("Gender Distribution in Dataset")
    return fig

def create_name_length_chart(pandas_df):
    """Create a chart showing name length distribution by gender."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Name length distribution by gender
    sns.histplot(data=pandas_df, x="name_length", hue="gender", multiple="stack", ax=ax)
    
    plt.title("Name Length Distribution by Gender")
    plt.xlabel("Name Length (characters)")
    plt.ylabel("Count")
    
    return fig

def show_prediction_results(prediction_result, model_type):
    """Display prediction results in a formatted card."""
    gender = prediction_result["predicted_gender"]
    confidence = prediction_result["confidence"] * 100
    
    # Determine color based on gender prediction
    gender_color = "#e74c3c" if gender == "F" else "#3498db"
    gender_text = "Female" if gender == "F" else "Male"
    
    # Create a styled container for the results
    st.markdown(f"""
    <div style="padding: 20px; border-radius: 10px; background-color: #f8f9fa; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 20px 0;">
        <h3 style="color: #2c3e50; margin-bottom: 15px;">Prediction Results ({model_type})</h3>
        <p style="font-size: 18px;">Name: <strong>{prediction_result["name"]}</strong></p>
        <p style="font-size: 18px;">Predicted Gender: <span style="color: {gender_color}; font-weight: bold;">{gender_text}</span></p>
        <div style="background-color: #ecf0f1; border-radius: 5px; height: 30px; width: 100%; margin: 15px 0;">
            <div style="background-color: {gender_color}; width: {confidence}%; height: 100%; border-radius: 5px; display: flex; align-items: center; justify-content: center;">
                <span style="color: white; font-weight: bold;">{confidence:.1f}%</span>
            </div>
        </div>
        <p style="font-size: 14px; color: #7f8c8d;">Model confidence in prediction</p>
    </div>
    """, unsafe_allow_html=True)

def compare_prediction_results(lr_prediction, nb_prediction):
    """Compare predictions from both models."""
    lr_gender = "Female" if lr_prediction["predicted_gender"] == "F" else "Male"
    nb_gender = "Female" if nb_prediction["predicted_gender"] == "F" else "Male"
    
    lr_confidence = lr_prediction["confidence"] * 100
    nb_confidence = nb_prediction["confidence"] * 100
    
    # Create a comparison table
    st.markdown(f"""
    <div style="padding: 20px; border-radius: 10px; background-color: #f8f9fa; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 20px 0;">
        <h3 style="color: #2c3e50; margin-bottom: 15px;">Model Comparison</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background-color: #ecf0f1;">
                <th style="padding: 10px; text-align: left; border: 1px solid #dee2e6;">Model</th>
                <th style="padding: 10px; text-align: left; border: 1px solid #dee2e6;">Predicted Gender</th>
                <th style="padding: 10px; text-align: left; border: 1px solid #dee2e6;">Confidence</th>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #dee2e6;">Logistic Regression</td>
                <td style="padding: 10px; border: 1px solid #dee2e6; color: {('#e74c3c' if lr_gender == 'Female' else '#3498db')}; font-weight: bold;">{lr_gender}</td>
                <td style="padding: 10px; border: 1px solid #dee2e6;">{lr_confidence:.1f}%</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #dee2e6;">Naive Bayes</td>
                <td style="padding: 10px; border: 1px solid #dee2e6; color: {('#e74c3c' if nb_gender == 'Female' else '#3498db')}; font-weight: bold;">{nb_gender}</td>
                <td style="padding: 10px; border: 1px solid #dee2e6;">{nb_confidence:.1f}%</td>
            </tr>
        </table>
        <div style="margin-top: 15px;">
            <p style="font-size: 14px; color: #7f8c8d;">
                {
                    "✓ Both models agree on the gender prediction." 
                    if lr_gender == nb_gender else 
                    "⚠️ Models disagree on the gender prediction. Consider the model with higher confidence."
                }
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)