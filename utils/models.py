"""
Model training and prediction module for Gender Prediction App
"""
import os
from typing import Dict, Tuple, Any
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer
from utils.features import extract_name_features

def train_logistic_regression(df_train: DataFrame) -> Tuple[PipelineModel, float, StringIndexer]:
    """
    Train a Logistic Regression model for gender prediction
    
    Args:
        df_train: Spark DataFrame with extracted features
        
    Returns:
        Trained model, accuracy score, and label indexer
    """
    # Label indexing
    label_indexer = StringIndexer(inputCol="SEXE", outputCol="label")
    label_indexer_model = label_indexer.fit(df_train)
    indexed_data = label_indexer_model.transform(df_train)
    
    # Split data
    train_data, test_data = indexed_data.randomSplit([0.8, 0.2], seed=42)
    
    # Initialize model
    lr = LogisticRegression(
        maxIter=10, 
        regParam=0.1, 
        elasticNetParam=0.8,
        labelCol="label", 
        featuresCol="features", 
        probabilityCol="probability"
    )
    
    # Create pipeline
    lr_pipeline = Pipeline(stages=[lr])
    
    # Train model
    lr_model = lr_pipeline.fit(train_data)
    
    # Evaluate model
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", 
        predictionCol="prediction", 
        metricName="accuracy"
    )
    
    lr_predictions = lr_model.transform(test_data)
    lr_accuracy = evaluator.evaluate(lr_predictions)
    
    return lr_model, lr_accuracy, label_indexer_model

def train_naive_bayes(df_train: DataFrame) -> Tuple[PipelineModel, float, StringIndexer]:
    """
    Train a Naive Bayes model for gender prediction
    
    Args:
        df_train: Spark DataFrame with extracted features
        
    Returns:
        Trained model, accuracy score, and label indexer
    """
    # Label indexing
    label_indexer = StringIndexer(inputCol="SEXE", outputCol="label")
    label_indexer_model = label_indexer.fit(df_train)
    indexed_data = label_indexer_model.transform(df_train)
    
    # Split data
    train_data, test_data = indexed_data.randomSplit([0.8, 0.2], seed=42)
    
    # Initialize model
    nb = NaiveBayes(
        smoothing=1.0, 
        modelType="multinomial",
        labelCol="label", 
        featuresCol="features", 
        probabilityCol="probability"
    )
    
    # Create pipeline
    nb_pipeline = Pipeline(stages=[nb])
    
    # Train model
    nb_model = nb_pipeline.fit(train_data)
    
    # Evaluate model
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", 
        predictionCol="prediction", 
        metricName="accuracy"
    )
    
    nb_predictions = nb_model.transform(test_data)
    nb_accuracy = evaluator.evaluate(nb_predictions)
    
    return nb_model, nb_accuracy, label_indexer_model

def predict_gender(model: PipelineModel, name: str, label_indexer_model: StringIndexer, spark: SparkSession) -> Tuple[str, float]:
    """
    Predict gender for a given name using a trained model
    
    Args:
        model: Trained model
        name: Name to predict gender for
        label_indexer_model: Fitted label indexer
        spark: SparkSession
        
    Returns:
        Predicted gender and probability
    """
    # Create DataFrame with input name
    input_df = spark.createDataFrame([(name,)], ["NOMPL"])
    
    # Extract features
    input_df = extract_name_features(input_df)
    
    # Make prediction
    prediction = model.transform(input_df)
    
    # Get label mapping
    label_mapping = {
        float(idx): label 
        for idx, label in enumerate(label_indexer_model.labels)
    }
    
    # Extract prediction and probability
    pred_row = prediction.select("prediction", "probability").collect()[0]
    
    gender_idx = int(pred_row["prediction"])
    gender = label_mapping[float(gender_idx)]
    probability = float(pred_row["probability"][gender_idx])
    
    return gender, probability

def save_model(model: PipelineModel, path: str) -> None:
    """
    Save a trained model to disk
    
    Args:
        model: Trained model to save
        path: Path to save the model
    """
    model.write().overwrite().save(path)

def load_model(path: str) -> PipelineModel:
    """
    Load a trained model from disk
    
    Args:
        path: Path to the saved model
        
    Returns:
        Loaded model
    """
    return PipelineModel.load(path)

def get_model_info(model_type: str) -> Dict[str, str]:
    """
    Get information about a specific model type
    
    Args:
        model_type: Type of model ('lr' or 'nb')
        
    Returns:
        Dictionary with model information
    """
    model_info = {
        'lr': {
            'name': 'Logistic Regression',
            'description': 'A linear model that predicts binary outcomes based on a set of features.',
            'advantages': [
                'Works well with linearly separable data',
                'Provides probability estimates',
                'Less prone to overfitting with regularization'
            ],
            'disadvantages': [
                'Assumes linear relationship between features and output',
                'May underperform with highly correlated features'
            ]
        },
        'nb': {
            'name': 'Naive Bayes',
            'description': 'A probabilistic classifier based on Bayes\' theorem with independence assumptions.',
            'advantages': [
                'Works well with high-dimensional data',
                'Performs well even with small training sets',
                'Fast training and prediction'
            ],
            'disadvantages': [
                'Makes strong independence assumptions',
                'Sensitive to irrelevant features'
            ]
        }
    }
    
    return model_info.get(model_type, {})