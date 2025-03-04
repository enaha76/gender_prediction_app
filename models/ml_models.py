from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import os
import joblib

def train_logistic_regression(train_data):
    """Train a Logistic Regression model for gender prediction."""
    # Set up the logistic regression model
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10, regParam=0.1)
    
    # Train the model
    lr_model = lr.fit(train_data)
    
    return lr_model

def train_naive_bayes(train_data):
    """Train a Naive Bayes model for gender prediction."""
    # Set up the Naive Bayes model
    nb = NaiveBayes(featuresCol="features", labelCol="label", smoothing=1.0)
    
    # Train the model
    nb_model = nb.fit(train_data)
    
    return nb_model

def predict_gender(model, input_data):
    """Use a trained model to predict gender for input data."""
    # Make predictions
    predictions = model.transform(input_data)
    
    # Extract prediction and probability
    result = predictions.select("name", "prediction", "probability").collect()[0]
    
    # Convert prediction (0 or 1) to M or F
    gender = "F" if result.prediction == 0 else "M"
    
    # Calculate confidence score (probability of the predicted class)
    confidence = result.probability[int(result.prediction)]
    
    return {
        "name": result.name,
        "predicted_gender": gender,
        "confidence": float(confidence)
    }

def evaluate_model(model, test_data):
    """Evaluate the performance of a model."""
    # Make predictions on test data
    predictions = model.transform(test_data)
    
    # Evaluate the model
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    
    # Calculate metrics
    evaluator.setMetricName("weightedPrecision")
    precision = evaluator.evaluate(predictions)
    
    evaluator.setMetricName("weightedRecall")
    recall = evaluator.evaluate(predictions)
    
    evaluator.setMetricName("f1")
    f1 = evaluator.evaluate(predictions)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def save_model(model, model_path):
    """Save a trained model to disk."""
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    model.save(model_path)
    
def load_model(spark, model_type, model_path):
    """Load a trained model from disk."""
    if model_type == "logistic_regression":
        from pyspark.ml.classification import LogisticRegressionModel
        return LogisticRegressionModel.load(model_path)
    elif model_type == "naive_bayes":
        from pyspark.ml.classification import NaiveBayesModel
        return NaiveBayesModel.load(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")