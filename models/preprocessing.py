import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, regexp_replace, lower, when
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, HashingTF, IDF, Tokenizer, StopWordsRemover
from pyspark.sql.types import StructType, StructField, StringType
import re

def initialize_spark():
    """Initialize and return a Spark session."""
    return SparkSession.builder \
        .appName("GenderPrediction") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.sql.session.timeZone", "UTC") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()

def load_data(spark, file_path):
    """Load the CSV data and split the columns properly."""
    # Read the entire file as a single column
    raw_data = spark.read.text(file_path)
    
    # Check the first few lines to determine if this is the header row
    first_row = raw_data.first()[0]
    if "NOMPL;SEXE" in first_row:
        # Skip header row if it exists
        raw_data = raw_data.filter(~col("value").contains("NOMPL;SEXE"))
    
    # Split the single column into name and gender
    data = raw_data.withColumn("tmp", split(col("value"), ";"))
    data = data.withColumn("name", col("tmp").getItem(0))
    data = data.withColumn("gender", col("tmp").getItem(1))
    
    # Drop intermediate columns
    data = data.drop("value", "tmp")
    
    # Clean and validate the data
    data = data.filter(col("gender").isin("M", "F"))
    data = data.filter(col("name").isNotNull())
    
    return data

def extract_features(data):
    """Extract features from names for machine learning."""
    # Preprocessing: Clean names and convert to lowercase
    data = data.withColumn("clean_name", 
                          regexp_replace(lower(col("name")), "[^a-z ]", ""))
    
    # Tokenize the name into individual parts
    tokenizer = Tokenizer(inputCol="clean_name", outputCol="name_tokens")
    data = tokenizer.transform(data)
    
    # Remove any stop words
    remover = StopWordsRemover(inputCol="name_tokens", outputCol="filtered_tokens")
    data = remover.transform(data)
    
    # Use TF-IDF to convert names to feature vectors
    hashingTF = HashingTF(inputCol="filtered_tokens", outputCol="raw_features", numFeatures=20)
    data = hashingTF.transform(data)
    
    idf = IDF(inputCol="raw_features", outputCol="features")
    idfModel = idf.fit(data)
    data = idfModel.transform(data)
    
    # Create a label index for the gender
    labelIndexer = StringIndexer(inputCol="gender", outputCol="label")
    data = labelIndexer.fit(data).transform(data)
    
    return data

def prepare_input_name(spark, name):
    """Prepare a single input name for prediction."""
    # Create a DataFrame with the input name
    input_data = spark.createDataFrame([(name,)], ["name"])
    
    # Apply the same preprocessing
    input_data = input_data.withColumn("clean_name", 
                                      regexp_replace(lower(col("name")), "[^a-z ]", ""))
    
    # Tokenize
    tokenizer = Tokenizer(inputCol="clean_name", outputCol="name_tokens")
    input_data = tokenizer.transform(input_data)
    
    # Remove stop words
    remover = StopWordsRemover(inputCol="name_tokens", outputCol="filtered_tokens")
    input_data = remover.transform(input_data)
    
    # TF-IDF features
    hashingTF = HashingTF(inputCol="filtered_tokens", outputCol="raw_features", numFeatures=20)
    input_data = hashingTF.transform(input_data)
    
    # Since we don't have an IDF model, we'll just use raw TF features
    input_data = input_data.withColumnRenamed("raw_features", "features")
    
    return input_data

def extract_first_name(full_name):
    """Extract the first name from a full name."""
    name_parts = full_name.strip().split()
    if name_parts:
        return name_parts[0]
    return full_name