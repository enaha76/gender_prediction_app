"""
Feature extraction module for Gender Prediction App
"""
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf, length
from pyspark.sql.types import FloatType, ArrayType, StringType, IntegerType
from pyspark.ml.feature import HashingTF, VectorAssembler

def extract_name_features(df_spark: DataFrame) -> DataFrame:
    """
    Extract features from names for gender prediction
    
    Args:
        df_spark: Spark DataFrame with preprocessed name columns
        
    Returns:
        DataFrame with extracted features
    """
    # Name length
    df_spark = df_spark.withColumn("name_length", length(col("first_name")).cast(FloatType()))
    
    # Last character
    last_char_udf = udf(lambda name: name[-1].lower() if name and len(name) > 0 else "", StringType())
    df_spark = df_spark.withColumn("last_char", last_char_udf(col("first_name")))
    
    # Last two characters
    last_two_udf = udf(lambda name: name[-2:].lower() if name and len(name) > 1 else "", StringType())
    df_spark = df_spark.withColumn("last_two_chars", last_two_udf(col("first_name")))
    
    # First character
    first_char_udf = udf(lambda name: name[0].lower() if name and len(name) > 0 else "", StringType())
    df_spark = df_spark.withColumn("first_char", first_char_udf(col("first_name")))
    
    # Vowel count
    vowel_count_udf = udf(lambda name: sum(1 for c in name.lower() if c in "aeiou") if name else 0, IntegerType())
    df_spark = df_spark.withColumn("vowel_count", vowel_count_udf(col("first_name")))
    
    # Consonant to vowel ratio
    consonant_vowel_ratio_udf = udf(
        lambda name: (
            (sum(1 for c in name.lower() if c not in "aeiou" and c.isalpha()) / 
            max(sum(1 for c in name.lower() if c in "aeiou"), 1))
            if name else 0.0
        ), 
        FloatType()
    )
    df_spark = df_spark.withColumn("cons_vowel_ratio", consonant_vowel_ratio_udf(col("first_name")))
    
    # Convert name to character array for TF features
    chars_udf = udf(lambda name: [c.lower() for c in name] if name else [], ArrayType(StringType()))
    df_spark = df_spark.withColumn("chars", chars_udf(col("first_name")))
    
    # Create TF features from characters
    hashingTF = HashingTF(inputCol="chars", outputCol="char_features", numFeatures=128)
    df_spark = hashingTF.transform(df_spark)
    
    # Create one-hot encoding for last character
    hashingTF_last = HashingTF(inputCol="last_char", outputCol="last_char_features", numFeatures=30)
    df_spark = hashingTF_last.transform(df_spark)
    
    # Combine all features
    assembler = VectorAssembler(
        inputCols=[
            "name_length", 
            "vowel_count", 
            "cons_vowel_ratio", 
            "char_features", 
            "last_char_features"
        ], 
        outputCol="features"
    )
    df_spark = assembler.transform(df_spark)
    
    return df_spark

def get_feature_names() -> list:
    """
    Get the names of features used in the model
    
    Returns:
        List of feature names
    """
    return [
        "Name length",
        "Vowel count",
        "Consonant to vowel ratio",
        "Character n-grams",
        "Last character"
    ]

def get_feature_descriptions() -> dict:
    """
    Get descriptions of features used in the model
    
    Returns:
        Dictionary of feature descriptions
    """
    return {
        "Name length": "Total number of characters in the name",
        "Vowel count": "Number of vowels in the name",
        "Consonant to vowel ratio": "Ratio of consonants to vowels",
        "Character n-grams": "Frequency of character patterns",
        "Last character": "The final character of the name"
    }