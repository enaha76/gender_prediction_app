"""
Preprocessing module for Gender Prediction App
"""
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, split, trim, lower

def clean_names(df_spark: DataFrame) -> DataFrame:
    """
    Clean and preprocess name data
    
    Args:
        df_spark: Spark DataFrame with NOMPL column
        
    Returns:
        DataFrame with cleaned names
    """
    # Trim whitespace
    df_spark = df_spark.withColumn("NOMPL", trim(col("NOMPL")))
    
    # Lowercase names for consistency
    df_spark = df_spark.withColumn("NOMPL_lower", lower(col("NOMPL")))
    
    # Extract first name (first word in the name)
    df_spark = df_spark.withColumn("first_name", 
                                 split(col("NOMPL"), " ").getItem(0))
    
    # Extract first name in lowercase for processing
    df_spark = df_spark.withColumn("first_name_lower", 
                                 lower(split(col("NOMPL"), " ").getItem(0)))
    
    return df_spark

def handle_missing_values(df_spark: DataFrame) -> DataFrame:
    """
    Handle missing values in the dataset
    
    Args:
        df_spark: Spark DataFrame
        
    Returns:
        DataFrame with handled missing values
    """
    # Count nulls
    null_counts = [(c, df_spark.filter(col(c).isNull()).count()) for c in df_spark.columns]
    
    # Drop rows with missing names or gender
    df_clean = df_spark.dropna(subset=["NOMPL", "SEXE"])
    
    return df_clean

def validate_data(df_spark: DataFrame) -> bool:
    """
    Validate that the dataframe has the required columns and format
    
    Args:
        df_spark: Spark DataFrame
        
    Returns:
        Boolean indicating if data is valid
    """
    required_columns = ["NOMPL", "SEXE"]
    
    # Check if required columns exist
    has_required_cols = all(col in df_spark.columns for col in required_columns)
    
    # Check if gender values are valid (M or F)
    if has_required_cols:
        valid_genders = df_spark.filter(~col("SEXE").isin(["M", "F"])).count() == 0
        return valid_genders
    
    return False