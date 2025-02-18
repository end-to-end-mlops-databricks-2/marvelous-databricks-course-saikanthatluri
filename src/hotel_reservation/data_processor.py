import datetime

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from hotel_reservation.config import ProjectConfig

class DataProcessor:
    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession):
        self.df = pandas_df # store the dataframe as self.df
        self.config = config # store the config
        self.spark = spark

    def data_process(self):
        """Preprocess the DataFrame stored in self.df"""
        # convert numerical columns in to log transformations
        num_features = self.config.num_features
        for col in num_features:
            try:
                self.df[col] = np.log(self.df[col] + 1)
            except Exception as e:
                # Handle potential errors (e.g., non-numeric data)
                print(f"Error transforming column '{col}': {e}. Skipping this column.")
                continue
        cat_features = self.config.cat_features
        target = self.config.target
        relevant_columns = num_features +  cat_features +[target]
        self.df = self.df[relevant_columns]

    def split_data(self, test_size=0.2, random_state=80):
        """Split the DataFrame (self.df) into training and test sets."""
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state= random_state)
        return train_set, test_set
    
    def save_to_catalog(self, train_set= pd.DataFrame, test_set= pd.DataFrame):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self):
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )