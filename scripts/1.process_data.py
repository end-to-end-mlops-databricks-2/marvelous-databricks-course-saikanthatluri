import logging

import yaml
from pyspark.sql import SparkSession

from hotel_reservation.config import ProjectConfig
from hotel_reservation.data_processor import DataProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Reading Configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# creating spark session
spark = SparkSession.builder.getOrCreate()

# reading data from volumes
df = spark.read.csv(
    f"/Volumes/{config.catalog_name}/{config.schema_name}/data/Hotel Reservations.csv", header=True, inferSchema=True
).toPandas()

# Initialize DataProcessor
data_processor = DataProcessor(df, config, spark)

# Process the data
data_processor.data_process()

# splitting data into train and test
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)

# Enable change data feed (only once!)
logger.info("Enable change data feed")
data_processor.enable_change_data_feed()
