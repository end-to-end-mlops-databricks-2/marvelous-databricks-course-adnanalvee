import yaml
import logging
import pandas as pd
from src.house_price.data_processor import DataProcessor
from src.house_price.config import ProjectConfig
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config = ProjectConfig.from_yaml(config_path="project_config.yml")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
# Initialize DataProcessor


data = pd.read_csv("data/data.csv")  # ✅ Ensure it's a DataFrame
data_processor = DataProcessor(data, config)  # ✅ Pass DataFrame, not a string
data_processor.preprocess()

# Preprocess the data
data_processor.preprocess()

# COMMAND ----------
# Split the data
X_train, X_test = data_processor.split_data()

logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

logger.info("saving to catalog")
data_processor.save_to_catalog(X_train, X_test, spark)