import mlflow
from pyspark.sql import SparkSession

from src.house_price.config import ProjectConfig, Tags
from src.models.feature_lookup_model import FeatureLookUpModel

mlflow.set_registry_uri("databricks-uc")

config = ProjectConfig.from_yaml(config_path="project_config.yml")

spark = SparkSession.builder.getOrCreate()

tags_dict = {"git_sha": "abcd12345", "branch": "week2"}
tags = Tags(**tags_dict)

fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# Create feeature table
fe_model.create_feature_table()

# define house age feature function
fe_model.define_feature_function()

# load data
fe_model.load_data()

# perform feature engineering
fe_model.feature_engineering()

# train the model
fe_model.train()

# register the model
fe_model.register_model()