from datetime import datetime

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMRegressor
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.house_price.config import ProjectConfig, Tags


class FeatureLookUpModel:
    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession):
        """
        Initialize the model with project configuration.
        """
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Define table names and function name
        self.feature_table_name = (
            f"{self.catalog_name}.{self.schema_name}.house_features"
        )
        self.function_name = (
            f"{self.catalog_name}.{self.schema_name}.calculate_house_age"
        )

        # MLflow configuration
        self.experiment_name = self.config.experiment_name_fe
        self.tags = tags.dict()

    def create_feature_table(self):
        """
        Create or replace the house_features table and populate it.
        """
        self.spark.sql(f"""
        CREATE OR REPLACE TABLE {self.feature_table_name}
        (Id STRING NOT NULL, OverallQual INT, GrLivArea INT, GarageCars INT);
        """)
        self.spark.sql(
            f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT house_pk PRIMARY KEY(Id);"
        )
        self.spark.sql(
            f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT Id, OverallQual, GrLivArea, GarageCars FROM {self.catalog_name}.{self.schema_name}.train_set"
        )
        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT Id, OverallQual, GrLivArea, GarageCars FROM {self.catalog_name}.{self.schema_name}.test_set"
        )
        logger.info("✅ Feature table created and populated.")

    def define_feature_function(self):
        """
        Define a function to calculate the house's age.
        """
        self.spark.sql(f"""
        CREATE OR REPLACE FUNCTION {self.function_name}(year_built INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        from datetime import datetime
        return datetime.now().year - year_built
        $$
        """)
        logger.info("✅ Feature function defined.")

    def load_data(self):
        """
        Load training and testing data from Delta tables.
        """
        self.train_set = self.spark.table(
            f"{self.catalog_name}.{self.schema_name}.train_set"
        ).drop("OverallQual", "GrLivArea", "GarageCars")
        self.test_set = self.spark.table(
            f"{self.catalog_name}.{self.schema_name}.test_set"
        ).toPandas()

        self.train_set = self.train_set.withColumn(
            "YearBuilt", self.train_set["YearBuilt"].cast("int")
        )
        self.train_set = self.train_set.withColumn(
            "Id", self.train_set["Id"].cast("string")
        )

        logger.info("✅ Data successfully loaded.")

    def feature_engineering(self):
        """
        Perform feature engineering by linking data with feature tables.
        """
        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=["OverallQual", "GrLivArea", "GarageCars"],
                    lookup_key="Id",
                ),
                FeatureFunction(
                    udf_name=self.function_name,
                    output_name="house_age",
                    input_bindings={"year_built": "YearBuilt"},
                ),
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        self.training_df = self.training_set.load_df().toPandas()
        current_year = datetime.now().year
        self.test_set["house_age"] = current_year - self.test_set["YearBuilt"]

        self.X_train = self.training_df[
            self.num_features + self.cat_features + ["house_age"]
        ]
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set[
            self.num_features + self.cat_features + ["house_age"]
        ]
        self.y_test = self.test_set[self.target]

        logger.info("✅ Feature engineering completed.")

    def train(self):
        """
        Train the model and log results to MLflow.
        """
        logger.info("🚀 Starting training...")

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)
            ],
            remainder="passthrough",
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", LGBMRegressor(**self.parameters)),
            ]
        )

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)

            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)

            logger.info(f"📊 Mean Squared Error: {mse}")
            logger.info(f"📊 Mean Absolute Error: {mae}")
            logger.info(f"📊 R2 Score: {r2}")

            mlflow.log_param("model_type", "LightGBM with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            signature = infer_signature(self.X_train, y_pred)

            self.fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="lightgbm-pipeline-model-fe",
                training_set=self.training_set,
                signature=signature,
            )

    def register_model(self):
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model-fe",
            name=f"{self.catalog_name}.{self.schema_name}.house_prices_model_fe",
            tags=self.tags,
        )

        # Fetch the latest version dynamically
        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.house_prices_model_fe",
            alias="latest-model",
            version=latest_version,
        )
