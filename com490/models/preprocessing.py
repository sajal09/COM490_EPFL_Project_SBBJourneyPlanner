from pyspark.sql.functions import create_map, lit, coalesce
from models.constants import MODE_MAP
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from random import randrange
from typing import List
import sys

class Preprocessor:
    def __init__(self, username: str, hadoopFS: str):
        self.username = username
        self.hadoopFS = hadoopFS
        self.spark = (SparkSession\
            .builder
            .appName(username + '-final')
            .config('spark.ui.port', randrange(4050, 4450, 5))
            .config("spark.executorEnv.PYTHONPATH", ":".join(sys.path))
            .config('spark.jars',
                    f'{hadoopFS}/data/com-490/jars/iceberg-spark-runtime-3.5_2.13-1.6.1.jar,'
                    f'{hadoopFS}/data/com-490/jars/sedona-spark-shaded-3.5_2.13-1.7.1.jar,'
                    f'{hadoopFS}/data/com-490/jars/geotools-wrapper-1.7.1-28.5.jar'
            )
            .config('spark.sql.extensions', 'org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions')
            .config('spark.sql.catalog.iceberg', 'org.apache.iceberg.spark.SparkCatalog')
            .config('spark.sql.catalog.iceberg.type', 'hadoop')
            .config('spark.sql.catalog.iceberg.warehouse', f'{hadoopFS}/data/com-490/silver/')
            .config('spark.sql.catalog.spark_catalog', 'org.apache.iceberg.spark.SparkSessionCatalog')
            .config('spark.sql.catalog.spark_catalog.type', 'hadoop')
            .config('spark.sql.catalog.spark_catalog.warehouse', f'{self.hadoopFS}/user/{self.username}/final/warehouse')
            .config("spark.sql.warehouse.dir", f'{self.hadoopFS}/user/{self.username}/final/spark/warehouse')
            .config("spark.executor.memory", "6g")
            .config("spark.executor.cores", "4")
            .config("spark.executor.instances", "4")
        ).master('yarn').getOrCreate()

        self.preprocessed_istdaten_path = f'{self.hadoopFS}/user/{self.username}/final/istdaten.parquet'
        self.stops_operators_path = f'{self.hadoopFS}/user/{self.username}/final/stops_operators.parquet'
        self.operator_stops_path = f'{self.hadoopFS}/user/{self.username}/final/operator_stops.parquet'


    def clean_istdaten(self):
        mapping_expr = create_map([F.lit(x) for pair in MODE_MAP.items() for x in pair])

        cleaned_sbb_istdaten = (
            self.spark.table("iceberg.sbb.istdaten")
            .filter(F.year("operating_day") >= 2023)
            .withColumn("product_id_clean", F.upper(F.trim(F.col("product_id"))))
            .withColumn("transport_clean", F.upper(F.trim(F.col("transport"))))
            .withColumn("mode",coalesce(mapping_expr.getItem(F.col("product_id_clean")),mapping_expr.getItem(F.col("transport_clean")),lit("OTHER"))
            )
        )
        return cleaned_sbb_istdaten

    def preprocess(self, istdaten : SparkSession, operator_ids : List[str], stop_names : List[str], year_threshold: int = 2025, overwrite: bool = True):
        """Filter and Preprocess istdaten for relevant operators, stops, and time period."""

        latest_pub_date = (self.spark.table("iceberg.sbb.stops").agg(F.max("pub_date")).collect()[0][0])
        last_available_date = (istdaten.agg(F.max("operating_day")).collect()[0][0])

        stop_ids = (
            self.spark.table("iceberg.sbb.stops")
            .filter(F.col("pub_date") == latest_pub_date)
            .filter(F.col("stop_name").isin(stop_names))
            .withColumn("bpuic", F.col("stop_id").cast("integer"))  # "85XXXXX" -> integer
            .select("bpuic")
            .distinct()
        )

        stops = (
            istdaten
            .filter(F.col("operating_day") == last_available_date)
            .join(stop_ids, on="bpuic", how="inner")
            .filter(F.col("arr_actual").isNotNull())
            .filter(F.upper(F.trim(F.col("arr_status"))).isin(["REAL", "GESCHAETZT"]))
            .select("operator_id", "operator_abrv", "operator_name")
            .distinct()
        )

        stops_operator_ids = [row.operator_id for row in stops.collect()]

        operator_stops = (
            istdaten
            .filter(F.col("operator_id").isin(operator_ids))
            .select("bpuic")
            .distinct()
        )

        (
            istdaten
            .filter(F.col("unplanned") == False)
            .filter(F.year("operating_day") >= year_threshold)
            .filter(F.col("arr_actual").isNotNull())
            .filter(F.col("arr_status").isNotNull())
            .filter(F.upper(F.trim(F.col("arr_status"))).isin(["REAL", "GESCHAETZT"]))
            .filter(F.col("operator_id").isin(stops_operator_ids))
            .join(operator_stops, on="bpuic", how="inner")
            .write.parquet(
                self.preprocessed_istdaten_path,
                mode='overwrite' if overwrite else 'append'
            )
        )

        self.spark.createDataFrame(
            stops.collect()
        ).write.parquet(
            self.stops_operators_path,
            mode='overwrite'
        )

        operator_stops.write.parquet(
            self.operator_stops_path,
            mode='overwrite' if overwrite else 'append'
        )

    def get_processed_data(self):
        istdaten = self.spark.read.parquet(self.preprocessed_istdaten_path)
        stops_operators = self.spark.read.parquet(self.stops_operators_path)
        operator_stops = self.spark.read.parquet(self.operator_stops_path)
        return istdaten, stops_operators, operator_stops


