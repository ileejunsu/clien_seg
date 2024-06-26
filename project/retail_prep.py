import os
import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import datediff,to_date,to_timestamp, col, max, count, sum, avg, stddev, round, abs, current_date, date_diff, lit, min
from pyspark.ml.feature import VectorAssembler

# Spark configuration
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ['HADOOP_HOME'] = r'C:\hadoop'  # Replace with your Hadoop directory
spark = SparkSession.builder \
    .appName("CustomerSegmentation") \
    .config("spark.ui.showConsoleProgress", "false") \
    .config("spark.log.level", "ERROR") \
    .config("hadoop.home.dir", r"C:\hadoop").getOrCreate()

# Load data
file_path = "online_retail_II.csv"  # Update with your actual file path
if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
    print("Please download the dataset and place it in the project root directory.")
    print("See README.md for instructions on obtaining the dataset.")
    exit(1)
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Data Preprocessing
df = df.dropna(subset=["Customer ID", "InvoiceDate"])
df = df.filter((col("Quantity").isNotNull()) & (col("Quantity") > 0) & (col("Price").isNotNull()) & (col("Price") > 0))
df = df.withColumn("Quantity", col("Quantity").cast("int"))
df = df.withColumn("Price", col("Price").cast("float"))
df = df.withColumn("InvoiceDate", to_timestamp(col("InvoiceDate"), "M/d/y H:mm"))
df = df.withColumn("TotalPrice", col("Quantity") * col("Price"))
df = df.withColumn("InvoiceDate", to_date(col("InvoiceDate")))

# Calculate RFM Features
reference_date = df.agg(max("InvoiceDate")).head()[0]
reference_date_col = lit(reference_date)  

df_features = df.groupBy("Customer ID").agg(
    sum("Quantity").alias("TotalQuantity"),
    sum("TotalPrice").alias("TotalSpent"),
    avg("TotalPrice").alias("AvgSpent"),
    count("Invoice").alias("NumPurchases"),
    max("InvoiceDate").alias("LastPurchasesDate"),
    min("InvoiceDate").alias("FirstPurchasesDate")
)

df_features = df_features.withColumn("Recency", datediff(reference_date_col, col("LastPurchasesDate")))
df_features = df_features.withColumn("DaysSinceFirstPurchase", datediff(reference_date_col, col("FirstPurchasesDate")))
df_features = df_features.withColumn("PurchaseFrequency", col("DaysSinceFirstPurchase") / col("NumPurchases"))
df_features = df_features.withColumn("AvgBasketSize", col("TotalQuantity") / col("NumPurchases"))


# Assemble features into a single vector
assembler = VectorAssembler(inputCols=[
    "TotalQuantity", "TotalSpent", "AvgSpent", "NumPurchases", "Recency",
    "DaysSinceFirstPurchase", "PurchaseFrequency", "AvgBasketSize"
], outputCol="features")
df_features = assembler.transform(df_features)


# Save df_features as Parquet (Overwrite if it exists)
df_features.write.mode('overwrite').parquet("df_features.parquet")