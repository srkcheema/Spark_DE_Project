# Python code for data ingestion with Spark
from pyspark.sql import SparkSession

# Ignite the SparkSession
spark = SparkSession.builder \
 .appName("Data Ingestion") \
 .getOrCreate()

# Ingest data from different sources
transaction_logs = spark.read.format("csv").load("transaction_logs.csv")
user_activity_logs = spark.read.format("json").load("user_activity_logs.json")
customer_demographics = spark.read.format("parquet").load("customer_demographics.parquet")

# Python code for data exploration and cleaning with Spark
from pyspark.sql.functions import col, count, isnan

# Embark on the journey of data exploration
transaction_logs.show(10)
user_activity_logs.printSchema()
customer_demographics.describe().show()

# Handle the emotions of missing values
transaction_logs = transaction_logs.dropna()
user_activity_logs = user_activity_logs.fillna(0)
customer_demographics = customer_demographics.na.drop()

# Let go of the burden of duplicates
transaction_logs = transaction_logs.dropDuplicates()
user_activity_logs = user_activity_logs.dropDuplicates()
customer_demographics = customer_demographics.dropDuplicates()

# Python code for data transformation with Spark
from pyspark.sql.functions import month, sum

# Embark on the journey of data transformation
monthly_sales = transaction_logs.groupBy(month("timestamp").alias("month")).agg(sum("amount").alias("total_sales"))
customer_activity = user_activity_logs.join(customer_demographics, "user_id")
customer_segments = customer_activity.groupBy("segment").count()

# Python code for advanced analytics and machine learning with Spark
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

# Prepare data for clustering
assembler = VectorAssembler(inputCols=["age", "income"], outputCol="features")
customer_data = assembler.transform(customer_demographics)

# Embark on the journey of advanced analytics
kmeans = KMeans(k=3, seed=42)
model = kmeans.fit(customer_data)
clustered_data = model.transform(customer_data)

# Python code for data visualization with Spark
import matplotlib.pyplot as plt

# Paint a masterpiece of insights
sales_data = monthly_sales.toPandas()
plt.plot(sales_data["month"], sales_data["total_sales"])
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.title("Monthly Sales Trend")
plt.show()