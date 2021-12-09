import org.apache.spark.sql.SparkSession

object modelshow {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("csvreader")
      .master("local")
      .getOrCreate()
    val model = spark.read.format("parquet").load("data/lrmodelD.snappy.parquet")
    model.show(false)
  }
}
