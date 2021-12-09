
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField}
import org.apache.spark.sql.{Row, SparkSession}

import scala.collection.immutable.ListMap
import scala.collection.mutable.ArrayBuffer
object csvtolibsvm {
  def main(args:Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("csvreader")
      .master("local")
      .getOrCreate()

    val sc = spark.sparkContext

    val prim = spark.read.format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat")
      .option("header","true")
      .option("inferSchema","true")
      .load("data/heart_disease_health_indicators_BRFSS2015.csv")

    prim.createOrReplaceTempView("resultsql")
    val result = spark.sql("select * from resultsql where CholCheck = 1")//
    //val result = spark.sql("select HeartDiseaseorAttack, HighBP, HighChol from resultsql where CholCheck =1")
    //For cholesterol data, we need to eliminate samples that have not been tested for cholesterol

    val colNames = result.columns

    val cols = colNames.map(f => col(f).cast(DoubleType))
    val resultTyped = result.select(cols:_*)
    val resultFinal = resultTyped.withColumn("HeartDiseaseorAttack",col("HeartDiseaseorAttack").cast(IntegerType))
    val fieldSeq: scala.collection.Seq[StructField] = resultFinal.schema.fields.toSeq.filter(f => f.dataType == DoubleType)
    val fieldNameSeq: Seq[String] = fieldSeq.map(f => f.name)
    val positionsArray: ArrayBuffer[LabeledPoint] = ArrayBuffer[LabeledPoint]()

    resultFinal.collect().foreach{
      row => positionsArray+=convertRowToLabeledPoint(row,fieldNameSeq,row.getAs("HeartDiseaseorAttack"));
    }
    val mRdd:RDD[LabeledPoint]= sc.parallelize(positionsArray)
    MLUtils.saveAsLibSVMFile(mRdd, "libsvm")

    spark.close()
  }

  @throws(classOf[Exception])
  private def convertRowToLabeledPoint(rowIn: Row, fieldNameSeq: Seq[String], label:Int): LabeledPoint =
  {
    try
    {
      val values: Map[String, Double] = rowIn.getValuesMap(fieldNameSeq)

      val sortedValuesMap = ListMap(values.toSeq.sortBy(_._1): _*)
      val rowValuesItr: Iterable[Double] = sortedValuesMap.values

      val positionsArray: ArrayBuffer[Int] = ArrayBuffer[Int]()
      val valuesArray: ArrayBuffer[Double] = ArrayBuffer[Double]()
      var currentPosition: Int = 0
      rowValuesItr.foreach
      {
        kv =>
          if (kv > 0)
          {
            valuesArray += kv
            positionsArray += currentPosition
          }
          currentPosition = currentPosition + 1;
      }
      val lp:LabeledPoint = new LabeledPoint(label,  org.apache.spark.mllib.linalg.Vectors.sparse(positionsArray.size,positionsArray.toArray, valuesArray.toArray))
      lp
    }
    catch
    {
      case ex: Exception =>
      throw new Exception(ex)
    }
  }
}
