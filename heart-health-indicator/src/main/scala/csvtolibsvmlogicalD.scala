
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField}
import org.apache.spark.sql.{Row, SparkSession}

import scala.collection.immutable.ListMap
import scala.collection.mutable.ArrayBuffer

object csvtolibsvmlogicalD {
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

    val result = spark.sql("select HeartDiseaseorAttack, Stroke,Diabetes from resultsql where CholCheck =1")
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
    MLUtils.saveAsLibSVMFile(mRdd, "libsvmDiseases")

    val data = MLUtils.loadLibSVMFile(sc, "libsvmDiseases/part-00000")
    val splits = data.randomSplit(Array(0.3, 0.7), seed = 2L)
    val training = splits(1)
    val test = splits(0)
    // Run training algorithm to build the model
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(training)

    // Compute raw scores on the test set.
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    // Get evaluation metrics.
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val accuracy = metrics.accuracy
    println(s"Accuracy = $accuracy")
    // Save and load model
    model.save(sc,"logicRegressionmodelD")
    println(model.weights)
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
