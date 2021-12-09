import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession

object logicalDiseases {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("csvreader")
      .master("local")
      .getOrCreate()
    val sc = spark.sparkContext
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
}

