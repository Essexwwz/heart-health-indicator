import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.FeatureHasher
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{Row, SparkSession}

object lrDiseasesandBMI {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("csvreader")
      .master("local")
      .getOrCreate()
    val prim = spark.read.format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat")
      .option("header","true")
      .option("inferSchema","true")
      .load("data/heart_disease_health_indicators_BRFSS2015.csv")

    prim.createOrReplaceTempView("resultsql")
    //val result = spark.sql("select * from resultsql where CholCheck = 1")//
    val data = spark.sql("select HeartDiseaseorAttack,BMI,Smoker,Stroke,Diabetes from resultsql where CholCheck =1")
    val splited = data.randomSplit(Array(0.6,0.4),11L)
    val train_index = splited(0)
    val test_index = splited(1)

    //feature hasher
    val hasher = new FeatureHasher()
      .setInputCols("BMI","Smoker","Stroke","Diabetes")
      .setOutputCol("feature")
    val train_hs = hasher.transform(train_index)
    val test_hs = hasher.transform(test_index)

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0)
      .setFeaturesCol("feature")
      .setLabelCol("HeartDiseaseorAttack")
      .setPredictionCol("HeartDiseaseorAttack_predict")
    val model_lr = lr.fit(train_hs)
    println(s"(Diabetes,BMI,Smoker,Stroke): ${model_lr.coefficients} intercept: ${model_lr.intercept}")
    val predictions = model_lr.transform(test_hs)
    val prdd =  predictions.select("HeartDiseaseorAttack","HeartDiseaseorAttack_predict").rdd.map{
      case Row(predict:Double,attack:Double)=>(predict,attack)
    }
    val metrics = new MulticlassMetrics(prdd)
    val accuracy = metrics.accuracy
    val weightedPrecision = metrics.weightedPrecision
    val weightedRecall = metrics.weightedRecall
    val f1 = metrics.weightedFMeasure
    println(s"LR评估结果：\n分类正确率：$accuracy\n加权正确率：$weightedPrecision\n加权召回率：$weightedRecall\nF1值：$f1")


  }
}
