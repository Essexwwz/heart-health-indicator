import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row.empty.schema
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{DataTypes, DoubleType, IntegerType, StructField}
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}

import scala.collection.immutable.ListMap
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
object corr {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("csvreader")
      .master("local")
      .getOrCreate()

    import spark.implicits._
    val sc = spark.sparkContext

    val prim = spark.read.format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("data/heart_disease_health_indicators_BRFSS2015.csv")


    prim.createOrReplaceTempView("resultsql")
    val result = spark.sql("select * from resultsql where CholCheck = 1")

    val ans1 = result.select("HeartDiseaseorAttack", "HighBP").rdd.map(x => (x(0).toString.toDouble, x(1).toString.toDouble))
    //For cholesterol data, we need to eliminate samples that have not been tested for cholesterol

    val Chol = result.select("HeartDiseaseorAttack", "HighChol").rdd.map(x => x(1).toString.toDouble)

    val Attack: RDD[Double] = ans1.map(x => x._1)
    val Bp: RDD[Double] = ans1.map(x => x._2)

    val c1 = Statistics.corr(Bp, Attack, "spearman")
    val c2 = Statistics.corr(Chol, Attack, "spearman")
    println("Spearman coefficient HighBP " + c1)
    println("Spearman coefficient HignChol" + c2)
    println("It can be seen that heart disease is slightly positively correlated with hypertension and high cholesterol")
    println("***********************************************")
    val Smoker = result.select("HeartDiseaseorAttack", "Smoker").rdd.map(x => x(1).toString.toDouble)
    val BMI = result.select("HeartDiseaseorAttack", "BMI").rdd.map(x => x(1).toString.toDouble)
    val Diabetes =result.select("HeartDiseaseorAttack", "Diabetes").rdd.map(x => x(1).toString.toDouble)
    val Phys = result.select("HeartDiseaseorAttack", "PhysActivity").rdd.map(x => x(1).toString.toDouble)
    val c3 = Statistics.corr(BMI,Attack,"spearman")
    val c4 = Statistics.corr(Smoker,Attack,"spearman")
    val c5 = Statistics.corr(Diabetes,Attack,"spearman")
    val c6 = Statistics.corr(Phys,Attack,"spearman")
    val c7 = Statistics.corr(result.select("HeartDiseaseorAttack", "Fruits").rdd.map(x => x(1).toString.toDouble),Attack,"spearman")
    val c8 = Statistics.corr(result.select("HeartDiseaseorAttack", "Veggies").rdd.map(x => x(1).toString.toDouble),Attack,"spearman")
    val c9= Statistics.corr(result.select("HeartDiseaseorAttack", "HvyAlcoholConsump").rdd.map(x => x(1).toString.toDouble),Attack,"spearman")
    val c10 = Statistics.corr(result.select("HeartDiseaseorAttack", "AnyHealthcare").rdd.map(x => x(1).toString.toDouble),Attack,"spearman")
    val c11 = Statistics.corr(result.select("HeartDiseaseorAttack", "NoDocbcCost").rdd.map(x => x(1).toString.toDouble),Attack,"spearman")
    val c12 = Statistics.corr(result.select("HeartDiseaseorAttack", "GenHlth").rdd.map(x => x(1).toString.toDouble),Attack,"spearman")
    val c13 = Statistics.corr(result.select("HeartDiseaseorAttack", "MentHlth").rdd.map(x => x(1).toString.toDouble),Attack,"spearman")
    val c14 = Statistics.corr(result.select("HeartDiseaseorAttack", "PhysHlth").rdd.map(x => x(1).toString.toDouble),Attack,"spearman")
    val c15 = Statistics.corr(result.select("HeartDiseaseorAttack", "DiffWalk").rdd.map(x => x(1).toString.toDouble),Attack,"spearman")
    val c16 = Statistics.corr(result.select("HeartDiseaseorAttack", "Sex").rdd.map(x => x(1).toString.toDouble),Attack,"spearman")
    val c17 = Statistics.corr(result.select("HeartDiseaseorAttack", "Age").rdd.map(x => x(1).toString.toDouble),Attack,"spearman")
    val c18 = Statistics.corr(result.select("HeartDiseaseorAttack", "Education").rdd.map(x => x(1).toString.toDouble),Attack,"spearman")
    val c19 = Statistics.corr(result.select("HeartDiseaseorAttack", "Income").rdd.map(x => x(1).toString.toDouble),Attack,"spearman")
    val c20 = Statistics.corr(result.select("HeartDiseaseorAttack", "Stroke").rdd.map(x => x(1).toString.toDouble),Attack,"spearman")
    println("Spearman coefficient BMI " + c3)
    println("Spearman coefficient Smoker " + c4)
    println("Spearman coefficient Stroke " + c20)
    println("Spearman coefficient Diabetes " + c5)
    println("***********************************************")
    println("Spearman coefficient PhysActivity " + c6)
    println("Spearman coefficient Fruits " + c7)
    println("Spearman coefficient Veggies " + c8)
    println("Spearman coefficient HvyAlcoholConsump " + c9)
    println("***********************************************")
    println("Spearman coefficient AnyHealthCare " + c10)
    println("Spearman coefficient NoDocbcCost " + c11)
    println("Spearman coefficient GenHealth " + c12)
    println("Spearman coefficient MentHlth " + c13)
    println("Spearman coefficient PhysHlth " + c14)
    println("Spearman coefficient Diffwalk " + c15)
    println("***********************************************")
    println("Spearman coefficient Sex " + c16)
    println("Spearman coefficient Age " + c17)
    println("Spearman coefficient Education " + c18)
    println("Spearman coefficient Income " + c19)
    println("***********************************************")
    spark.close()
  }
}