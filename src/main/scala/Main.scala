import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.{Dataset, DataFrame, SaveMode}
import org.apache.spark.sql.functions._

// install spark locally and add to path: export PATH=$PATH:/usr/local/spark/bin
// spark-submit --class "Main" target/scala-2.12/spark-nlp-starter-assembly-5.2.0.jar

// train/test CSV headers: id,keyword,location,text,target
final case class News(
  category: String,
  description: String,
)

object Main {

  def main(args: Array[String]): Unit = {
    implicit val spark = SparkSession.builder()
      .appName("spark-nlp-starter")
      .master("local[*]")
      .getOrCreate()
    
    import spark.implicits._
    spark.sparkContext.setLogLevel("ERROR")
    // Assuming you have a dataset of Tweets

    val outputPath = "/home/daniel/repos/spark-nlp-starter/src/main/resources/output"

    val train_ds: Dataset[News] = loadData("/home/daniel/repos/spark-nlp-starter/src/main/resources/data/news_category_train.csv")
    train_ds.show(10, truncate=50)
    
    println("Dataset loaded")
    // Define the stages of the Pipeline
    val documentAssembler = new DocumentAssembler()
      .setInputCol("description")
      .setOutputCol("document")
    
    val bertSentenceEmbeddings = UniversalSentenceEncoder
      .pretrained()
      .setInputCols(Array("document"))
      .setOutputCol("sentence_embeddings")
  
    // Initialize ClassifierDLApproach
    val classifierDL = new ClassifierDLApproach()
      .setInputCols("sentence_embeddings")
      .setOutputCol("class")
      .setLabelColumn("category")
      .setMaxEpochs(5) // Set the number of epochs
      .setEnableOutputLogs(true)

    // Pipeline with ClassifierDL
    val pipeline = new Pipeline().setStages(Array(documentAssembler, bertSentenceEmbeddings, classifierDL))
    println("Created Classifier pipeline")
    // Fit the model
    val classifierModel = pipeline.fit(train_ds)
    println("Fit Model")
    //predictions:
    val test_ds = loadData("/home/daniel/repos/spark-nlp-starter/src/main/resources/data/news_category_test.csv")
    println("Test Data Cleaned")
    
    println("Test Data Embedded")
    val predictions = classifierModel.transform(test_ds)
    predictions.show()
  }

  def loadData(inputFile: String)(implicit spark: SparkSession):Dataset[News] = {
    import spark.implicits._
    spark
      .read
      .format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ",") // Specify delimiter, default is ","
      .option("quote", "\"") // Default quote character
      .option("escape", "\"") // Escape character to handle quotes within quoted strings
      .option("multiline", "true") // Handle multi-line records
      .load(inputFile)
      .as[News]
  }

}
