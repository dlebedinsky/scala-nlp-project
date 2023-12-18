import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.{Dataset, DataFrame, SaveMode}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Encoders

// train/test CSV headers: id,keyword,location,text,target
final case class Tweet(
  id: String,
  keyword: String,
  location: String,
  text: String,
  target: String
)

final case class TestTweet(
  id: String,
  keyword: String,
  location: String,
  text: String,
)

object Main {

  def main(args: Array[String]): Unit = {
    implicit val spark = SparkSession.builder()
      .appName("spark-nlp-starter")
      .master("local[*]")
      .getOrCreate()
    
    import spark.implicits._
    spark.sparkContext.setLogLevel("ERROR")

    val train_ds: Dataset[Tweet] = loadData("/home/daniel/repos/scala-nlp-project/src/main/resources/data/train.csv")
    val train_clean = cleanData(train_ds)
    train_clean.show(10, truncate=50)
    
    println("Dataset loaded")
    // Define the stages of the Pipeline
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
    
    val bertSentenceEmbeddings = BertSentenceEmbeddings
      .pretrained("sent_small_bert_L8_512")
      .setInputCols(Array("document"))
      .setOutputCol("sentence_embeddings")
  
    // A high-level tool for deep learning models
    val classifierDL = new ClassifierDLApproach()
      .setInputCols(Array("sentence_embeddings"))
      .setOutputCol("prediction")
      .setLabelColumn("target")
      .setLr(0.002f)    // If the learning rate is too high, there will be no improvements.
      .setMaxEpochs(15)
      .setValidationSplit(0.2f) // Use 20% of the data for validation
      .setEnableOutputLogs(true)

    // Pipeline with ClassifierDL
    val pipeline = new Pipeline().setStages(Array(documentAssembler, bertSentenceEmbeddings, classifierDL))
    println("Created Classifier pipeline")
    // Fit the model
    val classifierModel = pipeline.fit(train_clean)
    println("Fit Model")
    //predictions:
    val test_ds = loadTestData("/home/daniel/repos/scala-nlp-project/src/main/resources/data/test.csv")
    val test_clean = cleanTestData(test_ds)
    println("Test Data Cleaned")
    val predictions = classifierModel.transform(test_clean)
    predictions.show()
    
    val resultDf = predictions.select($"*", explode($"prediction.result").as("predicted_label"))
    val columnsToDrop = Seq("document", "sentence_embeddings", "prediction")
    val trimDf = resultDf.drop(columnsToDrop: _*)
    trimDf.show()
    trimDf.write
      .option("header", "true")
      .option("delimiter", ",") 
      .mode("overwrite") // Options are: "overwrite", "append", "ignore", "error" (default)
      .csv("/home/daniel/repos/scala-nlp-project/src/main/resources/output")
  }

  def loadData(inputFile: String)(implicit spark: SparkSession):Dataset[Tweet] = {
    import spark.implicits._
    spark
      .read
      .format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ",") // Specify delimiter, default is ","
      .option("quote", "\"") 
      .option("escape", "\"") // Escape character to handle quotes within quoted strings
      .option("multiline", "true") // Handle multi-line records
      .load(inputFile)
      .as[Tweet]
  }

  def loadTestData(inputFile: String)(implicit spark: SparkSession):Dataset[TestTweet] = {
    import spark.implicits._
    spark
      .read
      .format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ",")
      .option("quote", "\"") 
      .option("escape", "\"")
      .option("multiline", "true") 
      .load(inputFile)
      .as[TestTweet]
  }

  def cleanData(ds: Dataset[Tweet]): Dataset[Tweet] = {
    ds.map(tweet => tweet.copy(text = cleanText(tweet.text)))(Encoders.product[Tweet])
  }
  def cleanTestData(ds: Dataset[TestTweet]): Dataset[TestTweet] = {
    ds.map(tweet => tweet.copy(text = cleanText(tweet.text)))(Encoders.product[TestTweet])
  }
  def cleanText(text: String): String = {
    val lowerCaseText = text.toLowerCase
    val noEmails = lowerCaseText.replaceAll("[a-z0-9+._-]+@[a-z0-9+._-]+\\.[a-z0-9+_-]+", "")
    val noLinks = noEmails.replaceAll("(http|https|ftp|ssh)://[\\w_-]+(?:\\.[\\w_-]+)+(?:[\\w.,@?^=%&:/~+#-]*[\\w@?^=%&/~+#-])?", "")
    noLinks.replaceAll("[^\\w ]", "").trim.replaceAll("\\s+", " ")  // Eliminate special characters, excess whitespace
  }

}
