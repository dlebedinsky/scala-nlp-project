file://<WORKSPACE>/src/main/scala/Main.scala
### java.lang.AssertionError: assertion failed: bad position: [2073:2057]

occurred in the presentation compiler.

action parameters:
offset: 2057
uri: file://<WORKSPACE>/src/main/scala/Main.scala
text:
```scala
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.{Dataset, DataFrame, SaveMode}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Encoders

// install spark locally and add to path: export PATH=$PATH:/usr/local/spark/bin
// spark-submit --class "Main" target/scala-2.12/spark-nlp-starter-assembly-5.2.0.jar

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
    // Assuming you have a dataset of Tweets

    val outputPath = "<HOME>/repos/spark-nlp-starter/src/main/resources/output"

    val train_ds: Dataset[Tweet] = loadData("<HOME>/repos/spark-nlp-starter/src/main/resources/data/train.csv")
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
  
    // Initialize ClassifierDLApproach
    val classifierDL = new ClassifierDLApproach()
      .setInputCols(Array("sentence_embeddings"))
      .setOutputCol("prediction")
      .setLabelColumn("target")
      .setLr(0.002f)    // If the learning rate is too high, there will be no improvements.@@
      .setMaxEpochs(10) // Set the number of epochs
      .setValidationSplit(0.2f) // Use 20% of the data for validation
      .setEnableOutputLogs(true)

    // Pipeline with ClassifierDL
    val pipeline = new Pipeline().setStages(Array(documentAssembler, bertSentenceEmbeddings, classifierDL))
    println("Created Classifier pipeline")
    // Fit the model
    val classifierModel = pipeline.fit(train_clean)
    println("Fit Model")
    //predictions:
    val test_ds = loadTestData("<HOME>/repos/spark-nlp-starter/src/main/resources/data/test.csv")
    val test_clean = cleanTestData(test_ds)
    println("Test Data Cleaned")
    val predictions = classifierModel.transform(test_clean)
    predictions.show()
    
    val columnsToDrop = Seq("document", "sentence_embeddings")
    val resultDf = predictions.select($"*", explode($"prediction.result").as("predicted_label")).drop(columnsToDrop: _*)
    resultDf.show()
    resultDf.write
      .option("header", "true")
      .option("delimiter", ",") // Specify delimiter, default is ","
      .mode("overwrite") // Options are: "overwrite", "append", "ignore", "error" (default)
      .csv("<HOME>/repos/spark-nlp-starter/src/main/resources/output")
  }

  def loadData(inputFile: String)(implicit spark: SparkSession):Dataset[Tweet] = {
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
      .as[Tweet]
  }

  def loadTestData(inputFile: String)(implicit spark: SparkSession):Dataset[TestTweet] = {
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
    noLinks.replaceAll("[^\\w ]", "").trim.replaceAll("\\s+", " ")
  }

}

```



#### Error stacktrace:

```
scala.reflect.internal.util.Position$.validate(Position.scala:41)
	scala.reflect.internal.util.Position$.range(Position.scala:58)
	scala.reflect.internal.util.InternalPositionImpl.copyRange(Position.scala:218)
	scala.reflect.internal.util.InternalPositionImpl.withStart(Position.scala:133)
	scala.reflect.internal.util.InternalPositionImpl.withStart$(Position.scala:133)
	scala.reflect.internal.util.Position.withStart(Position.scala:19)
	scala.meta.internal.pc.CompletionProvider.editRange$lzycompute$1(CompletionProvider.scala:415)
	scala.meta.internal.pc.CompletionProvider.editRange$2(CompletionProvider.scala:414)
	scala.meta.internal.pc.CompletionProvider.expected$1(CompletionProvider.scala:424)
	scala.meta.internal.pc.CompletionProvider.safeCompletionsAt(CompletionProvider.scala:501)
	scala.meta.internal.pc.CompletionProvider.completions(CompletionProvider.scala:58)
	scala.meta.internal.pc.ScalaPresentationCompiler.$anonfun$complete$1(ScalaPresentationCompiler.scala:187)
```
#### Short summary: 

java.lang.AssertionError: assertion failed: bad position: [2073:2057]