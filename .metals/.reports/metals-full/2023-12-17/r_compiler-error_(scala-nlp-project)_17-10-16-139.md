file://<WORKSPACE>/src/main/scala/Main.scala
### java.lang.ClassCastException: class scala.reflect.internal.Types$PolyType cannot be cast to class scala.reflect.internal.Types$OverloadedType (scala.reflect.internal.Types$PolyType and scala.reflect.internal.Types$OverloadedType are in unnamed module of loader java.net.URLClassLoader @434d619a)

occurred in the presentation compiler.

action parameters:
offset: 3399
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
    train_ds.show(10, truncate=50)
    
    println("Dataset loaded")
    // Define the stages of the Pipeline
    val documentAssembler = new DocumentAssembler()
      .setInputCol("description")
      .setOutputCol("document")
    
    val bertSentenceEmbeddings = BertSentenceEmbeddings
      .pretrained("sent_small_bert_L8_512")
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
    val test_ds = loadData("<HOME>/repos/spark-nlp-starter/src/main/resources/data/news_category_test.csv")
    println("Test Data Cleaned")
    
    println("Test Data Embedded")
    val predictions = classifierModel.transform(test_ds)
    predictions.show()
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

  def cleanData(ds: Dataset[Tweet]): Dataset[Tweet] = {
    val cleanedDf = ds.toDF().map { row =>
      val id = row.getAs[String]("id")
      val keyword = row.getAs[String]("keyword")
      val location = row.getAs[String]("location")
      val text = cleanText(row.getAs[String]("text"))
      val target = row.g@@etAs[String]("target")

      Tweet(id, keyword, location, text, target)
    }(Encoders.product[Tweet])

    cleanedDf.as[Tweet]
  }

}

```



#### Error stacktrace:

```
scala.reflect.internal.Symbols$Symbol.alternatives(Symbols.scala:1981)
	scala.meta.internal.pc.PcDefinitionProvider.definition(PcDefinitionProvider.scala:97)
	scala.meta.internal.pc.PcDefinitionProvider.definition(PcDefinitionProvider.scala:16)
	scala.meta.internal.pc.ScalaPresentationCompiler.$anonfun$definition$1(ScalaPresentationCompiler.scala:339)
```
#### Short summary: 

java.lang.ClassCastException: class scala.reflect.internal.Types$PolyType cannot be cast to class scala.reflect.internal.Types$OverloadedType (scala.reflect.internal.Types$PolyType and scala.reflect.internal.Types$OverloadedType are in unnamed module of loader java.net.URLClassLoader @434d619a)