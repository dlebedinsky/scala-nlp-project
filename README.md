# Scala Course final project: Tweet Classification with Spark NLP
 
### Motivation and Data Source
I created this project with the goal of submitting a classification of disaster-related vs non-disaster-related tweets to this [Kaggle competition.](https://www.kaggle.com/competitions/nlp-getting-started/data)
So far, it has achieved an accuracy value of 0.7977, according to their hidden test set keys.

### Installation
I followed [these instructions](https://www.tutorialspoint.com/apache_spark/apache_spark_installation.htm) to install the dependencies for this project. When you reach the Download Apache Spark step, you must select version 3.4.2, "pre-built for Apache Hadoop 3.3 and later." Optionally, you can run the following to make it easier to run the `spark-submit` command:
```shell
export PATH=$PATH:/usr/local/spark/bin
```

### Use
After you executed `sbt compile assembly` to get a JAR (without Apache Spark), you can use `spark-submit` like this:

```shell
spark-submit --driver-memory 4g --class Main target/scala-2.12/spark-nlp-starter-assembly-5.1.0.jar
```

This will execute the code in `Main` class, show training and validation loss/accuracy by epoch in the console, and classify the test data in src/main/resources/output. I have optimized the command for systems with relatively low memory (~8GB). Sample console output is included in src/main/resources/. 

### Future improvements
I hope to eventually try the following with this project:
* Run the training pipeline in a cloud environment with a powerful GPU, so that I can feasibly train the ClassiferDL model for more epochs and with a smaller learning rate, to achieve a better test accuracy; or run on a distributed Spark cluster, if GPU access is infeasible.
* Visualize the training/validation loss and accuracy improvements natively in Scala, and create a confusion matrix visualizing the inaccuracy distribution, possibly using Vegas or a similar library.
*  Experiment with alternative sentence embedding models, or add a tokenizer intermediate step.
