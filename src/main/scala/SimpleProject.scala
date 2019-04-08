import com.salesforce.op._
import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._
import com.salesforce.op.readers.DataReaders
import com.salesforce.op.stages.impl.classification.BinaryClassificationModelSelector
import com.salesforce.op.stages.impl.classification.BinaryClassificationModelsToTry._
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession


/**
  * @param id              individual id
  * @param age             individual's age
  * @param workclass       individual's employment category
  * @param education       individual's education level
  * @param educationnum    number of years in education
  * @param maritalstatus   individual's marital status
  * @param occupation      individual's occupation
  * @param relationship    individual's relationship
  * @param race            individual's race
  * @param sex             individual's sex
  * @param capitalgain     individual's capital gain
  * @param capitalloss     individual's capital loss
  * @param hoursperweek    number of hours worked
  * @param nativecountry   individual's native country
  * @param income          whether individual's income is above or below 50K
  */
case class Individual
(
  id: Int,
  age: Option[Int],
  workclass: Option[String],
  education: Option[String],
  educationnum: Double,
  maritalstatus: Option[String],
  occupation: Option[String],
  relationship: Option[String],
  race: Option[String],
  sex: Option[String],
  capitalgain: Double,
  capitalloss: Double,
  hoursperweek: Double,
  nativecountry: Option[String],
  income: String
)

object SimpleProject {

  def main(args: Array[String]): Unit = {
    if (args.isEmpty) {
      println("You need to pass in the CSV file path as an argument")
      sys.exit(1)
    }
    val csvFilePath = args(0)
    println(s"Using user-supplied CSV file path: $csvFilePath")

    // Set up a SparkSession as normal
    implicit val spark = SparkSession.builder.config(new SparkConf().setMaster("local")).getOrCreate()
    import spark.implicits._ // Needed for Encoders for the Passenger case class

    ////////////////////////////////////////////////////////////////////////////////
    // RAW FEATURE DEFINITIONS
    /////////////////////////////////////////////////////////////////////////////////

    val income = FeatureBuilder.RealNN[Individual].extract(_.income.compareTo("<=50K").toRealNN).asResponse
    val age = FeatureBuilder.Integral[Individual].extract(_.age.toIntegral).asPredictor
    val workclass = FeatureBuilder.PickList[Individual].extract(_.workclass.map(_.toString).toPickList).asPredictor
    val education = FeatureBuilder.PickList[Individual].extract(_.education.map(_.toString).toPickList).asPredictor
    val educationnum = FeatureBuilder.Real[Individual].extract(_.educationnum.toReal).asPredictor
    val maritalstatus = FeatureBuilder.PickList[Individual].extract(_.maritalstatus.map(_.toString).toPickList).asPredictor
    val occupation = FeatureBuilder.PickList[Individual].extract(_.occupation.map(_.toString).toPickList).asPredictor
    val relationship = FeatureBuilder.PickList[Individual].extract(_.relationship.map(_.toString).toPickList).asPredictor
    val race = FeatureBuilder.PickList[Individual].extract(_.race.map(_.toString).toPickList).asPredictor
    val sex = FeatureBuilder.PickList[Individual].extract(_.sex.map(_.toString).toPickList).asPredictor
    val capitalgain = FeatureBuilder.Real[Individual].extract(_.capitalgain.toReal).asPredictor
    val capitalloss = FeatureBuilder.Real[Individual].extract(_.capitalloss.toReal).asPredictor
    val hoursperweek = FeatureBuilder.Real[Individual].extract(_.hoursperweek.toReal).asPredictor
    val nativecountry = FeatureBuilder.Country[Individual].extract(_.nativecountry.map(_.toString).toCountry).asPredictor

    ////////////////////////////////////////////////////////////////////////////////
    // TRANSFORMED FEATURES
    /////////////////////////////////////////////////////////////////////////////////

    val individualFeatures = Seq(
      age, workclass, education, educationnum, maritalstatus, occupation, relationship,
      sex, capitalgain, capitalloss, hoursperweek, nativecountry
    ).transmogrify()


    // Optionally check the features with a sanity checker
    val checkedFeatures = income.sanityCheck(individualFeatures, removeBadFeatures = true)

    // Define the model we want to use (here a simple logistic regression) and get the resulting output
    val prediction = BinaryClassificationModelSelector.withTrainValidationSplit(
      modelTypesToUse = Seq(OpLogisticRegression, OpRandomForestClassifier)
    ).setInput((income, individualFeatures)).getOutput()

    val evaluator = Evaluators.BinaryClassification().setLabelCol(income).setPredictionCol(prediction)

    ////////////////////////////////////////////////////////////////////////////////
    // WORKFLOW
    /////////////////////////////////////////////////////////////////////////////////

    val dataReader = DataReaders.Simple.csvCase[Individual](path = Option(csvFilePath), key = _.id.toString)

    val workflow = new OpWorkflow().setResultFeatures(income, prediction).setReader(dataReader)

    val model = workflow.train()
    println(s"Model summary:\n${model.summaryPretty()}")

    println("Scoring the model")
    val (scores, metrics) = model.scoreAndEvaluate(evaluator = evaluator)

    println("Metrics:\n" + metrics)

    spark.stop()
  }
}
