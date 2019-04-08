name := "transmogrify-project"

scalaVersion := "2.11.12"

val transmogrifaiVersion = "0.5.1"

val sparkVersion = "2.3.2"

resolvers += Resolver.bintrayRepo("salesforce", "maven")

lazy val sparkDependencies = Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion
)

libraryDependencies += "com.salesforce.transmogrifai" %% "transmogrifai-core" % transmogrifaiVersion

libraryDependencies ++= sparkDependencies

