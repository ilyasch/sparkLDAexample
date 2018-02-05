import os
import sys


PATH_TO_PROJECT = os.getcwd()
PATH_SEPARATOR = "/"
PATH_TO_DATA = PATH_TO_PROJECT+PATH_SEPARATOR+"newset.csv"
PATH_TO_MODEL = PATH_TO_PROJECT+PATH_SEPARATOR+"model"


# Path for spark source folder
os.environ['SPARK_HOME']= PATH_TO_PROJECT
# Append pyspark  to Python Path
sys.path.append(PATH_TO_PROJECT+PATH_SEPARATOR+"python/pyspark/")
sys.path.append(PATH_TO_PROJECT+PATH_SEPARATOR+"python/")
sys.path.append(PATH_TO_PROJECT+PATH_SEPARATOR+"python/lib/py4j-0.10.4-src.zip")




try:
    from pyspark import SparkContext, SparkConf
    from pyspark.sql import SparkSession
    from pyspark.mllib.clustering import LDA, LDAModel
    from pyspark.mllib.linalg import Vectors
    from pyspark.mllib.linalg import SparseVector
    from pyspark.sql import SQLContext, Row
    from pyspark.ml.feature import CountVectorizer,CountVectorizerModel
    from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover
    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)


if __name__ == "__main__":

    # Load and parse the data
    spark = SparkSession.builder.getOrCreate()

    #nmbTopics, alpha, beta, itrations
    #params = sys.argv

    PATH_TO_MODEL = os.path.join(PATH_TO_PROJECT, "model")
    isEmpthy = True

    for dirpath, dirnames, files in os.walk(PATH_TO_MODEL):
        if files:
            print(dirpath, 'has files')
            isEmpthy = False
            break
        else:
            print(dirpath, 'is empty')
            isEmpthy = True
            break

    if isEmpthy:
            print ("Model already exists, Check summary.txt ")
            #sameModel = LDAModel.load(spark.sparkContext,PATH_TO_MODEL)
    else:
            print("Training the model")
            #DF
            rawdata = spark.read.load(PATH_TO_DATA, format="csv", header=True)
            rawdata = rawdata.na.fill('Empthy')
            rawdata.show(10)

            #Tokenize My inportant words after preprocessing
            tokenizer = Tokenizer(inputCol="words", outputCol="somewords")
            wordsDataFrame = tokenizer.transform(rawdata)
            wordsDataFrame.show(10)

            hashingTF = HashingTF(inputCol="somewords", outputCol="rawFeatures", numFeatures=20)
            featurizedData = hashingTF.transform(wordsDataFrame)
            idf = IDF(inputCol="rawFeatures", outputCol="features")
            idfModel = idf.fit(featurizedData)
            rescaledData = idfModel.transform(featurizedData)
            rescaledData.select("somewords", "features").show(10)

            cv = CountVectorizer(inputCol="somewords", outputCol="vectors")
            cvmodel = cv.fit(rescaledData)
            df_vect = cvmodel.transform(rescaledData)
            df_vect.show(10)

            sparsevector = df_vect.select('id','vectors').rdd.map(lambda x: [int(x[0]), Vectors.dense(x[1])])

            numTopics = 35
            # Train the LDA model
            model = LDA.train(sparsevector, k=numTopics, seed=123,maxIterations=150,docConcentration=0.01,topicConcentration=0.01, optimizer="online")

            # Save and load model
            model.save(spark.sparkContext, PATH_TO_MODEL)

            topics = model.topicsMatrix()
            vocabArray = cvmodel.vocabulary
            wordNumbers = 10  # number of words per topic
            topicIndices = spark.sparkContext.parallelize(model.describeTopics(maxTermsPerTopic=wordNumbers))

            def topic_render(topic):  # specify vector id of words to actual words
                terms = topic[0]
                result = []
                for i in range(wordNumbers):
                    term = vocabArray[terms[i]]
                    result.append(term)
                return result


            topics_final = topicIndices.map(lambda topic: topic_render(topic)).collect()
            outputfile = open(PATH_TO_PROJECT+PATH_SEPARATOR+"summary1.txt","w")

            for topic in range(0,numTopics):
                print ("Topic" + str(topic) + ":")
                outputfile.write("Topic"+ str(topic) + ":")
                for term in topics_final[topic]:
                    print (term)
                    outputfile.write(term)
                print ('\n')
                outputfile.write("\n")

            outputfile.close()
