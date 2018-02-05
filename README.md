1) What does my project do:
Large amounts of data becomes difficult to access what we are looking for. So, we need tools and techniques to organize, search and understand vast quantities of information.
Topic modelling provides us with methods to organize, understand and summarize large collections of textual information. It helps in:
    * Discovering hidden topical patterns that are present across the collection.
    * Annotating documents according to these topics.
    * Using these annotations to organize, search and summarize texts.
One of the famous algorithm of topic modelling is Latent Dirichlet Allocation (LDA), it is a widely used topic modelling technique and the TextRank process: a graph-based algorithm to extract relevant key phrases. This project aims to explain how to use LDA within Spark 2.2 to summurize the dataset in input (newset.csv) and explore the topics and terms that correlate together.


2) What script did I develop and what they do:
In the LDA model, each document is viewed as a mixture of topics that are present in the corpus. The model proposes that each term in the document is attributable to one of the document’s topics.
This project is based on SparkLDA (https://spark.apache.org/docs/2.2.0/mllib-clustering.html#latent-dirichlet-allocation-lda). It takes in a collection of documents from newset.csv where each document is reprented as vectors of word counts corresponding to tokens extracted from a talk-turns within conversations.
THe algorithm is coded in mainserver.py


3) Input and output of each script:
Input: The DATASET newset.csv where each row presents a document, and each document is a talk-turn.
The column "id" represents the number og ongoing talk-turns within conversations, the column "Role" presents the speaker, and the row "words" represents the documents after Tokenization. Other columns are not importants for our method.
Output: 
	* The folder /model/: Built by SparkLDA after the training and contains the model knowledge.
  	* The file summary.txt: Contains all the hidden topics discovered by LDA.


4) What you should do to run the code, train again the model and some links of related work, LDA, SPARK modules, etc.
To create a new model: Delete the folders inside /model/ and TRAIN your new model by typing from the terminal: mainserver.py
--Spark Modeules:
    mllib.clustering: LDA and LDAModel
    mllib.linalg: Vectors and SparseVector
    ml.feature:  CountVectorize, CountVectorizerModel, HashingTF, IDF, Tokenizer, and StopWordsRemover
--LINKS:
    https://www.kdnuggets.com/2016/07/text-mining-101-topic-modeling.html
    http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/
