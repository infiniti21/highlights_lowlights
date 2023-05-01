! pip install pyspark
! pip install nltk
from pyspark.ml.feature import Word2Vec, Tokenizer, HashingTF, IDF, StopWordsRemover
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import CountVectorizer
from pyspark.sql.functions import lit, split
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, FloatType
from pyspark.sql.functions import explode, col
from pyspark.sql.functions import udf
from pyspark import SparkConf, SparkContext
import pandas as pd
import numpy as np
import re
from pyspark.sql.functions import regexp_replace

from nltk.corpus import stopwords
import nltk

# Filtering only business ids with sushi in business description
fil_ids = list(pd.read_csv('intersection.csv')['intersection'])

conf = SparkConf().setAppName("yelpin").setMaster("local[*]")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

df = spark.read.csv( '/user/Mani/yelp_review.csv', header=True)
df = df.withColumn("label", df["stars"].cast("double"))
df = df.dropna(subset=['label', 'text', 'funny', 'cool',"useful"])

nltk.download('stopwords')
stop = stopwords.words('english')

new_stopwords = ['', ' ']
stop.extend(new_stopwords)

df = df.select('business_id','review_id','text', 'label')
df = df.filter(df.label.isin(1.0,2.0,3.0,4.0,5.0))

punct_df = df.withColumn("text_punct_rem", regexp_replace("text", r"[^\w\s]", ""))

# create a tokenizer object
tokenizer = Tokenizer(inputCol="text_punct_rem", outputCol="words")

# apply the tokenizer on the DataFrame
tokenized_df = tokenizer.transform(punct_df)

remover = StopWordsRemover(inputCol="words", outputCol="clean_words")

cleaned_df = remover.transform(tokenized_df)

# define a UDF to join the list of words into a string
join_udf = udf(lambda x: " ".join(x), StringType())

# apply the UDF to the clean_words column and store the result in a new column called clean_text
cleaned_df = cleaned_df.withColumn("clean_text", join_udf("clean_words"))

final_df2 = cleaned_df.select("business_id", 'review_id', "label", "clean_words", "clean_text")

df.unpersist()
punct_df.unpersist()
tokenized_df.unpersist()
cleaned_df.unpersist()

# bus_list = ['wZkjm9TGa7nXLYHxhIRO5w', 'vOMDU31gdylrzBhAKC9QbA']
bus_list = fil_ids
# final_df = final_df2.filter(final_df2.business_id.isin([bus_list[1],]))

final_df = final_df2.filter(final_df2.business_id.isin(bus_list))
final_df2.unpersist()

good_rating_df = final_df.filter(final_df.label.isin(4, 4.5, 5))
final_df.unpersist()

rem_results = ['best', 'great', 'place', 'good', 'back', 'again', '', 'really', 'always', 'go', 
               'mainly', 'last', 'amazing', 'love', 'restaurant']

def topic_count(topics_list, clean_words):
        return [(t, 1 if t in clean_words else 0) for t in topics_list]

udf_topic_count = udf(topic_count, ArrayType(StructType([StructField('topic', StringType()), StructField('present', IntegerType())])))
    
def topic_rating(topic_counts, label):
        return [(t[0], (t[1], label)) for t in topic_counts]

udf_topic_rating = udf(topic_rating, ArrayType(StructType([StructField('topic', StringType()), StructField('present', StructType([StructField('present', IntegerType()), StructField('label', FloatType())]))])))

i = 0  # Counter for number of businesses successfully executed
highlights_dict = {'business': [], 'highlights' : [],}
for bus in bus_list[i:]:
    try:
        good_bus_df = good_rating_df.filter(good_rating_df.business_id.isin([bus,]))
        num_topics = 10
        # Create the document-term matrix using CountVectorizer
        cv = CountVectorizer(inputCol='clean_words', outputCol='features')
        # Get the vocabulary from the CountVectorizer
        cv_model = cv.fit(good_bus_df)
        vocabulary = cv_model.vocabulary
        topics = [vocabulary[i] for i in range(num_topics) if vocabulary[i] not in rem_results]
        # add code to add a column 'topics' to good_bus_df which has the list 'topics' in each cell. Store resulting df in 'good_bus_topics'
        good_bus_topics_df = good_bus_df.withColumn('topics', lit(' '.join([vocabulary[i] for i in range(num_topics) if vocabulary[i] not in rem_results])))
        good_bus_topics_df = good_bus_topics_df.withColumn("topics_list", split(good_bus_topics_df.topics, " ")).drop("topics")
    #     good_bus_topics_df.show(2)

        rel_df = good_bus_topics_df.select('label', 'clean_words', 'topics_list')

        rel_df = rel_df.withColumn('topic_counts', udf_topic_count('topics_list', 'clean_words'))

        rel_df = rel_df.withColumn('topic_counts_rating', udf_topic_rating('topic_counts', 'label'))

        rel_df = rel_df.select(explode(col('topic_counts_rating')).alias('topic_count_rating'))

        rdd = rel_df.rdd
        result_rdd = rdd.map(lambda x: x[0]).map(lambda x: (x[0], (x[1][0], x[1][0]*x[1][1], 1))).reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1], x[2]+y[2])).mapValues(lambda x: (x[0]/x[2], x[1]/x[0])).mapValues(lambda x: (x[0] * (1-x[1]/5))).sortBy(lambda x: x[1], ascending=False)
        highlights_dict['business'].append(bus)
        highlights_dict['highlights'].append(result_rdd.collect())
        highlights_df = pd.DataFrame.from_dict(highlights_dict)
        highlights_df.to_csv('business_highlights.csv')
        good_bus_df.unpersist()
        good_bus_topics_df.unpersist()
        rel_df.unpersist()
        rdd.unpersist()
        result_rdd.unpersist()
        print(i)
        i += 1
    except Exception as e:
        print(i, "An error occurred:", e)
        i += 1
        pass

