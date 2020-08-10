from preprocess.crawl import Crawl
from preprocess.filter import Filter
import pandas as pd
import tqdm
import numpy as np
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from tqdm import tqdm, notebook
import pandas as pd
from typing import List
import time
import asyncio
from collections import Counter
from datetime import datetime
from preprocess.stemming import Stemming

start_time = time.time()


def main() -> None:
    #Choose Your Crawl Output
    file_name: str = "crawl.csv"
    threshold_sentiment_positive:float = 0.21
    threshold_sentiment_negative:float = 0.21

    #Positive Negative Words
    file_name_positive_label: str = "./positive_label_word.txt"
    file_name_negative_label: str = "./negative_label_word.txt"

    #Choose Your Preferable Processor Capability
    total_thread_processor: int =  8

    df = pd.read_csv(filepath_or_buffer=file_name, nrows=30000)
    df.drop_duplicates(subset="tweet", keep="first", inplace=True)
    print("==========================")
    df["tweet"] = df["tweet"].str.lower()
    df.to_csv("result_preprocess.csv")

    df_filtered_object = Filter()
    df_filtered_object: pd.DataFrame = df_filtered_object.filter_unused_character(
        df=df
    )
    df_filtered_object.to_csv("result_2_filtering.csv")

    # Sastrawi
    stopword = StopWordRemoverFactory().create_stop_word_remover()
    stemmer = StemmerFactory().create_stemmer()

    filter_common_word_obj = Filter()
    # Replacement Tweet Common word
    list_tweet_replaced_with_common_word: List[str] = list()
    for x in tqdm(df_filtered_object["tweet"]):
        temp_result: str = filter_common_word_obj.replace_common_word(tweet=x)
        list_tweet_replaced_with_common_word.append(temp_result)
    df_filtered_object["tweet"] = list_tweet_replaced_with_common_word

    # Stop Word Remove
    list_tweet_stopword: List[str] = list()
    for x in df_filtered_object["tweet"]:
        stop_word_result: str = stopword.remove(text=x)
        list_tweet_stopword.append(stop_word_result)
    df_filtered_object["tweet"] = list_tweet_stopword

    # Process Stemming
    stem_object = Stemming()
    result_stem_object: List[str] = stem_object.stem(tweets=df_filtered_object["tweet"], total_thread=total_thread_processor)
    df_filtered_object["tweet"] = result_stem_object

    # Save Stemming
    df_filtered_object.to_csv(path_or_buf="result_3_stemming.csv")

    # Count Most Word
    combined_word: str = None
    for x in df_filtered_object["tweet"]:
        combined_word = str(combined_word) + x

    counter_result: Counter = Counter(combined_word.split())
    counter_result_most_common: List[tuple] = counter_result.most_common(10)
    print(counter_result_most_common)

    import matplotlib.pyplot as plt

    x_val = [x[0] for x in counter_result_most_common]
    y_val = [x[1] for x in counter_result_most_common]
    plt.plot(x_val, y_val)
    plt.plot(x_val, y_val, "or")
    plt.savefig("graph_most_common_word.jpg")
    plt.clf()

    # Read Label Positive And Negative Words
    positive = pd.read_csv(filepath_or_buffer=file_name_positive_label, header=None)
    positive = positive[0].values.tolist()
    positive = "|".join(positive)

    negative = pd.read_csv(filepath_or_buffer=file_name_negative_label, header=None)
    negative = negative[0].values.tolist()
    negative = "|".join(negative)

    # Remove Empty Value(NaN, None, NaT)
    df_filtered_object = df_filtered_object[pd.notnull(df_filtered_object["tweet"])]

    # Weigh Each Tweet Positive Or Negative Sentiment
    df_filtered_object["positive"] = [
        len(re.findall(positive, x.lower()))
        / (1 if len(x.split()) == 0 else len(x.split()))
        for x in df_filtered_object["tweet"]
    ]
    df_filtered_object["negative"] = [
        len(re.findall(negative, x.lower()))
        / (1 if len(x.split()) == 0 else len(x.split()))
        for x in df_filtered_object["tweet"]
    ]
    df_filtered_object.to_csv(path_or_buf="result_4_labeling.csv")

    #Draw Sentiment Into Graph 
    import matplotlib.pyplot as plt
    total_negative_tweets: int = len(
        [x for x in df_filtered_object["negative"] if x >= threshold_sentiment_negative]
    )
    total_positive_tweets: int = len(
        [x for x in df_filtered_object["positive"] if x >= threshold_sentiment_positive]
    )
    #Count Neutral Tweet
    counter_neutral_tweets: int = 0
    for index, positive_value_of_tweet in tqdm(enumerate(df_filtered_object["positive"])):
        if positive_value_of_tweet == 0 or positive_value_of_tweet < threshold_sentiment_positive:
            negative_value_of_tweet = df_filtered_object.iloc[index, 35]
            if negative_value_of_tweet == 0 or negative_value_of_tweet < threshold_sentiment_negative:
                counter_neutral_tweets = counter_neutral_tweets + 1
    print(total_negative_tweets)
    print(total_positive_tweets)
    print(counter_neutral_tweets)
    total_tweets = df.shape[0]

    df_bar = pd.Series(
        data=[total_negative_tweets, total_positive_tweets, counter_neutral_tweets],
        index=[
            "negative " + str(total_negative_tweets),
            "positive " + str(total_positive_tweets),
            "neutral " + str(counter_neutral_tweets) 
        ],
    )

    bar = plt.bar(df_bar.index, df_bar.values)
    bar[0].set_color("#EE204D")
    bar[1].set_color("#00FF00")
    bar[2].set_color("#0000FF")
    plt.savefig("graph_sentiment_bar_chart.jpg")
    plt.clf()
        
        

if __name__ == "__main__":
    main()

print("--- %s seconds ---" % (time.time() - start_time))
