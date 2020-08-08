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

start_time = time.time()



def main() -> None:
    loop = asyncio.get_event_loop()
    file_name: str = "result.csv"
    crawl_object = Crawl()
    is_created: bool = crawl_object.crawl_and_save(word="Banjir Jakarta", file_name=file_name)

    if is_created:
        df = pd.read_csv(file_name)
        df.drop_duplicates(subset ="tweet", 
                    keep = False, inplace = True) 
        print("==========================")
        df["tweet"] = df["tweet"].str.lower()
        df.to_csv("result_preprocess.csv")

        df_filtered_object = Filter()
        df_filtered_object: pd.DataFrame = df_filtered_object.filter_unused_character(df=df)

        print(df_filtered_object.head())
        df_filtered_object.to_csv("result_filtering.csv")
        
        # Sastrawi
        stopword = StopWordRemoverFactory().create_stop_word_remover()
        stemmer = StemmerFactory().create_stemmer()
        
        filter_common_word_obj = Filter()
        # Replacement Tweet Common word
        list_tweet_replaced_with_common_word: List[str] = list()
        for x in df_filtered_object["tweet"]:
            temp_result: str = filter_common_word_obj.replace_common_word(tweet=x)
            list_tweet_replaced_with_common_word.append(temp_result)
        df_filtered_object["tweet"] = list_tweet_replaced_with_common_word

        # Stop Word Remove
        list_tweet_stopword: List[str] = list()
        for x in df_filtered_object["tweet"]:
            stop_word_result: str = stopword.remove(text=x)
            list_tweet_stopword.append(stop_word_result)
        df_filtered_object["tweet"] = list_tweet_stopword

        # # Stemming
        # list_tweet_stemming: List[str] = list()
        # for x in df_filtered_object["tweet"]:
        #     stemming_result: str = stemmer.stem(text=x)
        #     list_tweet_stemming.append(stemming_result)
        # df_filtered_object["tweet"] = list_tweet_stemming

        # Save Stemming
        df_filtered_object.to_csv(path_or_buf="result_stemming.csv")

        # Count Most Word
        combined_word : str = None
        for x in df_filtered_object["tweet"]:
            combined_word = str(combined_word) + x

        counter_result: Counter = Counter(combined_word.split())
        counter_result_most_common: List[tuple] = counter_result.most_common(10)
        print(counter_result_most_common)

        import matplotlib.pyplot as plt

        x_val = [x[0] for x in counter_result_most_common]
        y_val = [x[1] for x in counter_result_most_common]
        plt.plot(x_val,y_val)
        plt.plot(x_val,y_val,'or')
        plt.show()


if __name__ == "__main__":
    main()

print("--- %s seconds ---" % (time.time() - start_time))