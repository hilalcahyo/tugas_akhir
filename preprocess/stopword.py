from typing import List


list_tweet_stopword: List[str] = list()
    for x in df_filtered_object["tweet"]:
        stop_word_result: str = stopword.remove(text=x)
        list_tweet_stopword.append(stop_word_result)
    df_filtered_object["tweet"] = list_tweet_stopword