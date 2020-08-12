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
from preprocess.tokenizer_local import TokenizerLocal
from gensim import models
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import collections
import re
import string

start_time = time.time()


def main() -> None:
    # Choose Your Crawl Output
    file_name: str = "crawl_result.csv"
    threshold_sentiment_positive: float = 0.21
    threshold_sentiment_negative: float = 0.21

    # Positive Negative Words
    file_name_positive_label: str = "./positive_label_word.txt"
    file_name_negative_label: str = "./negative_label_word.txt"

    # Choose Your Preferable Processor Capability
    total_thread_processor: int = 8

    df = pd.read_csv(filepath_or_buffer=file_name, nrows=30000)
    df.drop_duplicates(subset="tweet", keep="first", inplace=True)
    print("==========================")
    df["tweet"] = df["tweet"].str.lower()
    df.to_csv("result_preprocess.csv")

    df_filtered_object = Filter()
    df_filtered_object: pd.DataFrame = df_filtered_object.filter_unused_character(df=df)
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
    # stem_object = Stemming()
    # result_stem_object: List[str] = stem_object.stem(tweets=df_filtered_object["tweet"], total_thread=total_thread_processor)
    # df_filtered_object["tweet"] = result_stem_object

    # Save Stemming
    # df_filtered_object.to_csv(path_or_buf="result_3_stemming.csv")

    # load stemming
    df_filtered_object = pd.read_csv(filepath_or_buffer="./result_3_stemming.csv")
    print(df_filtered_object.head())
    # Count Most Word
    combined_word: str = None
    for x in tqdm(df_filtered_object["tweet"]):
        combined_word = str(combined_word) + str(x)

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
        for x in tqdm(df_filtered_object["tweet"])
    ]
    df_filtered_object["negative"] = [
        len(re.findall(negative, x.lower()))
        / (1 if len(x.split()) == 0 else len(x.split()))
        for x in tqdm(df_filtered_object["tweet"])
    ]
    df_filtered_object.to_csv(path_or_buf="result_4_labeling.csv")

    # Draw Sentiment Into Graph
    import matplotlib.pyplot as plt

    total_negative_tweets: int = len(
        [x for x in df_filtered_object["negative"] if x >= threshold_sentiment_negative]
    )
    total_positive_tweets: int = len(
        [x for x in df_filtered_object["positive"] if x >= threshold_sentiment_positive]
    )
    # Count Neutral Tweet
    counter_neutral_tweets: int = 0
    for index, positive_value_of_tweet in tqdm(
        enumerate(df_filtered_object["positive"])
    ):
        if (
            positive_value_of_tweet == 0
            or positive_value_of_tweet < threshold_sentiment_positive
        ):
            negative_value_of_tweet = df_filtered_object.iloc[index, 35]
            if (
                negative_value_of_tweet == 0
                or negative_value_of_tweet < threshold_sentiment_negative
            ):
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
            "neutral " + str(counter_neutral_tweets),
        ],
    )

    bar = plt.bar(df_bar.index, df_bar.values)
    bar[0].set_color("#EE204D")
    bar[1].set_color("#00FF00")
    bar[2].set_color("#0000FF")
    plt.savefig("graph_sentiment_bar_chart.jpg")
    plt.clf()

    df_filtered_object["sentimen"] = [
        0
        if df_filtered_object.iloc[x].positive >= df_filtered_object.iloc[x].negative
        else 1
        for x in range(df_filtered_object.shape[0])
    ]
    df_filtered_object.to_csv(path_or_buf="result_5_sentiment_label.csv")

    # Tokenizer
    t = TokenizerLocal()
    tokens: List[List[str]] = t.create_token(tweets=(df_filtered_object["tweet"]).tolist())
    
    # Removes Stopwords
    f = Filter()
    filtered_words: List[List[str]] = [
        f.remove_stop_words(tokens=x) for x in tqdm(tokens)
    ]

    # Join Token
    result = [" ".join(x) for x in filtered_words]

    # Add Column "tweet_token" to dataframe
    df_filtered_object["tokens"] = filtered_words

    # Add Column "tweet_final" to dataframe
    df_filtered_object["tweet_final"] = result

    # Remove unused column
    df_final = df_filtered_object[
        [
            "date",
            "time",
            "username",
            "tweet",
            "tokens",
            "tweet_final",
            "positive",
            "negative",
            "sentimen",
        ]
    ]
    df_final.to_csv("result_6_tokenization.csv")

    

def training(total_row: int = 1000, embedding_dim: int= 100, batch_size: int = 100) -> None:
    ### Split data into test and train
    df = pd.read_csv(filepath_or_buffer="result_6_tokenization.csv", nrows=total_row)
    data_train, data_test = train_test_split(df, test_size=0.1, random_state=42)
    print(len(data_train), len(data_test))
    # Train
    all_training_words = [word for tokens in data_train["tokens"] for word in tokens]
    training_sentence_lengths = [len(tokens) for tokens in data_train["tokens"]]
    TRAINING_VOCAB = sorted(list(set(all_training_words)))
    print("%s words total, with a vocabulary size of %s" % (len(all_training_words), len(TRAINING_VOCAB)))
    print("Max sentence length is %s" % max(training_sentence_lengths))
    # Test
    all_test_words = [word for tokens in data_test["tokens"] for word in tokens]
    test_sentence_lengths = [len(tokens) for tokens in data_test["tokens"]]
    TEST_VOCAB = sorted(list(set(all_test_words)))
    print("%s words total, with a vocabulary size of %s" % (len(all_test_words), len(TEST_VOCAB)))
    print("Max sentence length is %s" % max(test_sentence_lengths))

    ### Load Google News Word2Vec model
    word2vec_path = 'D:\Documents\Skripsi_Semangat\skripsi\word2vec\idwiki_word2vec_100.zip'
    # word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    word2vec = models.word2vec.Word2Vec.load("D:\Documents\Skripsi_Semangat\skripsi\word2vec\idwiki_word2vec_100.model")

    # Get Embed
    training_embeddings = get_word2vec_embeddings(word2vec, data_train, generate_missing=True)
    MAX_SEQUENCE_LENGTH = 50
    EMBEDDING_DIM = embedding_dim

    ### Tokenize and Pad sequences
    tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
    tokenizer.fit_on_texts(data_train["tokens"].tolist())
    training_sequences = tokenizer.texts_to_sequences(data_train["tokens"].tolist())

    train_word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(train_word_index))

    train_cnn_data = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    train_embedding_weights = np.zeros((len(train_word_index)+1, EMBEDDING_DIM))
    for word,index in train_word_index.items():
        train_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
    print(train_embedding_weights.shape)

    test_sequences = tokenizer.texts_to_sequences(data_test["tweet_final"].tolist())
    test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    label_names = ['positive', 'negative']
    y_train = data_train[label_names].values
    x_train = train_cnn_data
    y_tr = y_train

    model = ConvNet(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM, 
                len(list(label_names)))


    # Train CNN
    num_epochs = 3
    batch_size: int = batch_size
    hist = model.fit(x_train, y_tr, epochs=num_epochs, validation_split=0.1, shuffle=True, batch_size=batch_size)
    model.save("./model")
    predictions = model.predict(test_cnn_data, batch_size=1000, verbose=1)

    labels = [0, 1]

    prediction_labels=[]
    for p in predictions:
        prediction_labels.append(labels[np.argmax(p)])

    print(">> Predicted Result >> ", sum(data_test.sentimen==prediction_labels)/len(prediction_labels))
    print(data_test.sentimen.value_counts())

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_comments, generate_missing=False):
    embeddings = clean_comments['tokens'].apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    return list(embeddings)

def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
    
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embeddings],
                            input_length=max_sequence_length,
                            trainable=False)
    
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    convs = []
    filter_sizes = [2,3,4,5,6]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=200, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = GlobalMaxPooling1D()(l_conv)
        convs.append(l_pool)


    l_merge = concatenate(convs, axis=1)

    x = Dropout(0.1)(l_merge)  
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    preds = Dense(labels_index, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()
    return model

def load_model(total_row=29000):
    import keras
    from keras.preprocessing.text import Tokenizer


    df = pd.read_csv(filepath_or_buffer="result_6_tokenization.csv", nrows=total_row)
    data_train, data_test = train_test_split(df, test_size=0.1, random_state=42)

    all_training_words = [word for tokens in data_train["tokens"] for word in tokens]
    TRAINING_VOCAB = sorted(list(set(all_training_words)))

    tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
    tokenizer.fit_on_texts(data_train["tokens"].tolist())

    model = keras.models.load_model('model')

    test_sequences = tokenizer.texts_to_sequences(data_test["tweet_final"].tolist())
    MAX_SEQUENCE_LENGTH = 50
    test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    predictions = model.predict(test_cnn_data, batch_size=1000, verbose=1)
    labels = [0, 1]
    prediction_labels=[]
    for p in predictions:
        prediction_labels.append(labels[np.argmax(p)])

    print(">> Hasil >> ", sum(data_test.sentimen==prediction_labels)/len(prediction_labels))
    print(data_test.sentimen.value_counts())


if __name__ == "__main__":
    # main()
    #training(total_row=29000, embedding_dim=2900, batch_size=5000)
    load_model()


print("--- %s seconds ---" % (time.time() - start_time))
