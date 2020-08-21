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
from keras.layers import (
    Dense,
    Dropout,
    Reshape,
    Flatten,
    concatenate,
    Input,
    Conv1D,
    GlobalMaxPooling1D,
    Embedding,
)
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


def preprocess(total_row: int = 10000, total_thread_processor: int = 2) -> None:
    # Choose Your Crawl Output
    threshold_sentiment_positive: float = 0.1
    threshold_sentiment_negative: float = 0.1

    # Positive Negative Words
    file_name_positive_label: str = "./positive_label_word.txt"
    file_name_negative_label: str = "./negative_label_word.txt"

    # Choose Your Preferable Processor Capability
    total_thread_processor: int = total_thread_processor

    # Crawl File Name
    file_name: str = "crawl_result.csv"

    df = pd.read_csv(filepath_or_buffer=file_name, nrows=total_row)
    df.drop_duplicates(subset="tweet", keep="first", inplace=True)

    # Remove unused column
    df = df[["date", "time", "username", "tweet",]]

    # Preprocessing | Case Folding
    df["tweet_original"] = df["tweet"]
    df["preprocessing_case_folding"] = df["tweet"].str.lower()
    df["tweet"] = df["tweet"].str.lower()

    f = Filter()
    df = f.filter_unused_character(df=df)

    # Preprocessing | Replacement Tweet Common word
    filter_common_word_obj = Filter()
    list_tweet_replaced_with_common_word: List[str] = list()
    for x in tqdm(df["tweet"]):
        temp_result: str = filter_common_word_obj.replace_common_word(tweet=x)
        list_tweet_replaced_with_common_word.append(temp_result)
    df["preprocessing_replace_common_word"] = list_tweet_replaced_with_common_word
    df["tweet"] = list_tweet_replaced_with_common_word

    # Preprocessing | Stop Word Remove
    stopword = StopWordRemoverFactory().create_stop_word_remover()
    list_tweet_stopword: List[str] = list()
    for x in df["tweet"]:
        stop_word_result: str = stopword.remove(text=x)
        list_tweet_stopword.append(stop_word_result)
    df["preprocessing_stopword"] = list_tweet_stopword
    df["tweet"] = list_tweet_stopword

    # Process Stemming
    stem_object = Stemming()
    result_stem_object: List[str] = stem_object.stem(
        tweets=df["tweet"], total_thread=total_thread_processor
    )
    df["preprocessing_stemming"] = result_stem_object
    df["tweet"] = result_stem_object

    # Preprocessing | Read Label Positive And Negative Words from Dictionary
    positive = pd.read_csv(filepath_or_buffer=file_name_positive_label, header=None)
    positive = positive[0].values.tolist()
    positive = "|".join(positive)

    negative = pd.read_csv(filepath_or_buffer=file_name_negative_label, header=None)
    negative = negative[0].values.tolist()
    negative = "|".join(negative)

    # Preproccesing | Remove Empty Value(NaN, None, NaT)
    df = df[pd.notnull(df["tweet"])]

    # Preprocessing | Weigh Tweet Positive Or Negative Sentiment
    df["positive"] = [
        len(re.findall(positive, x.lower()))
        / (1 if len(x.split()) == 0 else len(x.split()))
        for x in tqdm(df["tweet"])
    ]
    df["negative"] = [
        len(re.findall(negative, x.lower()))
        / (1 if len(x.split()) == 0 else len(x.split()))
        for x in tqdm(df["tweet"])
    ]

    # Preproccesing | Labeling Sentiment Column
    df["sentimen"] = [
        1 if df.iloc[x].positive >= df.iloc[x].negative else 0
        for x in range(df.shape[0])
    ]

    # Preprocessing | Tokenizer
    t = TokenizerLocal()
    tokens: List[List[str]] = t.create_token(tweets=(df["tweet"]).tolist())

    # Preprocessing | Removes Stopwords(Using Lib)
    f = Filter()
    df["tokens"] = [f.remove_stop_words(tokens=x) for x in tqdm(tokens)]
    df["tweet_final"] = [" ".join(x) for x in df["tokens"]]

    # Remove unused column
    df = df[
        [
            "date",
            "time",
            "preprocessing_case_folding",
            "preprocessing_replace_common_word",
            "preprocessing_stopword",
            "preprocessing_stemming",
            "username",
            "tweet_original",
            "tweet",
            "tweet_final",
            "tokens",
            "positive",
            "negative",
            "sentimen",
        ]
    ]
    df.to_csv("result_preprocess.csv")

    # Additional | Draw Sentiment Into Graph
    import matplotlib.pyplot as plt

    total_negative_tweets: int = len(
        [x for x in df["negative"] if x >= threshold_sentiment_negative]
    )
    total_positive_tweets: int = len(
        [x for x in df["positive"] if x >= threshold_sentiment_positive]
    )

    df_bar = pd.Series(
        data=[total_negative_tweets, total_positive_tweets],
        index=[
            "negative " + str(total_negative_tweets),
            "positive " + str(total_positive_tweets),
        ],
    )

    bar = plt.bar(df_bar.index, df_bar.values)
    bar[0].set_color("#EE204D")
    bar[1].set_color("#00FF00")
    plt.savefig("graph_sentiment_bar_chart.jpg")
    plt.clf()

    # Additional | Count Most Word
    combined_word: str = None
    for x in tqdm(df["tweet"]):
        combined_word = str(combined_word) + str(x)

    counter_result: Counter = Counter(combined_word.split())
    counter_result_most_common: List[tuple] = counter_result.most_common(10)
    print(counter_result_most_common)

    x_val = [x[0] for x in counter_result_most_common]
    y_val = [x[1] for x in counter_result_most_common]
    plt.plot(x_val, y_val)
    plt.plot(x_val, y_val, "or")
    plt.savefig("graph_most_common_word.jpg")
    plt.clf()


def training(
    total_row: int = 1000,
    embedding_dim: int = 100,
    batch_size: int = 100,
    num_epochs: int = 2,
) -> None:

    # Split data into test and train
    df = pd.read_csv(filepath_or_buffer="result_preprocess.csv", nrows=total_row)
    data_train, data_test = train_test_split(df, test_size=0.1, random_state=42)
    print(len(data_train), len(data_test))

    # Train
    all_training_words = [word for tokens in data_train["tokens"] for word in tokens]
    training_sentence_lengths = [len(tokens) for tokens in data_train["tokens"]]
    TRAINING_VOCAB = sorted(list(set(all_training_words)))
    print(
        "%s words total, with a vocabulary size of %s"
        % (len(all_training_words), len(TRAINING_VOCAB))
    )
    print("Max sentence length is %s" % max(training_sentence_lengths))

    # Test
    all_test_words = [word for tokens in data_test["tokens"] for word in tokens]
    test_sentence_lengths = [len(tokens) for tokens in data_test["tokens"]]
    TEST_VOCAB = sorted(list(set(all_test_words)))
    print(
        "%s words total, with a vocabulary size of %s"
        % (len(all_test_words), len(TEST_VOCAB))
    )
    print("Max sentence length is %s" % max(test_sentence_lengths))

    # Load Word2Vec Indonesia
    word2vec = models.word2vec.Word2Vec.load("./idwiki_word2vec_200_new_lower.model")

    # Get Embed
    training_embeddings = get_word2vec_embeddings(
        word2vec, data_train, generate_missing=True
    )

    # MAX_SEQUENCE_LENGTH = 50
    MAX_SEQUENCE_LENGTH = 50
    EMBEDDING_DIM = embedding_dim

    ### Tokenize and Pad sequences
    tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
    tokenizer.fit_on_texts(data_train["tokens"].tolist())
    training_sequences = tokenizer.texts_to_sequences(data_train["tokens"].tolist())

    train_word_index = tokenizer.word_index
    print("Found %s unique tokens." % len(train_word_index))

    train_cnn_data = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    train_embedding_weights = np.zeros((len(train_word_index) + 1, EMBEDDING_DIM))
    for word, index in train_word_index.items():
        train_embedding_weights[index, :] = (
            word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
        )
    print(train_embedding_weights.shape)

    test_sequences = tokenizer.texts_to_sequences(data_test["tokens"].tolist())
    # print("Test Sequence : ", test_sequences)
    test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    label_names = ["positive", "negative"]
    # label_names: List[str] = ['sentimen']
    y_train = data_train[label_names].values
    print("y_train >> ", y_train)
    x_train = train_cnn_data
    y_tr = y_train

    model = ConvNet(
        train_embedding_weights,
        MAX_SEQUENCE_LENGTH,
        len(train_word_index) + 1,
        EMBEDDING_DIM,
        len(list(label_names)),
    )

    # Train CNN
    ## Call Back
    from keras import callbacks

    callbacks = [
        callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor="val_loss",
            # "no longer improving" being defined as "no better than 1e-2 less"
            min_delta=1e-2,
            # "no longer improving" being further defined as "for at least 2 epochs"
            patience=2,
            verbose=1,
        )
    ]
    hist = model.fit(
        x_train,
        y_tr,
        epochs=num_epochs,
        validation_split=0.2,
        shuffle=True,
        batch_size=batch_size,
        callbacks=callbacks,
    )
    model.save("./model")
    predictions = model.predict(test_cnn_data, batch_size=batch_size, verbose=1)

    labels = [1, 0]

    prediction_labels = []
    for p in predictions:
        prediction_labels.append(labels[np.argmax(p)])

    print(
        ">> Predicted Result >> ",
        sum(data_test.sentimen == prediction_labels) / len(prediction_labels),
    )
    print(data_test.sentimen.value_counts())

    # Confusion Matrix
    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = np.argmax(predictions, axis=1)
    print("Confusion Matrix")
    print(confusion_matrix(data_test.sentimen, y_pred))
    target_names = ["positif", "negatif"]
    print(classification_report(data_test.sentimen, y_pred, target_names=target_names))


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list) < 1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [
            vector[word] if word in vector else np.random.rand(k)
            for word in tokens_list
        ]
    else:
        vectorized = [
            vector[word] if word in vector else np.zeros(k) for word in tokens_list
        ]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged


def get_word2vec_embeddings(vectors, clean_comments, generate_missing=False):
    embeddings = clean_comments["tokens"].apply(
        lambda x: get_average_word2vec(x, vectors, generate_missing=generate_missing)
    )
    return list(embeddings)


def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):

    embedding_layer = Embedding(
        num_words,
        embedding_dim,
        weights=[embeddings],
        input_length=max_sequence_length,
        trainable=False,
    )

    sequence_input = Input(shape=(max_sequence_length,), dtype="int32")
    embedded_sequences = embedding_layer(sequence_input)

    convs = []
    filter_sizes = [2, 3, 4, 5, 6]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=200, kernel_size=filter_size, activation="relu")(
            embedded_sequences
        )
        l_pool = GlobalMaxPooling1D()(l_conv)
        convs.append(l_pool)

    l_merge = concatenate(convs, axis=1)

    x = Dropout(0.1)(l_merge)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    preds = Dense(labels_index, activation="sigmoid")(x)

    model = Model(sequence_input, preds)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
    model.summary()
    return model


def load_model(total_row=29000, filename: str = None):
    import keras
    from keras.preprocessing.text import Tokenizer

    df = pd.read_csv(filepath_or_buffer=filename, nrows=total_row)
    data_train, data_test = train_test_split(df, test_size=0.1, random_state=42)

    all_training_words = [word for tokens in data_train["tokens"] for word in tokens]
    TRAINING_VOCAB = sorted(list(set(all_training_words)))

    tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
    tokenizer.fit_on_texts(data_train["tokens"].tolist())

    model = keras.models.load_model("./model")

    test_sequences = tokenizer.texts_to_sequences(data_test["tokens"].tolist())
    MAX_SEQUENCE_LENGTH = 50
    test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    predictions = model.predict(test_cnn_data, batch_size=1000, verbose=1)
    labels = [1, 0]
    prediction_labels = []
    for p in predictions:
        prediction_labels.append(labels[np.argmax(p)])

    print(
        ">> Hasil >> ",
        sum(data_test.sentimen == prediction_labels) / len(prediction_labels),
    )
    print("Total Data : ", data_test.sentimen.value_counts())

    # Confusion Matrix
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    import matplotlib as plt
    y_pred = np.argmax(predictions, axis=1)
    print("Confusion Matrix")
    x1, x2, y1, y2 = (confusion_matrix(y_true=data_test.sentimen.tolist(), y_pred=y_pred, labels=[0,1])).ravel()
    print("FN : ", x1)
    print("TN : ", x2)
    print("TP : ", y1)
    print("FP : ", y2)
    target_names = ["Positive", "Negative"]
    print(classification_report(y_true=data_test.sentimen, y_pred=y_pred, labels=[0,1], target_names=target_names))

    # # Visualize | Confussion Matrix
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm).plot()

    # # Visualize | Bar
    # from visualize.chart import Chart

    # c = Chart()
    # c.bar(label_positive=, predict_positive=, label_negative=, predict_negative=)


def write_vector_to_csv(filename: str = None) -> None:
    df = pd.read_csv(filepath_or_buffer=filename)
    df["preprocess_vector"] = df["tokens"]
    print("preprocess_vector")
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    tokenizer = Tokenizer(num_words=100000)
    tokenizer.fit_on_texts((df["preprocess_vector"]).tolist())
    sequences = tokenizer.texts_to_sequences((df["preprocess_vector"]).tolist())
    df["preprocess_vector"] = sequences
    df = df[
        [
            "date",
            "time",
            "preprocessing_case_folding",
            "preprocessing_replace_common_word",
            "preprocessing_stopword",
            "preprocessing_stemming",
            "username",
            "tweet_original",
            "tweet",
            "tweet_final",
            "tokens",
            "positive",
            "negative",
            "sentimen",
            "preprocess_vector"
        ]
    ]
    df.to_csv("result_preprocess.csv")

def preprocess_weigh_label(filename: str="result_preprocess.csv", total_thread: int=4):
    # Read
    df = pd.read_csv(filepath_or_buffer=filename)
    
    # Positive Negative Words
    file_name_positive_label: str = "./positive_label_word.txt"
    file_name_negative_label: str = "./negative_label_word.txt"

    # Preprocessing | Read Label Positive And Negative Words from Dictionary
    positive = pd.read_csv(filepath_or_buffer=file_name_positive_label, header=None)
    positive = positive[0].values.tolist()
    positive = "|".join(positive)

    negative = pd.read_csv(filepath_or_buffer=file_name_negative_label, header=None)
    negative = negative[0].values.tolist()
    negative = "|".join(negative)

    # Temporary | Remove Numeric
    # clean_tweet: List[str] = list()
    # for x in df["tweet_final"]:
    #     clean_tweet.append(_remove_numeric(str(x)))
    # df["tweet_final"] = clean_tweet

    # Temporary | Join Tweet Final 
    # tweet_final_temp: List[str] = list()
    # for index, x in enumerate(df.tweet_final):
    #     x = (x.replace("[", "").replace("]", "").replace("\"", "").replace("\'", "").replace(",", ""))
    #     tweet_final_temp.append(x)
    # df.tweet_final = tweet_final_temp

    # Preproccesing | Remove Empty Value(NaN, None, NaT)
    df = df[pd.notnull(df["tweet_final"])]

    # Preprocessing | Weigh Tweet Positive Or Negative Sentiment
    df["positive"] = [
        len(re.findall(positive.lower(), x.lower()))
        / (1 if len(x.split()) == 0 else len(x.split()))
        for x in tqdm(df["tweet_final"])
    ]

    df["negative"] = [
        len(re.findall(negative.lower(), x.lower()))
        / (1 if len(x.split()) == 0 else len(x.split()))
        for x in tqdm(df["tweet_final"])
    ]

    # Preproccesing | Labeling Sentiment Column
    df["sentimen"] = [
        1 if df.iloc[x].positive > df.iloc[x].negative else 0
        for x in tqdm(range(df.shape[0]))
    ]

    df = df[
        [
            "date",
            "time",
            "preprocessing_case_folding",
            "preprocessing_replace_common_word",
            "preprocessing_stopword",
            "preprocessing_stemming",
            "username",
            "tweet_original",
            "tweet",
            "tweet_final",
            "tokens",
            "positive",
            "negative",
            "sentimen",
        ]
    ]

    df.to_csv("result_preprocess.csv")

def _remove_numeric(tweet: str= None) -> str: 
    pattern = '[0-9]'
    return [re.sub(pattern, '', i) for i in tweet.split()] 
  
if __name__ == "__main__":
    start_time = time.time()
    #preprocess(total_row=12000, total_thread_processor=4)
    #preprocess_weigh_label()
    write_vector_to_csv(filename="result_preprocess.csv")
    # training(total_row=10000, embedding_dim=8, batch_size=64, num_epochs=3)
    #load_model(total_row=10000, filename="result_preprocess.csv")

    print("--- %s seconds ---" % (time.time() - start_time))
