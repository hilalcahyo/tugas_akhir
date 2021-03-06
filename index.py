import sys
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
import keras.backend as K
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
    LSTM,
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


def preprocess(total_row: int = 10000, total_thread_processor: int = 2, file_name_input: str=None) -> None:
    # Positive Negative Words
    file_name_positive_label: str = "positif.txt"
    file_name_negative_label: str = "negatif.txt"

    # Choose Your Preferable Processor Capability
    total_thread_processor: int = total_thread_processor

    df = pd.read_csv(filepath_or_buffer=file_name_input, nrows=total_row,sep=',', engine='python')
    df.drop_duplicates(subset="tweet", keep="first", inplace=True)

    # Remove unused column
    df = df[
        [
            "date",
            "time",
            "username",
            "tweet",
        ]
    ]

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


def training(
    total_row: int = 1000,
    embedding_dim: int = 100,
    batch_size: int = 128,
    num_epochs: int = 2,
    filename: str = None
) -> None:

    # Split data into test and train
    df = pd.read_csv(filepath_or_buffer=filename, nrows=total_row)
    data_train, data_test = train_test_split(df, test_size=0.1, random_state=42)
    print(len(data_train), len(data_test))

    # Make Dictionary Data Train
    # all_training_words = [word for tokens in data_train["tokens"] for word in tokens]
    all_training_words = [
        word for tokens in data_train["tokens"] for word in eval(tokens)
    ]
    training_sentence_lengths = [len(eval(tokens)) for tokens in data_train["tokens"]]
    TRAINING_VOCAB = sorted(list(set(all_training_words)))
    print(
        "%s words total, with a vocabulary size of %s"
        % (len(all_training_words), len(TRAINING_VOCAB))
    )
    print("Max sentence length is %s" % max(training_sentence_lengths))

    # Make Dictionary Data Test
    all_test_words = [word for tokens in data_test["tokens"] for word in eval(tokens)]
    test_sentence_lengths = [len(eval(tokens)) for tokens in data_test["tokens"]]
    TEST_VOCAB = sorted(list(set(all_test_words)))
    print(
        "%s words total, with a vocabulary size of %s"
        % (len(all_test_words), len(TEST_VOCAB))
    )
    print("Max sentence length is %s" % max(test_sentence_lengths))

    # Load Word2Vec Indonesia
    word2vec = models.word2vec.Word2Vec.load("./idwiki_word2vec_200.model")

    # Projected Token To Vector Using Word2Vec
    training_embeddings = get_word2vec_embeddings(
        word2vec, data_train, generate_missing=True
    )
    import math

    # MAX_SEQUENCE_LENGTH = 50 menyamakan panjang twitter
    MAX_SEQUENCE_LENGTH = max(training_sentence_lengths)
    MAX_SEQUENCE_LENGTH = int(math.ceil((MAX_SEQUENCE_LENGTH) / 10.0)) * 10
    EMBEDDING_DIM = embedding_dim

    ### Tokenize and Add Padding sequences
    tokenizer = Tokenizer(
        num_words=len(TRAINING_VOCAB),
        lower=True,
        char_level=False,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    )
    tokenizer.fit_on_texts(data_train["tokens"].tolist())
    training_sequences = tokenizer.texts_to_sequences(data_train["tokens"].tolist())
    print("training_sequences(MAX) : ", max(training_sequences))

    train_word_index = tokenizer.word_index
    print("Found %s unique tokens." % len(train_word_index))

    train_cnn_data = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print("train_cnn_data size ", len(train_cnn_data))
    print("train_cnn_data ", train_cnn_data)
    train_embedding_weights = np.zeros((len(train_word_index) + 1, EMBEDDING_DIM))
    print("train_embedding_weights size ", len(train_embedding_weights))
    print("train_embedding_weights ", train_embedding_weights)

    for word, index in train_word_index.items():
        train_embedding_weights[index, :] = (
            word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
        )
    print(train_embedding_weights.shape)

    test_sequences = tokenizer.texts_to_sequences(data_test["tokens"].tolist())
    test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print(" test_cnn_data ", test_cnn_data)

    # Choose Column For Weighting Tokens
    label_names = ["positive", "negative"]
    y_train = data_train[label_names].values
    x_train = train_cnn_data
    y_tr = y_train

    print("$ train_embedding_weights ", train_embedding_weights)
    print("$ MAX_SEQUENCE_LENGTH ", MAX_SEQUENCE_LENGTH)
    print("$ len(train_word_index) + 1 ", len(train_word_index) + 1)
    print("$ EMBEDDING_DIM ", EMBEDDING_DIM)
    print("$ len(list(label_names)) ", len(list(label_names)))

    model = ConvNet(
        train_embedding_weights,
        MAX_SEQUENCE_LENGTH,
        len(train_word_index) + 1,
        EMBEDDING_DIM,
        len(list(label_names)),
    )
    # sys.exit()

    from keras import callbacks

    # If epoch doesnt improve then stop
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
    # Visualize Train Model
    model.fit(
        x_train,
        y_tr,
        epochs=num_epochs,
        validation_split=0.2,
        shuffle=True,
        batch_size=batch_size,
        #callbacks=callbacks,
    )
    # Save Model To File
    model.save("./model")

    # Predicted Model
    predictions = model.predict(test_cnn_data, batch_size=batch_size, verbose=1)

    # Mapping/Normalization Prediction(0..1)
    labels = [1, 0]
    prediction_labels = []
    for p in predictions:
        prediction_labels.append(labels[np.argmax(p)])

    # Show Accuracy & Compare prediction
    print(
        ">> Predicted Result >> ",
        sum(data_test.sentimen == prediction_labels) / len(prediction_labels),
    )
    print(data_test.sentimen.value_counts())

    # Confusion Matrix
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sn
    import matplotlib.pyplot as plt
    y_pred = np.argmax(predictions, axis=1)
    print("Confusion Matrix")
    (tn, fp, fn, tp) = (
        confusion_matrix(
            y_true=data_test.sentimen.tolist(), y_pred=prediction_labels, labels=[1, 0]
        )
    ).ravel()
    cf_matrix_train = (
        confusion_matrix(
            y_true=data_test.sentimen.tolist(), y_pred=prediction_labels, labels=[1, 0]
        )
    )
    print("(tn, fp, fn, tp)", (tn, fp, fn, tp))
    target_names = ["Positive", "Negative"]
    print(
        classification_report(
            y_true=data_test.sentimen,
            y_pred=prediction_labels,
            labels=[1, 0],
            target_names=target_names,
        )
    )
    df_cf_matrix_train = pd.DataFrame(cf_matrix_train, index = [i for i in ["Negative Tweets", "Positive Tweets"]],
    columns = [i for i in ["Negative Tweets", "Positive Tweets"]])

    plt.figure(figsize = (10,7))
    sn.heatmap(cf_matrix_train, annot=True, cbar=False, fmt="d")
    plt.savefig('confusion_matrix_train.jpg')

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
    # input_dim: Integer. Size of the vocabulary, i.e. maximum integer index + 1.
    # output_dim: Integer. Dimension of the dense embedding.
    # input_length: Length of input sequences, when it is constant. This argument is required if you are going to connect Flatten then Dense layers upstream (without it, the shape of the dense outputs cannot be computed).
    embedding_layer = Embedding(
        input_dim=num_words,
        output_dim=embedding_dim,
        weights=[embeddings],
        input_length=max_sequence_length,
        trainable=False,
    )

    sequence_input = Input(shape=(max_sequence_length,), dtype="int64")
    embedded_sequences = embedding_layer(sequence_input)

    convs = []
    filter_sizes = [2, 4, 8, 16]

    for filter_size in filter_sizes:
        l_conv = Conv1D(
            filters=embedding_dim, kernel_size=filter_size, activation="relu"
        )(embedded_sequences)
        l_pool = GlobalMaxPooling1D()(l_conv)
        convs.append(l_pool)

    l_merge = concatenate(convs, axis=1)
    x = Dropout(0.1)(l_merge)
    x = Dense(200, activation="relu")(x)
    #x = Dense(128, activation="relu")(x)
    #x = Dense(8, activation="relu")(x)
    preds = Dense(labels_index, activation="sigmoid")(x)

    model = Model(sequence_input, preds)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
    model.summary()
    return model


def load_model(total_row=29000, filename: str = None):
    import keras
    from keras.preprocessing.text import Tokenizer

    # Word Cloud
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        ConfusionMatrixDisplay,
    )

    df = pd.read_csv(filepath_or_buffer=filename, nrows=total_row)
    data_train, data_test = train_test_split(df, test_size=0.9, random_state=42)

    all_training_words = [word for tokens in data_test["tokens"] for word in eval(tokens)]
    TRAINING_VOCAB = sorted(list(set(all_training_words)))

    tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
    tokenizer.fit_on_texts(data_test["tokens"].tolist())

    model = keras.models.load_model("./model")

    test_sequences = tokenizer.texts_to_sequences(data_test["tokens"].tolist())
    MAX_SEQUENCE_LENGTH = 60
    test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    predictions: List[List[float, float]] = model.predict(test_cnn_data, batch_size=1000, verbose=1)
    labels = [1, 0]
    prediction_labels: List[int] = []
    print("$ Total Prediction : ", len(predictions))
    print("Label : [1,0] =====> [positif , negatif]")
    print("Sample prediction without normalize", predictions[0:5])
    for p in predictions:
        prediction_labels.append(labels[np.argmax(p)])
    print("Sample prediction normalize", prediction_labels[0:5])
    print("Data Set Validation -  Sentimen Value :", (data_test.sentimen.tolist())[0:5])
    print("Data Set Validation -  tweet Value :", (data_test.tweet_final.tolist())[0:5])
    print(
        "Accuracy: ",
        sum(data_test.sentimen == prediction_labels) / len(prediction_labels),
    )
    print("Total Data : ", data_test.sentimen.value_counts())
    print("Total prediction", len(prediction_labels))

    # Confusion Matrix
    y_pred = np.argmax(predictions, axis=1)
    print("Confusion Matrix")
    (tn, fp, fn, tp) = (
        confusion_matrix(
            y_true=data_test.sentimen.tolist(), y_pred=prediction_labels, labels=[1, 0]
        )
    ).ravel()
    cf_matrix_testing = (
        confusion_matrix(
            y_true=data_test.sentimen.tolist(), y_pred=prediction_labels, labels=[1, 0]
        )
    )
    print("(tp, fn, fp, tn)", (tn, fp, fn, tp))
    target_names = ["Positive", "Negative"]
    print(
        classification_report(
            y_true=data_test.sentimen,
            y_pred=prediction_labels,
            labels=[1, 0],
            target_names=target_names,
        )
    )
    import seaborn as sn
    df_cf_matrix_testing = pd.DataFrame(cf_matrix_testing, index = [i for i in ["Negative Tweets", "Positive Tweets"]],
    columns = [i for i in ["Negative Tweets", "Positive Tweets"]])

    plt.figure(figsize = (10,7))
    sn.heatmap(cf_matrix_testing, annot=True, cbar=False, fmt="d")
    plt.savefig('confusion_matrix_validation.jpg')


    # Wordcloud
    from wordcloud import WordCloud, STOPWORDS

    comment_words = ""
    stopwords = set(STOPWORDS)
    comment_words += " ".join(df.tokens) + " "
    print(len(comment_words))
    wordcloud = WordCloud(
        width=800,
        height=800,
        background_color="white",
        stopwords=stopwords,
        min_font_size=10,
        collocations=False,
    ).generate(comment_words)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig("graph_word_cloud.png")
    plt.close()

    # # Visualize | Confussion Matrix
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm).plot()

    # # Visualize | Bar
    # from visualize.chart import Chart

    # c = Chart()
    # c.bar(label_positive=, predict_positive=, label_negative=, predict_negative=)


# tampilan bab 4
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
            "preprocess_vector",
        ]
    ]
    df.to_csv("result_preprocess_old.csv")


def preprocess_weigh_label(
    filename: str = "result_preprocess_old.csv", total_thread: int = 4
):
    # Read
    df = pd.read_csv(filepath_or_buffer=filename)

    # Positive Negative Words
    file_name_positive_label: str = "./positif_ta2.txt"
    file_name_negative_label: str = "./negatif_ta2.txt"

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

    df.to_csv("result_preprocess_old.csv")


def _remove_numeric(tweet: str = None) -> str:
    pattern = "[0-9]"
    return [re.sub(pattern, "", i) for i in tweet.split()]


def draw_sentiment_and_most_word(filename: str = None, total_row: int = 0):
    import matplotlib.pyplot as plt

    df = pd.read_csv(filepath_or_buffer=filename, nrows=total_row)

    # Additional | Most Word
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

    # Additional | Draw Sentiment Into Graph

    total_negative_tweets: int = len([x for x in df["sentimen"] if x == 0])
    total_positive_tweets: int = len([x for x in df["sentimen"] if x != 0])

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

def balancing_csv_by_sentimen(filename: str = None, output_file: str=None):
    df = pd.read_csv(filepath_or_buffer=filename)
    df_sentimen_positif = df.loc[df['sentimen'] == 1]
    df_sentimen_negatif = df.loc[df['sentimen'] == 0]
    print(df_sentimen_positif.shape[0])
    print(df_sentimen_negatif.shape[0])
    minimal_row = min(int(df_sentimen_positif.shape[0]), int(df_sentimen_negatif.shape[0]))
    df_sentimen_negatif = df_sentimen_negatif[0:minimal_row]
    df_sentimen_positif = df_sentimen_positif[0:minimal_row]
    print(df_sentimen_positif.shape[0])
    print(df_sentimen_negatif.shape[0])
    combined_df = pd.concat([df_sentimen_positif, df_sentimen_negatif])
    print(combined_df.shape)
    combined_df.to_csv(output_file)

if __name__ == "__main__":
    start_time = time.time()
    #balancing_csv_by_sentimen(filename="dataset_train_and_validation.csv",output_file="dataset_train_and_validation_balance.csv")
    #balancing_csv_by_sentimen(filename="dataset_test.csv",output_file="dataset_test_balance.csv")
    # draw_sentiment_and_most_word(total_row=10000, filename="result_preprocess_old.csv")
    #preprocess(total_row=87000, total_thread_processor=4,file_name_input="crawl_newest.csv")
    # preprocess_weigh_label()
    # write_vector_to_csv(filename="result_preprocess_old.csv")
    #training(total_row=2776, embedding_dim=100, batch_size=8, num_epochs=10, filename="dataset_train_and_validation_balance.csv")
    load_model(total_row=716, filename="dataset_test_balance.csv")
    print("--- %s seconds ---" % (time.time() - start_time))
