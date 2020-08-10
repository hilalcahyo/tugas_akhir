# Tugas Akhir

This app can analyze sentiment twitter(for example: "Banjir Jakarta")

## To Do
- [x] Crawling
- [x] Drop Duplicate
- [x] Filter Unused Character
- [x] Normalize Common Word
- [x] Stop Word Remove
- [x] Stemming using multiprocess
- [x] Count Most Word
- [x] Labelling Tweets With Positive, Negative, And Neutral Sentiment
- [x] Draw A Graph Of Common Word
- [x] Draw A Graph Of Sentiment
- [ ] Bag Of Word Model
- [ ] Bernoulli Naive Bayes Model Train
- [ ] Classification Comparation
- [ ] WordCloud
- [ ] Bernoulli Predict
- [ ] Visualtation Of Bernoulli Predict

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
$ git clone https://github.com/hilalcahyo/tugas_akhir
$ cd tugas_akhir
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
$ pip install -r requirements.txt
```
## Requirements

- Python 3.6;
- pandas;
- twint;

### And coding style tests

We use black(python library) for code formater

```
$ pip install black
$ python ./black
```

### How To Run

1. Change Crawling Process Variable in run_crawl.py
    * 1.1 search key(sample file output **crawl_result.csv**)
    * 1.2 limit
2. change Sentiment Analysis in index.py
    * 2.1 file name(sample file to load **crawl_result.csv**)

```
$ python run_crawl.py
$ python index.py
```

## Built With

* [Pandas](https://pandas.pydata.org/) - Pandas
* [Twint](https://github.com/twintproject/twint) - Twitter scraping & OSINT

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Hilal Cahyo** - [HilalCahyo](https://github.com/hilalcahyo)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Inspiration from mediana saraswati (https://github.com/medianasaraswati)
