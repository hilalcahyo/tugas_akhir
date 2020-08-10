from preprocess.crawl import Crawl
import schedule
import time
import datetime


def main(search_key:str = None, file_name:str = None, since:datetime.datetime = None, until:datetime.datetime = None, limit_tweet: int = 0):
    print("I'm working...", datetime.datetime.now())
    
    crawl_object = Crawl()
    is_created: bool = crawl_object.crawl_and_save(
        word=search_key, file_name=file_name, since=x, until=y, limit=limit_tweet
    )
if __name__ == "__main__":
    x = datetime.datetime.now() - datetime.timedelta(days=220)
    y = datetime.datetime.now()
    print(x)
    print(y)
    x = x.strftime("%Y-%m-%d %H:%M:%S")
    y = y.strftime("%Y-%m-%d %H:%M:%S")
    main(search_key="Banjir Jakarta", file_name="crawl_3.csv", since=x, until=y, limit_tweet= 5)
