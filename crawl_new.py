from preprocess.crawl import Crawl
import schedule
import time
import datetime


def job():
    print("I'm working...", datetime.datetime.now())
    x = datetime.datetime.now() - datetime.timedelta(minutes=5)
    x = x.strftime("%Y %m %d %H:%M:%S")
    file_name: str = "crawl_1.csv"
    crawl_object = Crawl()
    is_created: bool = crawl_object.crawl_and_save(
        word="Banjir Jakarta", file_name=file_name, since=x
    )


schedule.every(5).minutes.do(job)

while 1:
    schedule.run_pending()
    time.sleep(1)
