import twint
from typing import List
import asyncio

class Crawl:
    def __init__(self):
        pass

    def crawl_and_save(self, word: str = None, file_name: str = None) -> bool:
        result: bool = False
        if word:
            print("TEST", word, file_name)
            # Config
            c = twint.Config()
            c.Search = word
            c.Since = "2020-01-01 00:00:00"
            c.Until = "2020-08-08 00:00:00"
            c.Limit = 100
            c.Store_csv = True
            c.Output = file_name

            # Run
            twint.run.Search(c)
            result = True
        return result