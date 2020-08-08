import twint
from typing import List
import asyncio
from datetime import datetime


class Crawl:
    def __init__(self):
        pass

    def crawl_and_save(
        self,
        word: str = None,
        file_name: str = None,
        limit: int = 3200,
        since: datetime = None,
        until: datetime = None,
    ) -> bool:
        result: bool = False
        if word:
            print("TEST", word, file_name)
            # Config
            c = twint.Config()
            c.Search = word
            if until != None and since != None:
                c.Since = since
                c.Until = until
            c.Limit = limit
            c.Store_csv = True
            c.Output = file_name

            # Run
            twint.run.Search(c)
            result = True
        return result
