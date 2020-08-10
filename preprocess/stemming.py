from typing import List
from multiprocessing import Pool, TimeoutError, cpu_count
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import tqdm
import threading


class Stemming:
    def __init__(self):
        pass

    def stem(self, tweets: List[str] = None, total_thread: int = 1) -> List[str]:
        print(len(tweets))
        stemmer = StemmerFactory().create_stemmer()
        threads = list()
        print("Total Thread", total_thread)
        with Pool(processes=total_thread) as pool:
            result = list(tqdm.tqdm(pool.imap(stemmer.stem, tweets), total=len(tweets)))
            return result
