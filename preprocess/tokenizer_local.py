import tqdm
import nltk
from typing import List
from nltk.tokenize import TweetTokenizer


class TokenizerLocal:
    def __init__(self):
        pass

    def create_token(self, tweets: List[str] = list()) -> List[List[str]]:
        # nltk.download('punkt')
        tknzr = TweetTokenizer()
        tokens = [tknzr.tokenize(text=x) for x in tweets]
        print(">>> ", tokens[0])
        return tokens
