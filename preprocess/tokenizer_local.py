import tqdm
import nltk
from nltk import WordNetLemmatizer, word_tokenize
from typing import List


class TokenizerLocal:
    def __init__(self):
        pass

    def create_token(self, tweets: List[str] = list()) -> List[List[str]]:
        nltk.download('punkt')
        tokens = [word_tokenize(text=x) for x in tqdm(tweets)]
        print(tokens[0])
        return tokens
