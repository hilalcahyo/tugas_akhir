from typing import List
import re
import pandas as pd


class Filter:
    def __init__(self):
        pass

    def filter_unused_character(self, df: pd.DataFrame = None) -> pd.DataFrame:
        print("ccccc", df)
        df["tweet"] = df["tweet"].str.replace("(https?://[\w\.\/]*)", "")  # http
        df["tweet"] = df["tweet"].str.replace(
            "(?:&(?:lt|nbsp|amp|gt);)", ""
        )  # lt,nbsp,gt,amp
        df["tweet"] = df["tweet"].str.replace("(@|#)\w+", "")  # @ dan #
        df["tweet"] = df["tweet"].str.replace(
            "[^A-Za-z0-9\s\-\/]", ""
        )  # selain huruf, spasi dan strip
        df["tweet"] = df["tweet"].str.replace("(\-|\/)", " ")  # -
        df["tweet"] = df["tweet"].str.replace("\n", " ")  # enter
        df["tweet"] = df["tweet"].str.replace("\s{2,}", " ")  # spasi lebih dari 2
        df["tweet"] = df["tweet"].str.replace("^rt.*", "")  # Remove RT tweet
        df["tweet"] = df["tweet"].str.replace("\.\.", " ")
        df["tweet"] = df["tweet"].str.replace(
            "pictwittercom", ""
        )  # remove pictwittercom
        # df.dropna(subset=["tweet"], inplace=True)  # Remove Empty cell
        return df

    def replace_common_word(self, tweet: str = None) -> str:
        data = re.sub(r"\bmalas\b", "males", tweet)
        data = re.sub(
            r"\bgue\b|\bgua\b|\bw\b|\baku\b|\bku\b|\bgw\b|\beug\b|\bsy\b|\bacu\b",
            "saya",
            data,
        )
        data = re.sub(r"\bst\b|\bstsiun\b|\bsta\b", "stasiun", data)
        data = re.sub(r"\bsmpe\b", "sampai", data)
        data = re.sub(r"\bbpk\b", "bapak", data)
        data = re.sub(r"\btks\b|\bthanks\b|\btengkyu\b", "makasih", data)
        data = re.sub(r"\byg\b", "yang", data)
        data = re.sub(r"\bjlr\b", "jalur", data)
        data = re.sub(r"\bhuft\b", "parah", data)
        data = re.sub(r"\bzonk\b", "jelek", data)
        data = re.sub(r"\bpkl\b|\bpukul\b", "", data)
        data = re.sub(r"\bgua\b|\bgue\b|\bw\b|\bgw\b", "saya", data)
        data = re.sub(r"\bselamat\b", "", data)
        data = re.sub(r"\bmakasih\b", "", data)
        data = re.sub(r"\byg\b", "yang", data)
        data = re.sub(r"\bmandek\b", "ketahan", data)
        data = re.sub(r"\bzonk\b", "jelek", data)
        data = re.sub(r"\bga\b|\bgak\b|\bengga\b", "enggak", data)
        data = re.sub(r"\bpeople\b", "orang", data)
        data = re.sub(r"\bnjir\b", "anjing", data)
        data = re.sub(r"\bDDT\b|\bddt\b", "double double track", data)
        data = re.sub(r"\bkzl\b|\bkesal\b", "kesel", data)

        return data
