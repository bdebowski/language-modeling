import torch
from transformers import *


class WordPredictor:
    def __init__(self, model_name):
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self._m = None

    def top_n_next(self, n):
        return [("joe", 0.123), ("karen", 0.191), ("magic", 0.34)]

    def feed(self, text):
        pass
