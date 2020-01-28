from enum import Enum

from transformers import *


class ModelName(Enum):
    DISTILBERT = 1


class ModelFactory:
    def __init__(self):
        self._CREATION_FXS = {
            ModelName.DISTILBERT: ModelFactory._create_distilbert}

    def create_from_name(self, model_name: ModelName):
        """
        Returns a tokenizer and language model for the model name specified.
        """
        return self._CREATION_FXS[model_name]()

    @staticmethod
    def _create_distilbert():
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
        model.eval()
        model.to("cuda")

        return tokenizer, model
