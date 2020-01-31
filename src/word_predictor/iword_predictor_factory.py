from enum import Enum

from transformers import BertTokenizer, BertForMaskedLM

from src.word_predictor.bert_word_predictor import BertWordPredictor


class ModelName(Enum):
    BERT_UNCASED = 1
    BERT_LARGE_CASED = 2


class IWordPredictorFactory:
    def __init__(self):
        self._CREATION_FXS = {
            ModelName.BERT_UNCASED: IWordPredictorFactory._create_bert_uncased,
            ModelName.BERT_LARGE_CASED: IWordPredictorFactory._create_bert_large_cased}

    def create_from_name(self, model_name: ModelName):
        """
        Returns an IWordPredictor model for the model name specified.
        """
        return self._CREATION_FXS[model_name]()

    @staticmethod
    def _create_bert_uncased():
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        model.eval()
        model.to("cuda")

        return BertWordPredictor(tokenizer, model, mem_length=512)

    @staticmethod
    def _create_bert_large_cased():
        tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
        model = BertForMaskedLM.from_pretrained("bert-large-cased")
        model.eval()
        model.to("cuda")

        return BertWordPredictor(tokenizer, model, mem_length=512)
