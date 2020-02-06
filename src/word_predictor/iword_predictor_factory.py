from enum import Enum

from transformers import BertTokenizer, BertForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel

from src.word_predictor.bert_word_predictor import BertWordPredictor
from src.word_predictor.gpt2_word_predictor import Gpt2WordPredictor


class ModelName(Enum):
    BERT_UNCASED = 1
    BERT_LARGE_CASED = 2
    GPT2 = 3
    GPT2_MEDIUM = 4
    GPT2_LARGE = 5
    GPT2_XLARGE = 6


class IWordPredictorFactory:
    def __init__(self):
        self._CREATION_FXS = {
            ModelName.BERT_UNCASED: IWordPredictorFactory._create_bert_uncased,
            ModelName.BERT_LARGE_CASED: IWordPredictorFactory._create_bert_large_cased,
            ModelName.GPT2: IWordPredictorFactory._create_gpt2,
            ModelName.GPT2_MEDIUM: IWordPredictorFactory._create_gpt2_medium,
            ModelName.GPT2_LARGE: IWordPredictorFactory._create_gpt2_large,
            ModelName.GPT2_XLARGE: IWordPredictorFactory._create_gpt2_xlarge}

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

        return BertWordPredictor(tokenizer, model, mem_length=256)

    @staticmethod
    def _create_bert_large_cased():
        tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
        model = BertForMaskedLM.from_pretrained("bert-large-cased")
        model.eval()
        model.to("cuda")

        return BertWordPredictor(tokenizer, model, mem_length=256)

    @staticmethod
    def _create_gpt2():
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()
        model.to("cuda")

        return Gpt2WordPredictor(tokenizer, model)

    @staticmethod
    def _create_gpt2_medium():
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        model.eval()
        model.to("cuda")

        return Gpt2WordPredictor(tokenizer, model)

    @staticmethod
    def _create_gpt2_large():
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        model = GPT2LMHeadModel.from_pretrained("gpt2-large")
        model.eval()
        model.to("cuda")

        return Gpt2WordPredictor(tokenizer, model)

    @staticmethod
    def _create_gpt2_xlarge():
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
        model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
        model.eval()
        model.to("cuda")

        return Gpt2WordPredictor(tokenizer, model)
