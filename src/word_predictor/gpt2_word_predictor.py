import torch

from src.word_predictor.iword_predictor import IWordPredictor
from src.util.rotating_sequence import RotatingSequence


class Gpt2WordPredictor(IWordPredictor):
    def __init__(self, tokenizer, model):
        self._tokenizer = tokenizer
        self._lm = model
        self._past = None
        self._model_output = None

        # For debugging
        #self._tokens_history = RotatingSequence(256)

    def feed(self, text, **kwargs):
        # todo: how to handle new lines?
        prefix_space = self._past is not None

        tokens = self._tokenizer.encode(text, add_prefix_space=prefix_space)
        #for t in tokens:
        #    self._tokens_history.insert(t)

        tokens_tensor = torch.tensor([tokens]).to("cuda")
        self._model_output, self._past = self._lm(tokens_tensor, past=self._past)

    def top_n_next(self, n):
        # debugging
        #print(self._tokenizer.decode(self._tokens_history.retrieve()))

        i_layer = 0
        i_final_token = -1

        if len(self._model_output.size()) == 2:
            p = self._model_output[i_layer]
        else:
            p = self._model_output[i_layer][i_final_token]

        top_n = torch.topk(torch.softmax(p, 0, torch.float32), n)

        scores = [v.item() for v in top_n.values]
        tokens = [self._tokenizer.decode(t) for t in top_n.indices.tolist()]

        return zip(scores, tokens)
