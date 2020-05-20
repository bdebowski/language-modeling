import torch

from src.word_predictor.iword_predictor import IWordPredictor
from src.util.rotating_sequence import RotatingSequence


class Gpt2WordPredictor(IWordPredictor):
    def __init__(self, tokenizer, model, device=None, use_past=False, mem_length=256):
        """
        use_past=True will speed up inference at the cost of higher memory consumption.  Set to False to avoid out of
        memory errors occurring after repeated forward feeds to the model.
        """
        self._tokenizer = tokenizer
        self._lm = model
        self._device = device
        self._past = None
        self._model_output = None

        self._use_past = use_past
        if not use_past:
            self._tokens_history = RotatingSequence(mem_length)

    @property
    def prepends_spaces(self):
        return True

    def feed(self, text, **kwargs):
        if not text:
            return

        # todo: how to handle new lines?
        prefix_space = text.startswith(" ")

        tokens = self._tokenizer.encode(text, add_prefix_space=prefix_space)

        if self._use_past:
            tokens_tensor = torch.tensor([tokens]).to(self._device)
            self._model_output, self._past = self._lm(tokens_tensor, past=self._past)
        else:
            for t in tokens:
                self._tokens_history.insert(t)
            with torch.no_grad():
                tokens_tensor = torch.tensor([self._tokens_history.retrieve()]).to(self._device)
                self._model_output = self._lm(tokens_tensor)

    def top_n_next(self, n):
        i_layer = 0
        i_batch = 0
        i_final_token = -1

        if self._use_past:
            if len(self._model_output.size()) == 2:
                p = self._model_output[i_layer]
            else:
                p = self._model_output[i_layer][i_final_token]
        else:
            p = self._model_output[i_layer][i_batch][-1]

        top_n = torch.topk(torch.softmax(p, 0, torch.float32), n)

        scores = [v.item() for v in top_n.values]
        tokens = [self._tokenizer.decode(t) for t in top_n.indices.tolist()]

        return zip(scores, tokens)
