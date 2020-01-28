import torch

from src.model_factory import ModelFactory


class WordPredictor:
    _INDEX_FINAL_LAYER = 0
    _INDEX_BATCH = 0
    _INDEX_FINAL_TOKEN = -1

    def __init__(self, model_name):
        self._tokenizer, self._lm = ModelFactory().create_from_name(model_name)
        self._lm_hidden_states = None

    def top_n_next(self, n):
        top_n = torch.topk(
            torch.softmax(
                self._lm_hidden_states[self._INDEX_FINAL_LAYER][self._INDEX_BATCH][self._INDEX_FINAL_TOKEN],
                0,
                torch.float32),
            n)

        # todo: this code is not generic to all models/tokenizers
        scores = [v.item() for v in top_n.values]
        tokens = self._tokenizer.convert_ids_to_tokens([i.item() for i in top_n.indices])

        return zip(scores, tokens)

    def feed(self, text):
        input_token_ids = torch.tensor([self._tokenizer.encode(text, add_special_tokens=True)])
        input_token_ids = input_token_ids.to("cuda")

        with torch.no_grad():   # todo: performance (repeated context creation/destruction)
            self._lm_hidden_states = self._lm(input_token_ids)
