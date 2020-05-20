import re

import torch

from src.util.rotating_sequence import RotatingSequence
from src.word_predictor.iword_predictor import IWordPredictor


class BertWordPredictor(IWordPredictor):
    _INDEX_FINAL_LAYER = 0
    _INDEX_BATCH = 0
    _INDEX_FINAL_TOKEN = -1

    _NUM_APPENDED_MASKS_MIN = 5
    _NUM_APPENDED_MASKS_MAX = 5

    def __init__(self, tokenizer, model, device=None, mem_length=512):
        self._tokenizer = tokenizer
        self._lm = model
        self._device = device
        self._lm_output_states = [None] * (self._NUM_APPENDED_MASKS_MAX - self._NUM_APPENDED_MASKS_MIN + 1)

        # mem_length - _NUM_APPENDED_MASKS_MAX because we will always append up to _NUM_APPENDED_MASKS_MAX [MASK]
        # tokens in each forward feed.
        self._token_ids_memory = RotatingSequence(mem_length - self._NUM_APPENDED_MASKS_MAX)
        self._token_ids_memory.insert(tokenizer.sep_token_id)

        self._re_sep_required = re.compile(r"([\.\?;])(\s|$)")
        self._re_sep_repl_str = r"\1 {} ".format(tokenizer.sep_token)

    @property
    def prepends_spaces(self):
        return False

    def feed(self, text, **kwargs):
        try:
            segment_id = kwargs["segment_id"]
        except KeyError:
            segment_id = 0

        # Tokenize text and insert SEP token after each/any sentence terminating character.
        tokenized_text = self._tokenizer.tokenize(self._re_sep_required.sub(self._re_sep_repl_str, text))

        # Convert tokens to their ids and append to memory
        text_token_ids = self._tokenizer.convert_tokens_to_ids(tokenized_text)
        for token_id in text_token_ids:  # todo: performance: better to create and use an insert_iterable method
            self._token_ids_memory.insert(token_id)

        # The complete tokenized text fed to the model each time is the concatenation of:
        # previously fed tokens + tokenized text + m x MASK tokens; total number of tokens fed is limited to mem_length
        # We perform n iterations of feeding to the model, changing m at each iteration.
        # We average the word predictions across the n iterations.
        for n in range(self._NUM_APPENDED_MASKS_MAX - self._NUM_APPENDED_MASKS_MIN + 1):
            token_ids_to_feed = self._token_ids_memory.retrieve() + [self._tokenizer.mask_token_id] * (self._NUM_APPENDED_MASKS_MIN + n)

            # Convert to tensor and push to GPU
            tokens_tensor = torch.tensor([token_ids_to_feed]).to(self._device)

            # BERT uses A/B segments.  We have to assign the tokens to the A or B segment.
            segment_ids = [segment_id] * len(token_ids_to_feed)
            segments_tensor = torch.tensor([segment_ids]).to(self._device)

            # Feed forward all the tokens
            with torch.no_grad():   # todo: performance: repeated context creation/destruction
                self._lm_output_states[n] = self._lm(tokens_tensor, token_type_ids=segments_tensor)[self._INDEX_FINAL_LAYER][self._INDEX_BATCH][-self._NUM_APPENDED_MASKS_MIN - n]

    def top_n_next(self, n):
        # debugging
        #print(self._tokenizer.convert_ids_to_tokens(self._token_ids_memory.retrieve()))

        top_n = torch.topk(
            torch.softmax(
                torch.sum(torch.stack(self._lm_output_states), dim=0),
                0,
                torch.float32),
            n)

        scores = [v.item() for v in top_n.values]
        tokens = self._tokenizer.convert_ids_to_tokens([i.item() for i in top_n.indices])

        return zip(scores, tokens)
