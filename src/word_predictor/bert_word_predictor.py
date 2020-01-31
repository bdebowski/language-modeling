import re

import torch

from src.util.rotating_sequence import RotatingSequence


class BertWordPredictor:
    _INDEX_FINAL_LAYER = 0
    _INDEX_BATCH = 0
    _INDEX_FINAL_TOKEN = -1

    def __init__(self, tokenizer, model, mem_length=512):
        self._tokenizer = tokenizer
        self._lm = model
        self._lm_hidden_states = None

        # mem_length - 1 because we will always append a [MASK] token in each forward feed.
        self._token_ids_memory = RotatingSequence(mem_length - 1)
        self._token_ids_memory.insert(tokenizer.sep_token_id)

        self._re_sep_required = re.compile(r"([\.\?;])(\s|$)")
        self._re_sep_repl_str = r"\1 {} ".format(tokenizer.sep_token)

    def feed(self, text):
        # Tokenize text and insert SEP token after each/any sentence terminating character.
        tokenized_text = self._tokenizer.tokenize(self._re_sep_required.sub(self._re_sep_repl_str, text))
        #tokenized_text = self._tokenizer.tokenize(text)

        # Convert tokens to their ids and append to memory
        text_token_ids = self._tokenizer.convert_tokens_to_ids(tokenized_text)
        for token_id in text_token_ids:  # todo: performance: better to create and use an insert_iterable method
            self._token_ids_memory.insert(token_id)

        # The complete tokenized text fed to the model each time is the concatenation of:
        # previously fed tokens + tokenized text + MASK token
        # the number of tokens fed are limited to mem_length
        token_ids_to_feed = self._token_ids_memory.retrieve()
        token_ids_to_feed.append(self._tokenizer.mask_token_id)

        # DEBUG
        #print(self._tokenizer.convert_ids_to_tokens(token_ids_to_feed))

        # Convert to tensor and push to GPU
        tokens_tensor = torch.tensor([token_ids_to_feed]).to("cuda")

        # BERT uses A/B segments.  We have to assign the tokens to the A or B segment.
        # We will assign all tokens to the A segment.
        segment_ids = [0] * len(token_ids_to_feed)
        segments_tensor = torch.tensor([segment_ids]).to("cuda")

        # Feed forward all the tokens
        with torch.no_grad():   # todo: performance: repeated context creation/destruction
            self._lm_hidden_states = self._lm(tokens_tensor, token_type_ids=segments_tensor)

    def top_n_next(self, n):
        top_n = torch.topk(
            torch.softmax(
                self._lm_hidden_states[self._INDEX_FINAL_LAYER][self._INDEX_BATCH][self._INDEX_FINAL_TOKEN],
                0,
                torch.float32),
            n)

        scores = [v.item() for v in top_n.values]
        tokens = self._tokenizer.convert_ids_to_tokens([i.item() for i in top_n.indices])

        return zip(scores, tokens)
