import random
import re

from src.word_predictor.iword_predictor_factory import ModelName, IWordPredictorFactory


class PredictWordProg:
    def __init__(self):
        self._text_history = ""
        self._re_sentence_termination = re.compile(r"[\.\?!]($|\"|')")

    def run(self, word_predictor):
        # Show starting message
        print("Enter some text to get the model started...")
        print("")

        while True:
            # read text from user and feed to model
            word_predictor.feed(self._read_user_input())

            # pick how many sentences to generate
            num_sentences = 2

            sentences = []
            for i in range(num_sentences):
                # generate sentence and add to list of sentences to display
                sentences.append(self._generate_sentence(word_predictor))

            # add sentences to display
            self._display(sentences)

    @staticmethod
    def _display(sentences):
        print("")
        print(" ".join(sentences))
        print("")

    def _generate_sentence(self, word_predictor):
        words = []
        while True:
            # predict top n next words and their likelihoods
            n = 10
            predictions = list(word_predictor.top_n_next(n))
            total_p = sum(map(lambda x: x[0], predictions))

            # select next word probabilistically from the top n
            r = random.random() * total_p
            s = 0.0
            i = 0
            word = ""
            while s < r and i < n:
                s += predictions[i][0]  # predictions are (p, w) tuples where w is the word and p is the likelihood
                word = predictions[i][1]
                i += 1

            # feed this word to the model
            word_predictor.feed(word)

            # repeat until word is a sentence terminator
            if self._re_sentence_termination.search(word):
                join_char = "" if word_predictor.prepends_spaces else " "
                return join_char.join(words) + word

            # Add word to sentence words if not done.
            words.append(word)

    @staticmethod
    def _read_user_input():
        return input("> ")


if __name__ == "__main__":
    PredictWordProg().run(IWordPredictorFactory().create_from_name(ModelName.GPT2_LARGE, device="cpu"))
