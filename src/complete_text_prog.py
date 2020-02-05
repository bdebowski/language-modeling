import random

from src.word_predictor.iword_predictor_factory import ModelName, IWordPredictorFactory


class PredictWordProg:
    def __init__(self):
        self._text_history = ""

    def run(self, word_predictor):
        # Show starting message
        print("Enter some text to get the model started...")
        print("")

        while True:
            # read text from user and feed to model
            word_predictor.feed(self._read_user_input())

            # pick how many sentences to generate
            num_sentences = 1

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

    @staticmethod
    def _generate_sentence(word_predictor):
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
            if word in {".", ";", "?"} or word.endswith(";"):
                return " ".join(words) + word

            # Add word to sentence words if not done.
            # Some models, like GPT2, encode/decode with a leading space.  Let's strip any such spaces and create them
            # explicitly when displaying text.
            words.append(word.strip())

    @staticmethod
    def _read_user_input():
        return input("> ")


if __name__ == "__main__":
    PredictWordProg().run(IWordPredictorFactory().create_from_name(ModelName.GPT2))
