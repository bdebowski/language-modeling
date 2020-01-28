import os

from word_predictor import WordPredictor


class PredictWordProg:
    def __init__(self):
        self.text_history = []

    def run(self, word_predictor):
        word_predictor.feed("")
        while True:
            self.display(word_predictor.top_n_next(10))
            tokens = self.read_user_input()
            for t in tokens:
                self.text_history.append(t)
                word_predictor.feed(t)

    def display(self, top_n_next):
        """
        Screen will look like so:
        --------------------------------------------
        Last n words written are shown here followed by ______

        predicted_next_0    score
        predicted_next_1    score
        predicted_next_2    score
        predicted_next_3    score
        predicted_next_4    score

        > user_input

        --------------------------------------------
        """
        os.system("cls")
        print("{} ____".format(" ".join(self.text_history[-20:])))
        print("")
        for w, p in top_n_next:
            print("{}\t{:2f}".format(w, p))
        print("")

    @staticmethod
    def read_user_input():
        return input("> ").split()


if __name__ == "__main__":
    PredictWordProg().run(WordPredictor())
