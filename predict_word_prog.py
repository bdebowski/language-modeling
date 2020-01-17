from word_predictor import WordPredictor


class PredictWordProg:
    "test"
    def __init__(self):
        self.text = ""

    def run(self, m):
        m.feed("")
        while True:
            self.display(m.top_n_next(10))
            m.feed(self.read())

    def display(self, top_n_next):
        for w, p in top_n_next:
            pass

    def read(self):
        return ""


if __name__ == "__main__":
    PredictWordProg().run(WordPredictor())
