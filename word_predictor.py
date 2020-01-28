

class WordPredictor:
    def __init__(self):
        self._m = None

    def top_n_next(self, n):
        return [("joe", 0.123), ("karen", 0.191), ("magic", 0.34)]

    def feed(self, text):
        pass
