from abc import abstractmethod, ABC


class IWordPredictor(ABC):
    @abstractmethod
    def feed(self, text):
        """
        Feed the text provided to the underlying model.
        """
        pass

    @abstractmethod
    def top_n_next(self, n):
        """
        Returns n 2-tuples of form (p, w) where w is a predicted next word and p is the likelihood of that word
        appearing next.
        """
        pass
