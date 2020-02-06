from abc import abstractmethod, abstractproperty, ABC


class IWordPredictor(ABC):
    @property
    @abstractmethod
    def prepends_spaces(self):
        """
        True if the word predictor prepends spaces to complete words it generates
        """
        pass

    @abstractmethod
    def feed(self, text, **kwargs):
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
