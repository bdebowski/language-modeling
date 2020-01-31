

class RotatingSequence:
    def __init__(self, length):
        self._buffer = [None] * length
        self._buffer_size = length
        self._i_next = 0
        self._num_items = 0

    def insert(self, item):
        """
        Inserts an item into the rotating sequence.
        """
        self._buffer[self._i_next] = item
        self._i_next += 1
        self._i_next = self._i_next % self._buffer_size
        self._num_items = min(self._buffer_size, self._num_items + 1)

    def retrieve(self):
        """
        Retrieves the current sequence.
        """
        a = self._buffer[self._i_next:self._num_items]  # first part of sequence to return
        b = self._buffer[:self._i_next]  # rest of sequence to return; can be empty
        return a + b
