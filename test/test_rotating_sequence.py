from unittest import TestCase
import random

from src.util.rotating_sequence import RotatingSequence


class TestRotatingSequence(TestCase):
    def test_insert_accurate_when_buffer_under_full(self):
        seq = RotatingSequence(3)
        seq.insert("a")
        seq.insert("b")

        self.assertSequenceEqual(["a", "b", None], seq._buffer)
        self.assertEqual(2, seq._num_items)
        self.assertEqual(2, seq._i_next)

    def test_insert_accurate_when_buffer_reaches_full(self):
        seq = RotatingSequence(3)
        seq.insert("a")
        seq.insert("b")
        seq.insert("c")

        self.assertSequenceEqual(["a", "b", "c"], seq._buffer)
        self.assertEqual(3, seq._num_items)
        self.assertEqual(0, seq._i_next)

    def test_insert_accurate_when_buffer_already_full(self):
        seq = RotatingSequence(3)
        seq.insert("a")
        seq.insert("b")
        seq.insert("c")
        seq.insert("d")

        self.assertSequenceEqual(["d", "b", "c"], seq._buffer)
        self.assertEqual(3, seq._num_items)
        self.assertEqual(1, seq._i_next)

    def test_retrieve_accurate_when_buffer_under_full(self):
        seq = RotatingSequence(3)
        seq.insert("a")
        seq.insert("b")

        self.assertSequenceEqual(["a", "b"], seq.retrieve())

    def test_retrieve_accurate_when_buffer_full(self):
        seq = RotatingSequence(3)
        seq.insert("a")
        seq.insert("b")
        seq.insert("c")

        self.assertSequenceEqual(["a", "b", "c"], seq.retrieve())

    def test_retrieve_accurate_when_buffer_circled_around(self):
        seq = RotatingSequence(3)
        seq.insert("a")
        seq.insert("b")
        seq.insert("c")
        seq.insert("d")

        self.assertSequenceEqual(["b", "c", "d"], seq.retrieve())

    def test_retrieve_accurate_when_buffer_circled_around_and_random_num_insertions(self):
        chars = [chr(c) for c in range(ord('a'), ord('z') + 1)]
        for i in range(20):  # 20 random test runs
            seq = RotatingSequence(3)
            num_insertions = random.randint(4, 26)
            for n in range(num_insertions):
                seq.insert(chars[n])

            self.assertSequenceEqual(chars[num_insertions - 3:num_insertions], seq.retrieve())
