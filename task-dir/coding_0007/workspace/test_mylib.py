import unittest

import mylib as m


class T(unittest.TestCase):
    def test_maximum(self):
        self.assertEqual(m.maximum([3, 9, 2]), 9)

    def test_total(self):
        self.assertEqual(m.total([1, 2, 3]), 6)

    def test_reverse_words(self):
        self.assertEqual(m.reverse_words('c a b'), 'b a c')

    def test_count_even(self):
        self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)


if __name__ == '__main__':
    unittest.main()
