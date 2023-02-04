import unittest
from utils import confidence_margin_mean


class TestUtils(unittest.TestCase):
    """
    Testing the utils, namely, the confidence margin function
    """

    def test_constant_confidence_margin(self):
        """
        The function should output the entry of any constant vector as the mean and the bounds
        """
        print("\nTesting constant vectors\n")
        for i in range(3):
            T = [i] * 10
            print("\nTesting the vector: ", T, "\n")
            self.assertEqual((i, i, i), confidence_margin_mean(T))

    def test_different_confidence(self):
        """
        The function should output the entry of any constant vector for any confidence margin
        """
        print("\nTesting constant vectors\n")
        margins = [0.1, 0.5, 0.95]
        T = [1] * 10
        for m in margins:
            print("\nTesting the vector: ", T, "with margin", m, "\n")
            self.assertEqual((1, 1, 1), confidence_margin_mean(T, m))


if __name__ == "__main__":
    # begin the unittest.main()
    unittest.main()
