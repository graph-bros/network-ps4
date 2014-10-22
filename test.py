import unittest

from numpy import *
import sbm


class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        pass

    def test_sbm_likelihood(self):
        """
        For testing purpose only:

        Example 2.4 in Lecture 6.
        Lgood = 0.0433
        lnLgood = -3.1395
        """

        p10_example = """0 1 1 0 0 0;
        1 0 1 0 0 0;
        1 1 0 1 0 0;
        0 0 1 0 1 1;
        0 0 0 1 0 1;
        0 0 0 1 1 0"""
        z = array([1, 1, 1, 2, 2, 2])
        lgood = 0.0433
        lnlgood = -3.1395

        A = matrix(p10_example)
        lresult = sbm.likelihood(A, z, log_scale=False)
        lnlresult = sbm.likelihood(A, z, log_scale=True)

        self.assertAlmostEqual(lgood, lresult, 4)
        self.assertAlmostEqual(lnlgood, lnlresult, 4)

    def test_generate_sbm(self):
        M = matrix("0.5 0.03; 0.03 0.5")
        z = append(repeat(1,5), repeat(2,5))
        print sbm.generate(M, z)


if __name__ == '__main__':
    unittest.main()
