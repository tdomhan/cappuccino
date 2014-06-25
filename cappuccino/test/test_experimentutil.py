from cappuccino.experimentutil import get_current_ybest, update_ybest
import unittest
import os


class TestYbestReadWrite(unittest.TestCase):

    def test_get_current_ybest(self):
    	open("ybest.txt", "w").write("0.5")
    	ybest = get_current_ybest()
        self.assertAlmostEqual(ybest, 0.5)
        self.cleanup()

    def test_update_ybest(self):
    	update_ybest(0.1)
    	ybest = get_current_ybest()
        self.assertAlmostEqual(ybest, 0.1)
        update_ybest(0.2)
    	ybest = get_current_ybest()
        self.assertAlmostEqual(ybest, 0.2)
        update_ybest(0.15)
    	ybest = get_current_ybest()
        self.assertAlmostEqual(ybest, 0.2)
        self.cleanup()

    def cleanup(self):
    	os.remove("ybest.txt")


if __name__ == '__main__':
    unittest.main()