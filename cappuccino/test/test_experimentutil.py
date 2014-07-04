from cappuccino.experimentutil import get_current_ybest, update_ybest, hpolib_experiment_main
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

    def test_hpolib_experiment_main(self):
        class FakeNet(object):
            def __init__(self):
                pass
            def run(self):
                pass

        def construct_caffeconvnet(params):
            return FakeNet()

        hpolib_experiment_main({}, construct_caffeconvnet=construct_caffeconvnet,
            experiment_dir="", working_dir="", mean_performance_on_last=10)



    def cleanup(self):
    	os.remove("ybest.txt")


if __name__ == '__main__':
    unittest.main()