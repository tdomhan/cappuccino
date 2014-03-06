import unittest
from cappuccino.caffeconvnet import CaffeConvNet


class TestCaffeConvNet(unittest.TestCase):

    def setUp(self):
        # create a set of legal parameters for CaffeConvNet
        preproc_params = {}
        conv_layers = []
        fc_layers = []
        network_params = {"num_conv_layers": 1,
                          "num_fc_layers": 1,
                          "lr": 0.1,
                          "lr_policy": {"type": "fixed"},
                          "momentum": 0.1,
                          "weight_decay": 0.004
                          }

        self.params = (preproc_params, conv_layers, fc_layers, network_params)


    def test_illegal_input(self):
        """
           * missing parameters 
           * too many parameters
           * input files that don't exist? (although this may break other tests with empty string input)
        """
        #self.assertRaises(AssertionError, CaffeConvNet, )
        pass

    def test_simple_run(self):
        #TODO: run on a quick example to make sure everything is working.
        pass

if __name__ == "__main__":
    unittest.main()
