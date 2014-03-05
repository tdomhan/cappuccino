import unittest
from cappuccino.tpesearchspace import convnet_space_to_tpe
from cappuccino.tpesearchspace import tpe_sample_to_caffenet
from cappuccino.convnetsearchspace import ConvNetSearchSpace
from cappuccino.caffeconvnet import CaffeConvNet
import hyperopt.pyll.stochastic

NUM_RANDOM_SAMPLES = 1000

class TestTPECaffnetIntegration(unittest.TestCase):

    def test_random_param_space_samples(self):
        """
            Create random configurations of the parameter space and see that
            CaffeNet is not blowing up.
            Note: CaffeNet is not actually run. We just check if it's a valid configuration.
        """
        space = ConvNetSearchSpace((3, 32, 32),
                                   max_conv_layers=2,
                                   max_fc_layers=3)
        tpe_space = convnet_space_to_tpe(space)

        for i in range(0, NUM_RANDOM_SAMPLES):
            print "Testing param sample %d/%d\r" % (i+1, NUM_RANDOM_SAMPLES),
            tpe_sample = hyperopt.pyll.stochastic.sample(tpe_space)
            caffenet_params = tpe_sample_to_caffenet(tpe_sample)
            try:
                CaffeConvNet(caffenet_params,
                             train_file="",
                             valid_file="",
                             mean_file="",
                             num_train=50000,
                             num_valid=10000,
                             batch_size_train=100,
                             batch_size_valid=100)
            except:
                self.fail(("Failed initializing CaffeConvNet"
                            "with parameters: %s" % str(caffenet_params)))
 

if __name__ == '__main__':
    unittest.main()
