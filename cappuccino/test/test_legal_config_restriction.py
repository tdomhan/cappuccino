import unittest
import traceback
from cappuccino.tpesearchspace import convnet_space_to_tpe
from cappuccino.tpesearchspace import tpe_sample_to_caffenet
from cappuccino.convnetsearchspace import ConvNetSearchSpace
from cappuccino.caffeconvnet import CaffeConvNet
import hyperopt.pyll.stochastic

NUM_RANDOM_SAMPLES = 200

class TestTPECaffnetIntegration(unittest.TestCase):

    def test_legal_config_restriction(self):
        """
            Create random configurations of the parameter space and see that
            the number output size never gets below zero, if we turn on the
            restriction of the config output.
        """
        space = ConvNetSearchSpace((3, 32, 32),
                                   max_conv_layers=8,
                                   max_fc_layers=3,
                                   implicit_conv_layer_padding=True)
        tpe_space = convnet_space_to_tpe(space)

        for i in range(0, NUM_RANDOM_SAMPLES):
            print "Testing param sample %d/%d\r" % (i+1, NUM_RANDOM_SAMPLES),
            tpe_sample = hyperopt.pyll.stochastic.sample(tpe_space)
            caffenet_params = tpe_sample_to_caffenet(tpe_sample)
            try:
                min_image_size = 4
                convnet = CaffeConvNet(caffenet_params,
                             train_file="",
                             valid_file="",
                             test_file="",
                             mean_file="",
                             restrict_to_legal_configurations=True,
                             min_image_size=min_image_size,
                             input_image_size=32,
                             num_train=50000,
                             num_valid=10000,
                             num_test=10000,
                             batch_size_valid=100,
                             batch_size_test=100)
                self.assertTrue(convnet._output_image_size >= min_image_size)
            except:
                print traceback.format_exc()
                self.fail(("Failed initializing CaffeConvNet"
                            "with parameters: %s" % str(caffenet_params)))
        #overwrite the carriage return
        print ""
 

if __name__ == '__main__':
    unittest.main()
