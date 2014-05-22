import unittest
from cappuccino.paramutil import construct_parameter_tree_from_labels
from cappuccino.paramutil import group_layers
from cappuccino.paramutil import purge_inactive_parameters
from cappuccino.paramutil import remove_inactive_layers
from cappuccino.smacsearchspace import convnet_space_to_smac
from cappuccino.smacsearchspace import smac_space_default_configuration
from cappuccino.convnetsearchspace import ConvNetSearchSpace
from cappuccino.caffeconvnet import CaffeConvNet
import traceback


class TestSMACCaffnetIntegration(unittest.TestCase):

    def test_default_param_space_samples(self):
        """
            Create the default SMAC configurations of the parameter space
            and see that CaffeNet is not blowing up.
            Note: CaffeNet is not actually run. We just check if it's a valid configuration.
        """
        space = ConvNetSearchSpace((3, 32, 32),
                                   max_conv_layers=2,
                                   max_fc_layers=3)

        smac_space = convnet_space_to_smac(space)
        default_params = smac_space_default_configuration(smac_space)
        default_params = purge_inactive_parameters(default_params)

        param_tree = construct_parameter_tree_from_labels(default_params)
        grouped_params = group_layers(param_tree)
        caffenet_params = remove_inactive_layers(grouped_params)

        try:
            CaffeConvNet(caffenet_params,
                         train_file="",
                         valid_file="",
                         test_file="",
                         mean_file="",
                         num_train=50000,
                         num_valid=10000,
                         num_test=10000,
                         batch_size_valid=100,
                         batch_size_test=100)
        except:
            #something blew up
            self.fail(("Failed initializing CaffeConvNet"
                        "with parameters: %s, because of %s" % (str(caffenet_params),
                                                                traceback.format_exc())))


if __name__ == '__main__':
    unittest.main()
