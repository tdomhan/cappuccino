import unittest
from cappuccino.caffeconvnet import CaffeConvNet


class TestCaffeConvNet(unittest.TestCase):

    def setUp(self):
        # create a set of legal parameters for CaffeConvNet
        pass

    def get_default_preproc_params(self):
        preproc_params = {'augment': {'type': 'none'},
                            'input_dropout': {'type': 'none'}}
        return preproc_params

    def get_default_network_params(self):

        network_params = {"num_conv_layers": 1,
                          "num_fc_layers": 0,
                          "lr": 0.1,
                          "lr_policy": {"type": "fixed"},
                          "momentum": 0.1,
                          "weight_decay": 0.004,
                          "batch_size_train": 100
                          }
        return network_params

    def get_default_conv_params(self, kernelsize=3, stride=1,
      padding={"type": "implicit"},
      pooling={"type": "none"}):
        conv_layer_params = {"type": "conv",
                              "kernelsize": kernelsize,
                              "num_output": 100,
                              "stride": stride,
                              "weight-filler": {"type": "xavier"},
                              "bias-filler": {"type": "const-zero"},
                              "weight-lr-multiplier": 1,
                              "bias-lr-multiplier": 2,
                              "weight-weight-decay_multiplier": 1,
                              "bias-weight-decay_multiplier": 0,
                              "padding": padding,
                              "pooling": pooling,
                              "norm": {"type": "none"},
                              "dropout": {"type": "no_dropout"}}
        return conv_layer_params

    def test_implicit_paddding_output_size_calculation(self):
        input_image_size = 32
        kernelsize = 3
        stride = 1
        padding = {"type": "implicit"}
        #because we have implicit padding + no pooling layers, we expected the output size to
        #be equal to the input size
        expected_output_size = input_image_size

        conv_layer_params = self.get_default_conv_params(kernelsize=kernelsize,
            stride=stride, padding=padding)
        conv_layers = [conv_layer_params]
        fc_layers = []
        self.caffenet_params = (self.get_default_preproc_params(),
                                conv_layers, fc_layers, self.get_default_network_params())
        convnet = CaffeConvNet(self.caffenet_params,
                               train_file="",
                               valid_file="",
                               test_file="",
                               mean_file="",
                               restrict_to_legal_configurations=False,
                               input_image_size=input_image_size,
                               num_train=50000,
                               num_valid=10000,
                               num_test=10000,
                               batch_size_valid=100,
                               batch_size_test=100)
        self.assertEqual(expected_output_size, convnet._output_image_size)


    def test_no_paddding_output_size_calculation(self):
        input_image_size = 32
        kernelsize = 3
        stride = 1
        padding = {"type": "none"}
        #because we have implicit padding + no pooling layers, we expected the output size to
        #be equal to the input size
        expected_output_size = input_image_size - kernelsize + 1

        conv_layer_params = self.get_default_conv_params(kernelsize=kernelsize,
            stride=stride, padding=padding)
        conv_layers = [conv_layer_params]
        fc_layers = []
        self.caffenet_params = (self.get_default_preproc_params(),
                                conv_layers, fc_layers, self.get_default_network_params())
        convnet = CaffeConvNet(self.caffenet_params,
                               train_file="",
                               valid_file="",
                               test_file="",
                               mean_file="",
                               restrict_to_legal_configurations=False,
                               input_image_size=input_image_size,
                               num_train=50000,
                               num_valid=10000,
                               num_test=10000,
                               batch_size_valid=100,
                               batch_size_test=100)
        self.assertEqual(expected_output_size, convnet._output_image_size)


    def test_pooling_output_size_calculation(self):
        input_image_size = 32
        kernelsize = 3
        stride = 1
        padding = {"type": "implicit"}
        pooling = {"type": "max", "stride": 2, "kernelsize": 3}
        #because we have implicit padding + no pooling layers, we expected the output size to
        #be equal to the input size
        expected_output_size = (input_image_size - 3) / 2 + 1

        conv_layer_params = self.get_default_conv_params(kernelsize=kernelsize,
            stride=stride, padding=padding, pooling=pooling)
        conv_layers = [conv_layer_params]
        fc_layers = []
        self.caffenet_params = (self.get_default_preproc_params(),
                                conv_layers, fc_layers, self.get_default_network_params())
        convnet = CaffeConvNet(self.caffenet_params,
                               train_file="",
                               valid_file="",
                               test_file="",
                               mean_file="",
                               restrict_to_legal_configurations=False,
                               input_image_size=input_image_size,
                               num_train=50000,
                               num_valid=10000,
                               num_test=10000,
                               batch_size_valid=100,
                               batch_size_test=100)
        self.assertEqual(expected_output_size, convnet._output_image_size)

if __name__ == "__main__":
    unittest.main()
