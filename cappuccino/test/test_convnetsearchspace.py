import unittest
from cappuccino.convnetsearchspace import ConvNetSearchSpace
from cappuccino.convnetsearchspace import Parameter

class TestConvNetSearchSpace(unittest.TestCase):

    def test_parameter(self):
        self.assertRaises(AssertionError, Parameter, 5, 3)

        param = Parameter(3, 5)
        self.assertEqual(param.min_val, 3)
        self.assertEqual(param.max_val, 5)

        #give float to an int parameter
        self.assertRaises(AssertionError, Parameter, 3., 5., is_int=True)

        #default value calculation:
        param = Parameter(3, 5)
        self.assertEqual(param.default_val, (3.+5.)/2.)

        #for log params we want the default to be inbetween, but in log space
        param = Parameter(0.00001, 0.1, log_scale=True)
        self.assertTrue(param.log_scale)
        self.assertAlmostEqual(param.default_val, 0.001)

    def check_parameter_type(self, params):
        """Make sure the it's either a parameter or a constant"""
        if isinstance(params, dict):
            params = params.values()
        for param in params:
            param_type = type(param)
            if param_type is dict:
                self.check_parameter_type(param)
            elif param_type is list:
                self.check_parameter_type(param)
            else:
                self.assertTrue(isinstance(param, Parameter) or
                                param_type is int or
                                param_type is float or
                                param_type is str, "param of type: " + str(param_type))

    def test_convnetsearchspace(self):
        space = ConvNetSearchSpace(input_dimension=(3,32,32),
                                   max_conv_layers=2,
                                   max_fc_layers=2)
        preproc_parameters = space.get_preprocessing_parameter_subspace()
        self.check_parameter_type(preproc_parameters)
    
        network_parameters = space.get_network_parameter_subspace()
        self.check_parameter_type(network_parameters)

        for idx in range(1, 3):
            layer_params = space.get_conv_layer_subspace(idx)
            self.check_parameter_type(layer_params)

            layer_params = space.get_fc_layer_subspace(idx)
            self.check_parameter_type(layer_params)

 
    def test_layer_indexing(self):
        space = ConvNetSearchSpace(input_dimension=(3,32,32),
                                   max_conv_layers=2,
                                   max_fc_layers=2)
        #indexing is 1-based:
        self.assertRaises(AssertionError,
                          space.get_conv_layer_subspace, 0)
        self.assertRaises(AssertionError,
                          space.get_fc_layer_subspace, 0)
        #we shouldn't be able to get layers that are out of range:
        self.assertRaises(AssertionError,
                          space.get_conv_layer_subspace, 2+1)
        self.assertRaises(AssertionError,
                          space.get_fc_layer_subspace, 2+1)

    def test_illegal_input(self):
        self.assertRaises(AssertionError, ConvNetSearchSpace, (-1,32,32), 2, 2)
        self.assertRaises(AssertionError, ConvNetSearchSpace, (1,-32,32), 2, 2)
        self.assertRaises(AssertionError, ConvNetSearchSpace, (1,32,-32), 2, 2)
        self.assertRaises(AssertionError, ConvNetSearchSpace, (1,32,32), -2, 2)
        self.assertRaises(AssertionError, ConvNetSearchSpace, (1,32,32), 2, -2)
        # we need at least one fc layer:
        self.assertRaises(AssertionError, ConvNetSearchSpace, (1,32,32), 0, 0)

        #flat input, conv layers should not work:
        self.assertRaises(AssertionError, ConvNetSearchSpace, (100,1,1), 2, 2)

    def test_flat_input(self):
        """With a flat input we don't want cropping, nor conv layers"""
        space = ConvNetSearchSpace(input_dimension=(100,1,1),
                                   max_conv_layers=0,
                                   max_fc_layers=2)
        preproc_parameters = space.get_preprocessing_parameter_subspace()
        for augment_method in preproc_parameters["augment"]:
            self.assertTrue(augment_method["type"] != "crop")

    def test_rectangular_image(self):
        """With a rectangular (but not square) image we don't want cropping"""
        #NOTE: this is a limitation of caffe and might be removed in the future
        space = ConvNetSearchSpace(input_dimension=(3,12,24),
                                   max_conv_layers=2,
                                   max_fc_layers=2)
        preproc_parameters = space.get_preprocessing_parameter_subspace()
        for augment_method in preproc_parameters["augment"]:
            self.assertTrue(augment_method["type"] != "crop")


if __name__ == "__main__":
    unittest.main()
