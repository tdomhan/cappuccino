import unittest
from cappuccino.paramutil import construct_parameter_tree_from_labels
from cappuccino.paramutil import group_layers


class TestParameterTreeConstruction(unittest.TestCase):

    def setUp(self):
        pass

    def test_construct_tree(self):
        params = {"convlayer0/kernelwidth": 1, "convlayer0/weight-filler@gaussian/std": 0.01}
        expected_tree = {"convlayer0": {"kernelwidth": 1, "weight-filler": {"type": "gaussian", "std": 0.01}}}
        param_tree = construct_parameter_tree_from_labels(params, escape_char_depth = "/", escape_char_choice = "@")

        self.assertDictEqual(expected_tree, param_tree)

    def test_bad_input(self):
        """
            after a choice elemente like "weight-filler@gaussian", we can't
            directly get the value, but rather need the different fields
            that we want to fill, e.g. "weight-filler@gaussian/std"
        """
        params = {"convlayer0/kernelwidth": 1, "convlayer0/weight-filler@gaussian": 0.01}

        self.assertRaises(AssertionError, construct_parameter_tree_from_labels, params)

        """
            There can only be one type specifier per level.
        """
        params = {"convlayer0/kernelwidth": 1, "convlayer0/weight-filler@gaussian@somethingelse/std": 0.01}
        self.assertRaises(AssertionError, construct_parameter_tree_from_labels, params)

class TestGroupLayer(unittest.TestCase):

    def test_group_layer(self):
        params = {"conv-layer-1": 1, "conv-layer-2":2, "fc-layer-3": 3, "network": 4}

        conv_layers, fc_layers, network = group_layers(params)
        self.assertEqual(conv_layers, [1,2])
        self.assertEqual(fc_layers, [3])
        self.assertEqual(network, 4)



if __name__ == '__main__':
    unittest.main()
