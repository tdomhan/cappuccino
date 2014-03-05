import unittest
from cappuccino.paramutil import construct_parameter_tree_from_labels
from cappuccino.paramutil import group_layers
from cappuccino.paramutil import flatten_to_leaves
from cappuccino.paramutil import purge_inactive_parameters
from cappuccino.paramutil import remove_inactive_layers


class TestParameterTreeConstruction(unittest.TestCase):

    def setUp(self):
        pass

    def test_construct_tree_simple(self):
        params = {"convlayer0/kernelwidth": 1, "convlayer0/weight-filler@gaussian/std": 0.01}
        expected_tree = {"convlayer0": {"kernelwidth": 1, "weight-filler": {"type": "gaussian", "std": 0.01}}}
        param_tree = construct_parameter_tree_from_labels(params, escape_char_depth = "/", escape_char_choice = "@")

        self.assertDictEqual(expected_tree, param_tree)

    def test_construct_tree_simple2(self):
        params =  {'fc-layer-3/weight-filler@gaussian/std': '0.002240578878878171',
                   'fc-layer-3/weight-filler@gaussian/type': 'gaussian'}
        expected_tree = {"fc-layer-3": {"weight-filler": {"type": "gaussian", "std": 0.002240578878878171}}}
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
        params = {"preprocessing": -1, "conv-layer-1": 1, "conv-layer-2":2, "fc-layer-3": 3, "network": 4}

        preprocessing, conv_layers, fc_layers, network = group_layers(params)
        self.assertEqual(conv_layers, [1,2])
        self.assertEqual(fc_layers, [3])
        self.assertEqual(network, 4)
        self.assertEqual(preprocessing, -1)

    def test_group_layer_no_conv(self):
        params = {"preprocessing": 1, "fc-layer-3": 3, "network": 4}

        preproc, conv_layers, fc_layers, network = group_layers(params)
        self.assertEqual(conv_layers, [])
        self.assertEqual(fc_layers, [3])
        self.assertEqual(preproc, 1)
        self.assertEqual(network, 4)

class TestFlattenParams(unittest.TestCase):

    def test_flatten_params(self):
        params = {"level1": {
                              "level2-leaf-1": 1,
                              "level2-leaf-2": 2,
                              "level2-leaf-3": {"level3" : 4}
                            }
                 }

        expected_params = {"level2-leaf-1": 1,
                           "level2-leaf-2": 2,
                           "level3": 4}

        flattened_params = flatten_to_leaves(params)
        self.assertEqual(flattened_params, expected_params)

    def test_non_unique_keys(self):
        params = {"level1-1": {
                              "level2-leaf-1": 1,
                             },
                  "level1-2": {
                              "level2-leaf-1": 1,
                              }
                 }
       #We have the same key twice in different leaves, that shouldn't work
        self.assertRaises(AssertionError, flatten_to_leaves, params)

    def test_list(self):
        params = [{"a": 1, "b": 2}, {"c": 3}]
        expected_params = {"a": 1, "b": 2, "c": 3}

        self.assertEqual(flatten_to_leaves(params),
                         expected_params)

    def test_tuple(self):
        params = ({"a": 1, "b": 2}, {"c": 3})
        expected_params = {"a": 1, "b": 2, "c": 3}

        self.assertEqual(flatten_to_leaves(params),
                         expected_params)

class TestPurgeInactiveParameters(unittest.TestCase):
    def test_input_not_modified(self):
        params = {"policy/type": "fixed",
                  "policy@fixed/lr": 1,
                  "policy@decay/decay_factor": 0.5}
        params_original = {"policy/type": "fixed",
                           "policy@fixed/lr": 1,
                           "policy@decay/decay_factor": 0.5}
        #run the purging and see if the input has not been modified
        purge_inactive_parameters(params)
        self.assertDictEqual(params,
                             params_original)


    def test_purge_inactive_parameters(self):
        params = {"policy/type": "fixed",
                  "policy@fixed/lr": 1,
                  "policy@decay/decay_factor": 0.5,
                  "test@trytoconfuse": 5,
                  "test@trytoconfuse/withthis": 5,
                  "another/parameter/type": "anothertype",
                  "another/parameter@anothertype/test": "test",
                  "another/parameter@unusedtype": "me",
                  "another/parameter@anotherunusedtype": "me",
                  'fc-layer-3/weight-filler/type': 'gaussian',
                  'fc-layer-3/weight-filler@gaussian/std': 0.001,
                  'fc-layer-3/weight-filler@xavier/std': 0.001,
                  }
        params_purged = {"policy/type": "fixed",
                         "policy@fixed/lr": 1,
                         "test@trytoconfuse": 5,
                         "test@trytoconfuse/withthis": 5,
                         "another/parameter/type": "anothertype",
                         "another/parameter@anothertype/test": "test",
                         'fc-layer-3/weight-filler/type': 'gaussian',
                         'fc-layer-3/weight-filler@gaussian/std': 0.001,
                         }

        self.assertDictEqual(purge_inactive_parameters(params),
                             params_purged)


class TestRemoveInactiveLayers(unittest.TestCase):
    def test_remove_inactive_layers(self):
        network_params = {"num_conv_layers": 2,
                          "num_fc_layers": 2}

        conv_layer_params = ["conv-layer-1",
                             "conv-layer-2",
                             "conv-layer-3",
                             "conv-layer-4"]

        conv_layer_active_params = ["conv-layer-1",
                                    "conv-layer-2"]
 
        fc_layer_params = ["fc-layer-1",
                           "fc-layer-2",
                           "fc-layer-3",
                           "fc-layer-4"]

        fc_layer_active_params = ["fc-layer-1",
                                  "fc-layer-2"]
 
        params = [{"preproc": "test"},
                  conv_layer_params,
                  fc_layer_params,
                  network_params]
                  
        test_active_params = remove_inactive_layers(params) 
        #make sure the input hasn't changed:
        self.assertEqual(len(conv_layer_params), 4)
        self.assertEqual(len(fc_layer_params), 4)

        self.assertItemsEqual(test_active_params[0], {"preproc": "test"})
        self.assertItemsEqual(test_active_params[1], conv_layer_active_params)
        self.assertItemsEqual(test_active_params[2], fc_layer_active_params)
        self.assertEqual(len(test_active_params), 4)

if __name__ == '__main__':
    unittest.main()
