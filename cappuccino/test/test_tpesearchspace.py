import unittest
from cappuccino.tpesearchspace import subspace_to_tpe
from cappuccino.tpesearchspace import convnet_space_to_tpe
from cappuccino.tpesearchspace import tpe_sample_to_caffenet
from cappuccino.convnetsearchspace import Parameter
from cappuccino.convnetsearchspace import ConvNetSearchSpace
from math import log


class TestTPESearchSpace(unittest.TestCase):
    """
        Check that the TPE search space is correctly assembled
        from a ConvNetSearchSpace.

    """

    def setUp(self):
        pass

    def test_simple_subspace_conversion(self):
        """Let's build up a small search space and try to convert it to SMAC.
        """
        search_space = {"param1": Parameter(12.3, 19.7),
                        "param2": Parameter(10, 100, log_scale=True),
                        "param3": Parameter(0, 100, default_val=20, is_int=True),
                        "param3": Parameter(2, 100, default_val=20, log_scale=True, is_int=True),
                        }
        tpe_space = subspace_to_tpe("testspace", search_space,
                                    escape_char_depth="/")

        #everything should be on the same level
        self.assertEqual(len(tpe_space.values()), 3)

        for param_name, param in search_space.iteritems():
            param_full_name = "testspace/" + param_name 
            self.assertTrue(param_full_name  in tpe_space)

            min_val = param.min_val
            max_val = param.max_val
            if param.log_scale:
                if param.is_int:
                    param_type = "qloguniform"
                else:
                    param_type = "loguniform"
                # because hyperopt draw samples of exp(uniform(low, high))
                # we first need to log transform
                min_val = log(float(min_val))
                max_val = log(float(max_val))
            else:
                if param.is_int:
                    param_type = "quniform"
                else:
                    param_type = "uniform"
            tpe_param = tpe_space[param_full_name]

            self.assertEqual(
                tpe_param.inputs()[0].arg['obj'].name,
                param_type)
            self.assertEqual(
                tpe_param.inputs()[0].arg["label"].obj,
                param_full_name)
            self.assertAlmostEqual(
                tpe_param.inputs()[0].arg['obj'].arg["high"].obj,
                max_val)
            self.assertAlmostEqual(
                tpe_param.inputs()[0].arg['obj'].arg["low"].obj,
                min_val)
            if param_type.startswith("q"):
                # we use quniform to encode integer, so q needs to be 1
                self.assertAlmostEqual(
                    tpe_param.inputs()[0].arg['obj'].arg["q"].obj,
                    1.)

    def test_subspace_conversion_conditions(self):
        """Let's build a search space with conditional parameters.
        """
        space1 = {"type": "space1",
                  "param1": Parameter(0, 10),
                  "param2": Parameter(10, 100)}
        space2 = {"type": "space2",
                  "param1": Parameter(0, 10),
                  "param2": Parameter(10, 100)}
        #space3 is a subspace without parameters:
        space3 = {"type": "space3"}

        search_space = {"condspace": [space1, space2, space3]}
 
        tpe_space = subspace_to_tpe("testspace", search_space)

        self.assertTrue("testspace/condspace"  in tpe_space)
        #TODO check for hp.choice parameter!?

    def test_convnet_space_conversion(self):
        space = ConvNetSearchSpace((3, 32, 32),
                                   max_conv_layers=2,
                                   max_fc_layers=3)
        tpe_space = convnet_space_to_tpe(space)
        #preprocssing, network, conv-layers, fc-layers, format
        self.assertEqual(len(tpe_space), 5)

        first_level_params = []
        for item in tpe_space:
            if isinstance(item, dict):
                first_level_params.extend(item.keys())

        self.assertTrue("format" in first_level_params)
        self.assertEqual(tpe_space[0]["format"], "tpe")

        self.assertTrue("network/lr" in first_level_params)
        self.assertTrue("network/lr_policy" in first_level_params)
        self.assertTrue("network/momentum" in first_level_params)
        self.assertTrue("network/weight_decay" in first_level_params)
        self.assertTrue("preprocessing/augment" in first_level_params)

        #TODO: check for conv and fc layers

class TestTPESampleToCaffenet(unittest.TestCase):

    def test_check_format(self):
        """should not work if the tpe format flag is missing."""
        params = [{"format": "smac"},
                  {"preprocessing/test": "preproc"},
                  {"network/test": "network"},
                  ({"conv-layer-1": "conv-1"},
                   {"conv-layer-2": "conv-2"},
                   {"conv-layer-3": "conv-3"}),
                  ({"fc-layer-1": "fc-1"},
                   {"fc-layer-2": "fc-2"},
                   {"fc-layer-3": "fc-3"}),
                  ]
        self.assertRaises(AssertionError,
                          tpe_sample_to_caffenet,
                          params)
 
    def test_simple_hyperopt_style(self):
        params = [{"format": "tpe"},
                  {"preprocessing/test": "preproc"},
                  {"network/test": "network"},
                  ({"conv-layer-1": "conv-1"},
                   {"conv-layer-2": "conv-2"},
                   {"conv-layer-3": "conv-3"}),
                  ({"fc-layer-1": "fc-1"},
                   {"fc-layer-2": "fc-2"},
                   {"fc-layer-3": "fc-3"}),
                  ]
        expected_params = ({"test": "preproc"},
                           ["conv-1", "conv-2", "conv-3"],
                           ["fc-1", "fc-2", "fc-3"],
                           {"test": "network"})
        caffenet_params = tpe_sample_to_caffenet(params)
        self.assertEqual(caffenet_params, expected_params)

    def test_simple_hpolib_style(self):
        params = {"format": "tpe",
                  "preprocessing/test": "preproc",
                  "network/test": "network",
                  "conv-layer-1": "conv-1",
                  "conv-layer-2": "conv-2",
                  "conv-layer-3": "conv-3",
                  "fc-layer-1": "fc-1",
                  "fc-layer-2": "fc-2",
                  "fc-layer-3": "fc-3",
                  }
        expected_params = ({"test": "preproc"},
                           ["conv-1", "conv-2", "conv-3"],
                           ["fc-1", "fc-2", "fc-3"],
                           {"test": "network"})
        caffenet_params = tpe_sample_to_caffenet(params)
        self.assertEqual(caffenet_params, expected_params)


if __name__ == '__main__':
    unittest.main()
