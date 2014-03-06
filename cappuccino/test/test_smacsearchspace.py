import unittest
import re
from cappuccino.convnetsearchspace import ConvNetSearchSpace, Parameter
from cappuccino.convnetsearchspace import Pylearn2Convnet
from cappuccino.smacsearchspace import subspace_to_smac, convnet_space_to_smac
from cappuccino.smacsearchspace import smac_space_to_str
from cappuccino.smacsearchspace import SMACParameter
from cappuccino.smacsearchspace import SMACCategorialParameter
from cappuccino.smacsearchspace import SMACNumericalParameter


class TestSubspaceToSmac(unittest.TestCase):

    def test_simple_subspace_conversion(self):
        """Let's build up a small search space and try to convert it to SMAC.
        """
        search_space = {"param1": Parameter(0, 10),
                        "param2": Parameter(10, 100, log_scale=True),
                        "param3": Parameter(0, 100,
                                            default_val=20,
                                            is_int=True),
                        }
        smac_space = []
        subspace_to_smac("testspace", smac_space, search_space)

        self.assertEqual(len(smac_space), 3)

        for parameter in smac_space:
            self.assertTrue(isinstance(parameter, SMACParameter), "expected SMACParameter.")
            self.assertTrue(len(parameter.depends_on) == 0)
            if parameter.name == "testspace/param1":
                self.assertEqual(parameter.min_val, 0)
                self.assertEqual(parameter.max_val, 10)
            elif parameter.name == "testspace/param2":
                self.assertEqual(parameter.min_val, 10)
                self.assertEqual(parameter.max_val, 100)
                self.assertEqual(parameter.log_scale, True)
            elif parameter.name == "testspace/param3":
                self.assertEqual(parameter.min_val, 0)
                self.assertEqual(parameter.max_val, 100)
                self.assertEqual(parameter.default, 20)
            else:
                self.assertTrue(False, "unkown parameter name")

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

        space = {"condspace": [space1, space2, space3]}
 
        smac_space = []
        subspace_to_smac("testspace", smac_space, space,
                                      escape_char_depth='/',
                                      escape_char_choice='@')

        #we have 4 parameters
        #+ 1 extra parameter that encode the conditionality
        self.assertEqual(len(smac_space), 4+1)

        for parameter in smac_space:
            self.assertTrue(isinstance(parameter, SMACParameter))
        params = {parameter.name: parameter for parameter in smac_space}

        self.assertEqual(params["testspace/condspace@space1/param1"].min_val, 0)
        self.assertEqual(params["testspace/condspace@space1/param1"].max_val, 10)

        self.assertTrue(isinstance(params["testspace/condspace/type"], SMACCategorialParameter))

        #the type parameter is converted into a categorical parameter with only a single value:
        self.assertTrue(params["testspace/condspace/type"].default, "space1")

        self.assertTrue(isinstance(params["testspace/condspace/type"], SMACCategorialParameter))
        #there are two conditional subspaces:
        self.assertEqual(params["testspace/condspace/type"].values, ["space1", "space2", "space3"])

        #check for dependencies:
        self.assertEqual(len(params["testspace/condspace@space1/param1"].depends_on), 1)
        param_dependency = params["testspace/condspace@space1/param1"].depends_on[0]
        self.assertEqual(param_dependency.parent_name, "testspace/condspace/type")

        self.assertEqual(len(param_dependency.values), 1)
        self.assertEqual(param_dependency.values[0], "space1")

 
class TestSmacNumericalParameter(unittest.TestCase):

    def test_int(self):
        #passing float to an int parameter:
        self.assertRaises(AssertionError,
                          SMACNumericalParameter,
                          "test", float(0), float(1), default=0.5,
                          is_int=True, log_scale=False)

class TestSMACParamToString(unittest.TestCase):

    def test_numerical_param_to_string(self):
        param = SMACNumericalParameter("test", 0, 1, default=0.5, is_int=False, log_scale=False)
        match = re.match(r"([\w\/]+) \[([0-9\.]+), ([0-9\.]+)\] \[([0-9\.]+)\]", str(param))
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "test")
        self.assertAlmostEqual(float(match.group(2)), 0)
        self.assertAlmostEqual(float(match.group(3)), 1)
        self.assertAlmostEqual(float(match.group(4)), 0.5)

        #make sure to not get floating point numbers if it's an int
        param = SMACNumericalParameter("test123", 0, 10, default=5, is_int=True, log_scale=False)
        match = re.match(r"([\w\/]+) \[([0-9]+), ([0-9]+)\] \[([0-9]+)\]", str(param))
        self.assertIsNotNone(match, "int number probably contains floating points.")
        self.assertEqual(match.group(1), "test123")
        self.assertAlmostEqual(int(match.group(2)), 0)
        self.assertAlmostEqual(int(match.group(3)), 10)
        self.assertAlmostEqual(int(match.group(4)), 5)
        
        param = SMACNumericalParameter("test", 0, 1, default=0.5, is_int=False, log_scale=True)
        match = re.match(r"([\w\/]+) \[([0-9\.]+), ([0-9\.]+)\] \[([0-9\.]+)\]l", str(param))
        self.assertIsNotNone(match, "doesn't match")
        self.assertEqual(match.group(1), "test")
        self.assertAlmostEqual(float(match.group(2)), 0)
        self.assertAlmostEqual(float(match.group(3)), 1)
        self.assertAlmostEqual(float(match.group(4)), 0.5)

        param = SMACNumericalParameter("test", 0, 1, default=0, is_int=True, log_scale=True)
        match = re.match(r"([\w\/]+) \[([0-9\.]+), ([0-9\.]+)\] \[([0-9\.]+)\]il", str(param))
        self.assertIsNotNone(match, "doesn't match")
        self.assertEqual(match.group(1), "test")
        self.assertAlmostEqual(int(match.group(2)), 0)
        self.assertAlmostEqual(int(match.group(3)), 1)
        self.assertAlmostEqual(int(match.group(4)), 0)

    def test_categorical_param_to_string(self):
        param = SMACCategorialParameter("test", values=[1,2,3], default=3)
        match = re.match(r"([\w\/]+) \{([0-9\.]+), ([0-9\.]+), ([0-9\.]+)\} \[([0-9\.]+)\]", str(param))
        self.assertIsNotNone(match, "doesn't match")
        self.assertEqual(match.group(1), "test")
        self.assertAlmostEqual(float(match.group(2)), 1)
        self.assertAlmostEqual(float(match.group(3)), 2)
        self.assertAlmostEqual(float(match.group(4)), 3)
        self.assertAlmostEqual(float(match.group(5)), 3)
 

class TestConvNetSpaceConversion(unittest.TestCase):


    def setUp(self):
        self.conv_params_to_check = ["kernelsize",
                                     "num_output_x_128",
                                     "stride",
                                     "weight-filler",
                                     "bias-filler",
                                     "weight-lr-multiplier",
                                     "bias-lr-multiplier",
                                     "weight-weight-decay_multiplier",
                                     "bias-weight-decay_multiplier",
                                     "norm",
                                     "pooling",
                                     "dropout"]
        self.fc_params_to_check = ["weight-filler",
                                    "bias-filler",
                                    "weight-lr-multiplier",
                                    "bias-lr-multiplier",
                                    "activation"]

    def test_convnet_space_conversion(self):
        space = ConvNetSearchSpace((3, 32, 32),
                                   max_conv_layers=2,
                                   max_fc_layers=3)

        self.space = space
        smac_space = convnet_space_to_smac(space)
        self.smac_space = smac_space

        self.assertGreater(len(smac_space), 0)
        for param in smac_space:
            self.assertTrue(isinstance(param, SMACParameter), "expected SMACParameter.")

        params = {parameter.name: parameter for parameter in smac_space}

        #check that we have all necessary parameters
        self.assertTrue("network/lr" in params)
        self.assertTrue("network/weight_decay" in params)
        self.assertTrue("network/momentum" in params)
        self.assertTrue("network/num_conv_layers" in params)
        self.assertTrue("network/num_fc_layers" in params)

        self.assertTrue("preprocessing/augment/type" in params)

        # test that we get the number of layers right:
        network_params_orig = space.get_network_parameter_subspace()
        num_conv_layers_smac = params["network/num_conv_layers"]
        num_conv_layers_orig = network_params_orig["num_conv_layers"]
        self.assertEqual(num_conv_layers_smac.default,
                         num_conv_layers_orig.default_val)
 
        num_fc_layers_smac = params["network/num_fc_layers"]
        num_fc_layers_orig = network_params_orig["num_fc_layers"]
        self.assertEqual(num_fc_layers_smac.default,
                         num_fc_layers_orig.default_val)
 

        #check the dependencies of the layer parameters are correct
        self.check_layer_dependencies(
                                 min_layers=1,
                                 max_layers = space.max_conv_layers,
                                 layer_name="conv-layer",
                                 depends_on="network/num_conv_layers",
                                 params_to_check=self.conv_params_to_check,
                                 params=params)

        self.check_layer_dependencies(
                                 min_layers=1,
                                 max_layers = space.max_fc_layers,
                                 layer_name="fc-layer",
                                 depends_on="network/num_fc_layers",
                                 params_to_check = self.fc_params_to_check,
                                 params=params)


    def check_layer_dependencies(self,
                                 min_layers,
                                 max_layers,
                                 layer_name,
                                 depends_on,
                                 params_to_check,
                                 params):
        """Check the dependencies between the number of layers and
        the individual layer parameters.

            The last layer is always present:
            layers:     [[], [fc3], [fc2, fc3], [fc1, fc2, fc3]]
            num_layers: [0,   1,     2,          3]
            depends_on(layer_id): [[],[3],[2,3],[1,2,3]]
            or:
                fc1 depends on num_layers = 3
                fc2 depends on num_layers = 2 or 3
                fc3 depends on num_layers = 1, 2 or 3
        """
        depends_on_values = []
        depends_on_values_iter = iter(range(max_layers, 0, -1))
        for layer_idx in range(1, max_layers+1):
            depends_on_values.append(next(depends_on_values_iter))
            for param_name in params_to_check:
                #we check the kernelsize, because it should always be present:
                layer_param_name = "%s-%d/%s" % (layer_name, layer_idx, param_name)
                if layer_param_name not in params:
                    layer_param_name = "%s-%d/%s/type" % (layer_name, layer_idx, param_name)
                    self.assertTrue(layer_param_name in params, "couldn't find param %s" % layer_param_name)
                layer_param = params[layer_param_name]
                #it should only depend on the number of conv layers:
                self.assertEqual(len(layer_param.depends_on), 1)
                self.assertEqual(layer_param.depends_on[0].parent_name, depends_on)
                self.assertEqual(layer_param.depends_on[0].values, depends_on_values)


    def test_fixed_architecture_space_conversion(self):
        """ In this test case the architecture, that is the number
        of conv and fc layers is fixed.
        """
        space = Pylearn2Convnet()
        smac_space = convnet_space_to_smac(space)

        self.assertGreater(len(smac_space), 0)
        for param in smac_space:
            self.assertTrue(isinstance(param, SMACParameter), "expected SMACParameter.")

        params = {parameter.name: parameter for parameter in smac_space}

        #check the format indicator exists:
        self.assertEqual(params["format"].default, "smac")
        self.assertEqual(len(params["format"].values), 1)

        #check that we have all necessary parameters
        self.assertTrue("network/lr" in params)
        self.assertTrue("network/weight_decay" in params)
        self.assertTrue("network/momentum" in params)
        self.assertTrue("network/num_conv_layers" in params)
        self.assertTrue("network/num_fc_layers" in params)

        self.assertEqual(len(params["network/num_conv_layers"].values), 1, "number of conv layers should be fixed to a single value")
        self.assertEqual(len(params["network/num_fc_layers"].values), 1, "number of fc layers should be fixed to a single value")

        #check that there are not dependencies on the conv and fc parameters
        for layer_idx in range(1, space.max_conv_layers+1):
            for param_name in self.conv_params_to_check:
                layer_param_name = "conv-layer-%d/%s" % (layer_idx, param_name)
                if layer_param_name not in params:
                    layer_param_name = "conv-layer-%d/%s/type" % (layer_idx, param_name)
                    self.assertTrue(layer_param_name in params, "couldn't find param %s" % layer_param_name)
                layer_param = params[layer_param_name]
                #shoudln't depend on anything, because the size is fixed
                self.assertEqual(len(layer_param.depends_on), 0)

        for layer_idx in range(1, space.max_fc_layers+1):
            for param_name in self.fc_params_to_check:
                layer_param_name = "fc-layer-%d/%s" % (layer_idx, param_name)
                if layer_param_name not in params:
                    layer_param_name = "fc-layer-%d/%s/type" % (layer_idx, param_name)
                    self.assertTrue(layer_param_name in params, "couldn't find param %s" % layer_param_name)
                layer_param = params[layer_param_name]
                #shoudln't depend on anything, because the size is fixed
                self.assertEqual(len(layer_param.depends_on), 0)

 
    def test_flat_input(self):
        """What happens if the input is flat, that means, not an image.
        Make sure we don't have any conv layers, no cropping preprossing.
        TODO: maybe we should put this in the test for the convnetsearchspace though.
        """
        pass

if __name__ == '__main__':
    unittest.main()
