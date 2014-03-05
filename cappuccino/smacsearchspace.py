from cappuccino.convnetsearchspace import ConvNetSearchSpace, Parameter
from cappuccino.paramutil import construct_parameter_tree_from_labels
from cappuccino.paramutil import group_layers
import copy


def encode_tree_path(label, escape_char, item_name):
    """returns the nested tree path."""
    assert(escape_char not in item_name), "%s contains the escape character"
    nested_label = label + escape_char + item_name
    return nested_label


class SMACDependency(object):
    def __init__(self, parent_name, values):
        self.parent_name = parent_name
        #the values that the parent parameter needs to be in.
        #the parent should be a categorical variable.
        self.values = values

    def __str__(self):
        """SMAC config string of this dependency. """
        #<child name> | <parent name> in {<parent val1>, ..., <parent valK>}
        return " | %s in {%s}" % (self.parent_name,
                                  ", ".join(map(str, self.values)))

    def __repr__(self):
        return self.__str__()


class SMACParameter(object):
    def __init__(self, name):
        self.name = name
        #a list of SMACDependency objects:
        self.depends_on = []

    def __str__(self):
        """SMAC config file representation of this parameter."""
        return ""

    def __repr__(self):
        return self.__str__()


class SMACCategorialParameter(SMACParameter):
    def __init__(self, name, values, default=None):
        super(SMACCategorialParameter, self).__init__(name)
        self.type = "categorical"
        assert(len(values) > 0)
        self.values = values
        if default is None:
            #TODO: what to choose as the default
            self.default = values[0]
        else:
            self.default = default

    def __str__(self):
        values_string = ", ".join(map(str, self.values))
        return "%s {%s} [%s]" % (self.name, values_string, str(self.default))


class SMACNumericalParameter(SMACParameter):
    def __init__(self, name, min_val, max_val,
                 default=None, is_int=False, log_scale=False):
        super(SMACNumericalParameter, self).__init__(name)
        self.type = "numerical"
        self.min_val = min_val
        self.max_val = max_val
        assert default is not None
        self.default = default
        self.is_int = is_int
        if self.is_int:
            assert (type(self.min_val) is int,
                    "excepted int, got " + str(type(self.min_val)))
            assert (type(self.max_val) is int,
                    "excepted int, got " + str(type(self.max_val)))
            assert (type(self.default) is int,
                    "excepted int, got " + str(type(self.default)))
        self.log_scale = log_scale

    def __str__(self):
        if self.is_int:
            base_str = "%s [%d, %d] [%d]" % (self.name, self.min_val,
                                             self.max_val, self.default)
        else:
            base_str = "%s [%f, %f] [%f]" % (self.name, self.min_val,
                                             self.max_val, self.default)
        if self.is_int:
            base_str += "i"
        if self.log_scale:
            base_str += "l"
        return base_str


def subspace_to_smac(label, params, subspace,
                     dependencies=[], escape_char_depth="/",
                     escape_char_choice="@"):
    """
       Recursively convert the search space defined by dicts, lists and
       Parameters into a SMAC equivalent search space.

       label: The label for the current subspace.
       subspace: The subspace of the global search space.
       escape_char_depth: The escape char for encoding the tree path
                          into the parameter label.
       escape_char_choice: The escape char for encoding choice nodes.
    """
    if isinstance(subspace, dict):
        for item_name, item in subspace.iteritems():
            nested_label = encode_tree_path(label, escape_char_depth,
                                            item_name)
            #each child should have it's own dependencies
            dependencies = copy.deepcopy(dependencies)
            subspace_to_smac(nested_label, params, item, dependencies,
                             escape_char_depth, escape_char_choice)
    elif isinstance(subspace, list):
        #add another variable to encode the choice:
        types = [item["type"] for item in subspace if "type" in item]
        choice_param_name = label+escape_char_depth+"type"
        choice_param = SMACCategorialParameter(name=choice_param_name,
                                               values=types)
        params.append(choice_param)
        for idx, item in enumerate(subspace):
            assert("type" in item)
            item_type = item.pop("type")
            dependency = SMACDependency(choice_param.name, values=[item_type])
            item_dependencies = copy.deepcopy(dependencies)
            item_dependencies.append(dependency)
            item_label = encode_tree_path(label, escape_char_choice, item_type)
            subspace_to_smac(item_label, params, item, item_dependencies,
                             escape_char_depth, escape_char_choice)
    elif isinstance(subspace, Parameter):
        parameter = subspace
        parameter = SMACNumericalParameter(label,
                                           parameter.min_val,
                                           parameter.max_val,
                                           default=parameter.default_val,
                                           is_int=parameter.is_int,
                                           log_scale=parameter.log_scale)

        parameter.depends_on = copy.deepcopy(dependencies)
        params.append(parameter)
    else:
        const = subspace
        #we represent constants as categorical variables with a single category
        parameter = SMACCategorialParameter(label,
                                            values=[str(const)])
        parameter.depends_on = dependencies

        params.append(parameter)


def convnet_space_to_smac(convnet_space):
    """Convert the given ConvNetSearchSpace into SMAC format.
    """
    #assert(isinstance(convnet_space, ConvNetSearchSpace))
    params = []

    preprocessing_params = convnet_space.get_preprocessing_parameter_subspace()
    subspace_to_smac("preprocessing", params, preprocessing_params)

    network_params = convnet_space.get_network_parameter_subspace()

    #need to be converted into categorical variables:
    num_conv_layers = network_params.pop("num_conv_layers")
    if isinstance(num_conv_layers, Parameter):
        assert num_conv_layers.min_val == 0
        values = range(num_conv_layers.min_val, num_conv_layers.max_val+1)
        num_conv_layers_param = SMACCategorialParameter(
            "network/num_conv_layers",
            values=values)
        conv_layers_fixed = False
    elif isinstance(num_conv_layers, int):
        #const value = categorical with a singe choice
        num_conv_layers_param = SMACCategorialParameter(
            "network/num_conv_layers",
            values=[num_conv_layers])
        conv_layers_fixed = True
    else:
        assert(False,
               "num_conv_layers either needs to be a Parameter or an int.")
    params.append(num_conv_layers_param)

    num_fc_layers = network_params.pop("num_fc_layers")
    if isinstance(num_fc_layers, Parameter):
        assert num_fc_layers.min_val == 1
        values = range(num_fc_layers.min_val, num_fc_layers.max_val+1)
        num_fc_layers_param = SMACCategorialParameter("network/num_fc_layers",
                                                      values=values)
        fc_layers_fixed = False
    elif isinstance(num_fc_layers, int):
        #const value = categorical with a singe choice
        num_fc_layers_param = SMACCategorialParameter("network/num_fc_layers",
                                                      values=[num_fc_layers])
        fc_layers_fixed = True
    else:
        assert False, "num_fc_layers either needs to be a Parameter or an int."
    params.append(num_fc_layers_param)

    subspace_to_smac("network", params, network_params)

    #conv layers
    conv_layer_depends_on = [max(num_conv_layers_param.values)]
    for layer_id in range(1, convnet_space.max_conv_layers+1):
            conv_layer_params = convnet_space.get_conv_layer_subspace(layer_id)
            label = "conv-layer-%d" % (layer_id)
            conv_layer_subspace = []
            subspace_to_smac(label, conv_layer_subspace, conv_layer_params)
            #add dependency:
            if not conv_layers_fixed:
                for param in conv_layer_subspace:
                    num_layers_dependency = SMACDependency(
                        num_conv_layers_param.name,
                        values=copy.deepcopy(conv_layer_depends_on))
                    param.depends_on.append(num_layers_dependency)

            params.extend(conv_layer_subspace)
            conv_layer_depends_on.append(min(conv_layer_depends_on)-1)

    #fc layers
    fc_layer_depends_on = [max(num_fc_layers_param.values)]
    for layer_id in range(1, convnet_space.max_fc_layers+1):
            fc_layer_params = convnet_space.get_fc_layer_subspace(layer_id)
            label = "fc-layer-%d" % (layer_id)
            fc_layer_subspace = []
            subspace_to_smac(label, fc_layer_subspace, fc_layer_params)
            #add dependency:
            if not fc_layers_fixed:
                for param in fc_layer_subspace:
                    num_layers_dependency = SMACDependency(
                        num_fc_layers_param.name,
                        values=copy.deepcopy(fc_layer_depends_on))
                    param.depends_on.append(num_layers_dependency)

            params.extend(fc_layer_subspace)
            fc_layer_depends_on.append(min(fc_layer_depends_on)-1)

    return params


def smac_space_to_str(smac_space):
    lines = []
    for param in smac_space:
        lines.append(str(param))
        for dependency in param.depends_on:
            lines.append(param.name + str(dependency))
    return "\n".join(lines)


def smac_sample_to_caffenet(params):
    """
        Convert a sample from smac into the format needed by the CaffeNet.
        params: dict of the form: {param_name: value}
                (param_name encodes the tree path, e.g.
                    "network/conv-layer-1/weight-filler")
    """
    param_tree = construct_parameter_tree_from_labels(params)
    caffe_convnet_params = group_layers(param_tree)
    return caffe_convnet_params


if __name__ == "__main__":
    #TODO: make this a converter script
    pass
