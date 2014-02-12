from cappuccino import ConvNetSearchSpace, Parameter
from hyperopt import hp
from hyperopt.pyll import scope
from math import log

"""
TODO:
use the scope for determining the number of layers:
    e.g.: 
from hyperopt.pyll import scope

depth = hp.quniform('depth', 0, 5, 1)
space = scope.switch(scope.int(depth),
    {'depth': 0, 'log_base_epsilon_0': log_base_epsilon_0, 'weight_norm_0': weight_norm_0, 'dropout_0': dropout_0},

    {'depth': 1, 'log_base_epsilon_0': log_base_epsilon_0, 'weight_norm_0': weight_norm_0, 'dropout_0': dropout_0,
 ...
 }

"""

"""
TODO: go from class to function design:
        convert_to_tpe(space, ...)
"""

def encode_tree_path(label, escape_char, item_name):
    assert(escape_char not in item_name), "%s contains the escape char: %s" % (item_name, escape_char)
    nested_label = label + escape_char + item_name
    return nested_label


def parameter_to_tpe(label, parameter):
    """
        returns the parameter in TPE format.
    """
    if parameter.is_int:
        if parameter.log_scale:
            return hp.qloguniform(label,
                                  log(parameter.min_val),
                                  log(parameter.max_val),
                                  1)
        else:
            return hp.quniform(label,
                               parameter.min_val,
                               parameter.max_val,
                               1)
    else:
        if parameter.log_scale:
            return hp.loguniform(label,
                                 log(parameter.min_val),
                                 log(parameter.max_val))
        else:
            return hp.uniform(label,
                                 parameter.min_val,
                                 parameter.max_val)


def subspace_to_tpe(label, subspace, escape_char_depth = "/", escape_char_choice = "@"):
    """
        Recursively convert the search space defined by dicts, lists and Parameters
        into a TPE equivalent search space.

        label: The label for the current subspace.
        subspace: The subspace of the global search space.
        escape_char_depth: The escape char for encoding the tree path into the parameter label.
        escape_char_choice: The escape char for encoding the tree path into the parameter label.
    """
    if isinstance(subspace, dict):
        converted_space = {}
        for item_name, item in subspace.iteritems():
            nested_label = encode_tree_path(label, escape_char_depth, item_name)
            converted_item = subspace_to_tpe(nested_label,
                                                  item)
            converted_space[nested_label] = converted_item
        return converted_space
    if isinstance(subspace, list):
        items = []
        for item in subspace:
            assert("type" in item)
            item_type = item["type"]
            item_label = encode_tree_path(label, escape_char_choice, item_type)
            items.append(subspace_to_tpe(item_label, item))
        return hp.choice(label, items)
    if isinstance(subspace, Parameter):
        return parameter_to_tpe(label, subspace)
    else:
        return subspace


def get_stacked_layers_subspace(layer_subspaces):
    """
        all combinations of layers: 1, 2, 3 etc
        e.g. [[conv1],[conv1, conv2], [conv1, conv2, conv3]]
    """
    layer_combinations = [[] for i in range(len(layer_subspaces))]
    for num_layers in range(1, len(layer_subspaces)+1):
        for current_layer in range(0, num_layers):
            layer_combinations[num_layers-1].append(layer_subspaces[current_layer])
#        pp = pprint.PrettyPrinter(indent=4)
#        pp.pprint(layer_combinations)

    return layer_combinations


def to_tpe(convnet_space):
    """
        Convert a search space defined as ConvNetSearchSpace
        to the TPE format.
    """
#    assert(isinstance(convnet_space, ConvNetSearchSpace))
    params = []

    network_params = convnet_space.get_network_parameter_subspace()
    assert network_params["num_conv_layers"].min_val == 0
    assert network_params["num_fc_layers"].min_val == 1
    network_param_subspace = subspace_to_tpe("network", network_params)
    params.append(network_param_subspace)

    #Convolutional layers:
    conv_layer_subspaces = []

    for layer_id in range(convnet_space.max_conv_layers):
        conv_layer_params = convnet_space.get_conv_layer_subspace(layer_id)
        label = "conv-layer-%d" % (layer_id+1)
        conv_layer_subspace = subspace_to_tpe(label,
                                                        conv_layer_params)
        conv_layer_subspaces.append(conv_layer_subspace)

    conv_layers_combinations = get_stacked_layers_subspace(conv_layer_subspaces)
#    conv_layers_space = hp.choice("conv-layers",
#                                  conv_layers_combinations)

    conv_layers_space = scope.switch(scope.int(network_param_subspace["network/num_conv_layers"]),
                                     None,#no conv layers
                                     *conv_layers_combinations)
    params.append(conv_layers_space)

    #Fully connected layers
    fc_layer_subspaces = []

    for layer_id in range(convnet_space.max_fc_layers):
        fc_layer_params = convnet_space.get_fc_layer_subspace(layer_id)
        label = "fc-layer-%d" % (layer_id+1)
        fc_layer_subspace = subspace_to_tpe(label,
                                                    fc_layer_params)
        fc_layer_subspaces.append(fc_layer_subspace)

    """
        We always want the last layer to show up, because it has special parameters.
        [[fc3], [fc2, fc3], [fc1, fc2, fc3]]
    """
    fc_layer_subspaces.reverse()

    fc_layers_combinations = get_stacked_layers_subspace(fc_layer_subspaces)
#    fc_layers_space = hp.choice("fc-layers",
#                                 fc_layers_combinations)
    fc_layers_space = scope.switch(scope.int(network_param_subspace["network/num_fc_layers"]),
                                     None,#no fc layers
                                     *fc_layers_combinations)
    params.append(fc_layers_space)


    params.append(fc_layers_space)
    return params

