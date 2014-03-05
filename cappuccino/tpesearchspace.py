from cappuccino.convnetsearchspace import ConvNetSearchSpace, Parameter
from hyperopt import hp
from hyperopt.pyll import scope
from paramutil import flatten_to_leaves, construct_parameter_tree_from_labels, group_layers
from math import log

"""
Unit-tests
----------
TODO: How could we unit-test the conversion between spaces?
 * Test that the leaves are either: float, int, Parameter and other values that we expected
 * Test number of elements in returned from the space
 * Test whether necessary elements are part of the hyperparameter space
 * Test conversion of very simple space, that contains all the necessary elements.
"""


def encode_tree_path(label, escape_char, item_name):
    """returns the nested tree path."""
    assert(escape_char not in item_name), "%s contains the escape char: %s" % (item_name, escape_char)
    nested_label = label + escape_char + item_name
    return nested_label


def parameter_to_tpe(label, parameter):
    """returns the parameter in TPE format."""
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
                                             item,
                                             escape_char_depth,
                                             escape_char_choice)
            converted_space[nested_label] = converted_item
        return converted_space
    if isinstance(subspace, list):
        items = []
        for item in subspace:
            assert("type" in item)
            item_type = item["type"]
            item_label = encode_tree_path(label, escape_char_choice, item_type)
            items.append(subspace_to_tpe(item_label,
                                         item,
                                         escape_char_depth,
                                         escape_char_choice))
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


def convnet_space_to_tpe(convnet_space):
    """
        Convert a search space defined as ConvNetSearchSpace
        to the TPE format.

        returns: search space in the TPE format.
    """
    assert(isinstance(convnet_space, ConvNetSearchSpace))
    params = []

    preprocessing_params = convnet_space.get_preprocessing_parameter_subspace()
    params.append(subspace_to_tpe("preprocessing", preprocessing_params))

    network_params = convnet_space.get_network_parameter_subspace()
    if isinstance(network_params["num_conv_layers"], Parameter):
        assert network_params["num_conv_layers"].min_val == 0
    if isinstance(network_params["num_fc_layers"], Parameter):
        assert network_params["num_fc_layers"].min_val == 1

    #in hyperopt we will represent the number of conv layers as a choice object
    #that's why we can strip them here:
    #num_conv_layers = network_params.pop("num_conv_layers")
    #num_fc_layers = network_params.pop("num_fc_layers")

    network_param_subspace = subspace_to_tpe("network", network_params)
    params.append(network_param_subspace)

    #Convolutional layers:
    conv_layer_subspaces = []

    for layer_id in range(1, convnet_space.max_conv_layers+1):
        conv_layer_params = convnet_space.get_conv_layer_subspace(layer_id)
        label = "conv-layer-%d" % (layer_id)
        conv_layer_subspace = subspace_to_tpe(label,
                                              conv_layer_params)
        conv_layer_subspaces.append(conv_layer_subspace)


    #to stay consistent with the fc layers we reverse the order, see below
    conv_layer_subspaces.reverse()

    conv_layers_combinations = get_stacked_layers_subspace(conv_layer_subspaces)

#    conv_layers_combinations.insert(0, []) #no conv layers
#    if isinstance(num_conv_layers, int):
#        #fixed number of layers
#        conv_layers_space = conv_layers_combinations[num_conv_layers]
#    else:
#        conv_layers_space = hp.choice('num_conv_layers', conv_layers_combinations)

    #Unfortunately scope.switch is not supported by the converter!
    conv_layers_space = scope.switch(scope.int(network_param_subspace["network/num_conv_layers"]),
                                     [],#no conv layers
                                     *conv_layers_combinations)


    params.append(conv_layers_space)

    #Fully connected layers
    fc_layer_subspaces = []

    for layer_id in range(1, convnet_space.max_fc_layers+1):
        fc_layer_params = convnet_space.get_fc_layer_subspace(layer_id)
        label = "fc-layer-%d" % (layer_id)
        fc_layer_subspace = subspace_to_tpe(label,
                                                    fc_layer_params)
        fc_layer_subspaces.append(fc_layer_subspace)

    """
        We always want the last layer to show up, because it has special parameters.
        [[fc3], [fc2, fc3], [fc1, fc2, fc3]]
    """
    fc_layer_subspaces.reverse()

    fc_layers_combinations = get_stacked_layers_subspace(fc_layer_subspaces)

#    if isinstance(num_fc_layers, int):
#        #fixed number of layers
#        fc_layers_space = fc_layers_combinations[num_fc_layers]
#    else:
#        fc_layers_space = hp.choice("num_fc_layers",
#                                    fc_layers_combinations)

    fc_layers_space = scope.switch(scope.int(network_param_subspace["network/num_fc_layers"]),
                                     None,#no fc layers
                                     *fc_layers_combinations)

    params.append(fc_layers_space)

    return params


def tpe_sample_to_caffenet(params):
    """
        Convert a parameter space sample from hyperopt
        into a format that caffe convnet understands.

        e.g.
        ({'preprocessing/mirror': {'preprocessing/mirror@off/type': 'off'}, 'preprocessing/crop': {'preprocessing/crop@none/type': 'none'}}, {'network/lr_policy': {'network/lr_policy@step/gamma': 0.9996274933033295, 'network/lr_policy@step/stepsize': 7.0, 'network/lr_policy@step/type': 'step'}, 'network/num_conv_layers': 0.0, 'network/momentum': 0.3701715323213701, 'network/lr': 0.0014660061654956985, 'network/weight_decay': 0.004909933297030581, 'network/num_fc_layers': 1.0}, (), ({'fc-layer-3/dropout': {'fc-layer-3/dropout/use_dropout': False}, 'fc-layer-3/weight-filler': {'fc-layer-3/weight-filler@gaussian/type': 'gaussian', 'fc-layer-3/weight-filler@gaussian/std': 2.9735970882233542e-05}, 'fc-layer-3/weight-lr-multiplier': 6.467391614940038, 'fc-layer-3/bias-lr-multiplier': 4.594272344706693, 'fc-layer-3/bias-filler': {'fc-layer-3/bias-filler@const-zero/type': 'const-zero'}, 'fc-layer-3/num_output': 10, 'fc-layer-3/activation': 'none', 'fc-layer-3/type': 'fc'},), ({'fc-layer-3/dropout': {'fc-layer-3/dropout/use_dropout': False}, 'fc-layer-3/weight-filler': {'fc-layer-3/weight-filler@gaussian/type': 'gaussian', 'fc-layer-3/weight-filler@gaussian/std': 2.9735970882233542e-05}, 'fc-layer-3/weight-lr-multiplier': 6.467391614940038, 'fc-layer-3/bias-lr-multiplier': 4.594272344706693, 'fc-layer-3/bias-filler': {'fc-layer-3/bias-filler@const-zero/type': 'const-zero'}, 'fc-layer-3/num_output': 10, 'fc-layer-3/activation': 'none', 'fc-layer-3/type': 'fc'},))
    """
    # Note, we first flatten and then go back to a tree
    # This is due to the fact that hpolib can only handle
    # flat parameters. Therefore the labels contain the full
    # tree path.
    flattened_params = flatten_to_leaves(params)
    param_tree = construct_parameter_tree_from_labels(flattened_params)
    caffe_convnet_params = group_layers(param_tree)
    return caffe_convnet_params

