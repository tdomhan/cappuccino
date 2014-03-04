import ast

def value_to_literal(value):
    """
        Tries to convert a value to either a 
        float, int or boolean.
    """
    try:
        return ast.literal_eval(value)
    except:
        return value

def construct_parameter_tree_from_labels(params,
                                         escape_char_depth = "/",
                                         escape_char_choice = "@"):
    """
        Given a dictionary of {label: parameter}, where label encodes the
        depth in the parameter tree, e.g. level0/level1/level2
        as well as the selected choice parametere, e.g. level0#choice@selectedval
        We convert this back to a tree of parameter values in order to feed it
        to a target algorithm.
    """
    param_tree = {} 

    for label, value in params.iteritems():
        value = value_to_literal(value)

        tree_path = label.split(escape_char_depth)
        # walk through the tree:
        current_node = param_tree
        for idx, step in enumerate(tree_path):
            node_type = None
            if escape_char_choice in step:
                choices = step.split(escape_char_choice)
                assert(len(choices) == 2), "Only one choice field allowed per level."
                step, node_type = choices
            if idx == len(tree_path)-1:
                #last, set value
                current_node[step] = value
                assert node_type == None, ('Can\'t have a value on a choice node without a separate label.'
                                           'e.g. this is illegal: {"convlayer0#weight-filler@gaussian": 2}')
            else:
                #go deeper
                if not step in current_node:
                    current_node[step] = {}
                current_node = current_node[step]
                if node_type is not None:
                    current_node["type"] = node_type

    return param_tree

def group_layers(params_tree):
    """
        params_tree: a dictionary with the conv-layer, fc-layer and network parameters.
        groups them into those three groups.
        convolutional layers are stacked on top of each other in the right order.
    """
    conv_layers = []
    fc_layers = []
    for name, val in params_tree.iteritems():
        if name.startswith("conv-layer-"):
            idx = int(name.split("-")[2]) 
            conv_layers.append((idx,val))
        elif name.startswith("fc-layer-"):
            idx = int(name.split("-")[2])
            fc_layers.append((idx,val))
        else:
            pass

    conv_layers = sorted(conv_layers)
    fc_layers = sorted(fc_layers)

    #remove indices:
    if len(conv_layers) > 0:
        conv_layers = list(zip(*conv_layers)[1])
    #remove indices:
    if len(fc_layers):
        fc_layers = list(zip(*fc_layers)[1])

    network_params = params_tree["network"]

    preprocessing_params = params_tree["preprocessing"]

    return (preprocessing_params, conv_layers, fc_layers, network_params)


def flatten_to_leaves(params):
    """
        Parameters are given as a tree dict-of-dicts.
        The tree will be flattened by returning a dict of
        the leave key-values.

        Example:
            {"level1": {"level2-leaf-1": value, "level2-leaf-2": another_value}}
            will return:
            {"level2-leaf-1": value, "level2-leaf-2": another_value}

        Note: The leave keys are expected to be unique!
    """
    if isinstance(params, dict):
        flattened_params = {}
        for key, value in params.iteritems():
            if isinstance(value, dict):
                flattened_leave = flatten_to_leaves(value)
                for leave_key, leave_value in flattened_leave.iteritems():
                    assert leave_key not in flattened_params, "The keys at the leaves must be unique! %s" % leave_key
                    flattened_params[leave_key] = leave_value
            else:
                assert key not in flattened_params
                flattened_params[key] = value
        return flattened_params
    elif isinstance(params, (list, tuple)):
        flattened_params = {}
        for value in params:
            flattened_leave = flatten_to_leaves(value)
            assert isinstance(flattened_leave, dict)
            for leave_key, leave_value in flattened_leave.iteritems():
                assert leave_key not in flattened_params, "The keys at the leaves must be unique! %s" % leave_key
                flattened_params[leave_key] = leave_value
        return flattened_params
    else:
        return params



def hpolib_to_caffenet(params):
    """
        Convert the parameters provided by HPOLib into the caffenet format.
    """
    param_tree = construct_parameter_tree_from_labels(params)
    caffe_convnet_params = group_layers(param_tree)
    return caffe_convnet_params

