
def construct_parameter_tree_from_labels(params, escape_char_depth = "/", escape_char_choice = "@"):
    """
        Given a dictionary of {label: parameter}, where label encodes the
        depth in the parameter tree, e.g. level0#level1#level2
        as well as the selected choice parametere, e.g. level0#choice@selectedval
        We convert this back to a tree of parameter values in order to feed it
        to a target algorithm.
    """
    param_tree = {} 

    for label, value in params.iteritems():
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
                if node_type:
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
    def stack_layers(layers):
        layers_stacked = [None] * len(layers)
        for idx, val in layers:
            layers_stacked[idx] = val
        return layers_stacked
 
    conv_layers = stack_layers(conv_layers)
    fc_layers = stack_layers(fc_layers)
    network_params = params_tree["network"]

    return (conv_layers, fc_layers, network_params)



