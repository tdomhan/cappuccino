from cappuccino import ConvNetSearchSpace, Parameter
from hyperopt import hp

class TPEConvNetSearchSpace(ConvNetSearchSpace):
    def __init__(self, **kwargs):
        super(TPEConvNetSearchSpace, self).__init__(**kwargs)

    def convnet_space_to_tpe(self, label, subspace):
        """
            Convert the search space defined by dicts, lists and Parameters
            into a TPE equivalent search space.
        """
        if isinstance(subspace, dict):
            converted_space = {}
            for item_name, item in subspace.iteritems():
                converted_item = self.convnet_space_to_tpe(label+"-"+item_name,
                                                           item)
                converted_space[item_name] = converted_item
            return converted_space
        if isinstance(subspace, list):
            items = []
            for item in subspace:
                assert("type" in item)
                item_type = item["type"]
                item_label = label+"-"+item_type
                items.append(self.convnet_space_to_tpe(item_label, item))
            return hp.choice(label, items)
        if isinstance(subspace, Parameter):
            return self.parameter_to_tpe(label, subspace)
        else:
            return subspace


    def parameter_to_tpe(self, label, parameter):
        """
        returns the parameter in TPE format
        """
        if parameter.is_int:
            if parameter.log_scale:
                return hp.qloguniform(label,
                                      parameter.min_val,
                                      parameter.max_val,
                                      1)
            else:
                return hp.quniform(label,
                                   parameter.min_val,
                                   parameter.max_val,
                                   1)
        else:
            if parameter.log_scale:
                return hp.loguniform(label,
                                  parameter.min_val,
                                  parameter.max_val)
            else:
                return hp.uniform(label,
                                     parameter.min_val,
                                     parameter.max_val)

    def get_stacked_layers_subspace(self, layer_subspaces):
        """
            all combinations of layers: 1, 2, 3 etc
            e.g. [[conv1],[conv1, conv2], [conv1, conv2, conv3]]
        """
        layer_combinations = [[] for i in range(len(layer_subspaces))]
        for num_layers in range(1, len(layer_subspaces)+1):
            for current_layer in range(0, num_layers):
                print len(layer_combinations[num_layers-1])
                layer_combinations[num_layers-1].append(layer_subspaces[current_layer])
#        pp = pprint.PrettyPrinter(indent=4)
#        pp.pprint(layer_combinations)

        return layer_combinations
 

    def get_tpe_search_space(self):
        params = []
        #Convolutional layers:
        conv_layer_subspaces = []
        conv_layer_params = self.get_conv_layer_subspace()

        for layer_id in range(self.max_conv_layers):
            label = "conv-layer-%d" % (layer_id)
            conv_layer_subspace = self.convnet_space_to_tpe(label,
                                                            conv_layer_params)
            conv_layer_subspaces.append(conv_layer_subspace)

        conv_layers_combinations = self.get_stacked_layers_subspace(conv_layer_subspaces)
        conv_layers_space = hp.choice("conv-layers",
                                      conv_layers_combinations)
        params.append(conv_layers_space)

        #Fully connected layers
        fc_layer_subspaces = []

        for layer_id in range(self.max_fc_layers):
            fc_layer_params = self.get_fc_layer_subspace(layer_id)
            label = "fc-layer-%d" % (layer_id)
            fc_layer_subspace = self.convnet_space_to_tpe(label,
                                                        fc_layer_params)
            fc_layer_subspaces.append(fc_layer_subspace)

        fc_layers_combinations = self.get_stacked_layers_subspace(fc_layer_subspaces)
        fc_layers_space = hp.choice("fc-layers",
                                     fc_layers_combinations)
        params.append(fc_layers_space)

        network_param_subspace = self.get_network_parameter_subspace()
        params.append(self.convnet_space_to_tpe("network", network_param_subspace))

        return params


