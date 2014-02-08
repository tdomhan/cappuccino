
class Parameter:
    def __init__(self, min_val, max_val, is_int=False, log_scale=False):
        assert(min_val < max_val)
        self.min_val = min_val
        self.max_val = max_val
        self.is_int = is_int
        self.log_scale = log_scale

    def __str__(self):
        return "Parameter(min: %d, max: %d, is_int: %d, log_scale: %d)" % (self.min_val,
                                                            self.max_val,
                                                            self.is_int,
                                                            self.log_scale)
    
    def __repr__(self):
        return self.__str__()


class ConvNetSearchSpace(object):
    """
        Search space for a convolutional neural network.

        The search space is defined by dicts, lists and Parameters.

        Each dict is a collection of Parameters.
        Each list is a choice between multiple parameters,
            each of which are a dict, that must contain a
            value for "type".

        The search space is an arbitrary concatenation of the
            above elements.
    """
    def __init__(self,
                 max_conv_layers=3,
                 max_fc_layers=3,
                 num_classes=10,
                 input_dimensions=(100, 1, 1)):
        """
            max_conv_layers: maximum number of convolutional layers
            max_fc_layers: maximum number of fully connected layers
            num_classes: the number of output classes
            input_dimensions: dimensions of the data input
                              in case of image data: channels x width x height
        """
        self.max_conv_layers = max_conv_layers
        self.max_fc_layers = max_fc_layers
        self.num_classes = num_classes
        self.input_dimensions = input_dimensions

    def get_network_parameter_subspace(self):
        params = {}
        params["lr"] = Parameter(0, 0.8, is_int=False)
        params["momentum"] = Parameter(0, 1, is_int=False)
        params["weight_decay"] = Parameter(0, 0.1, is_int=False)
        return params

    def get_conv_layer_subspace(self, layer_idx):
        params = {}
        params["type"] = "conv"
        params["kernelsize"] = Parameter(2, 5, is_int=True)
        params["num_output"] = Parameter(5, 500, is_int=True)
        params["stride"] = Parameter(1, 3, is_int=True)
        params["weight-filler"] = [{"type": "gaussian",
                                    "std": Parameter(0.01, 10, is_int=False)},
                                   {"type": "xavier",
                                    "std": Parameter(0.01, 10, is_int=False)}]
        params["weight-lr-multiplier"] = Parameter(0.01, 10, is_int=False)
        params["bias-lr-multiplier"] = Parameter(0.01, 10, is_int=False)
        params["weight-weight-decay_multiplier"] = Parameter(0.01, 10, is_int=False)
        params["bias-weight-decay_multiplier"] = Parameter(0.01, 10, is_int=False)
        #TODO: pooling layer
        return params

    def get_fc_layer_subspace(self, layer_idx):
        """
            Get the subspace of fully connect layer parameters.

            layer_idx: the zero-based index of the layer
        """
        params = {}
        params["type"] = "fc"
        params["weight-filler"] = [{"type": "gaussian",
                                    "std": Parameter(0.01, 10, is_int=False)},
                                   {"type": "xavier",
                                    "std": Parameter(0.01, 10, is_int=False)}]
        params["weight-lr-multiplier"] = Parameter(0.01, 10, is_int=False)
        params["bias-lr-multiplier"] = Parameter(0.01, 10, is_int=False)
        params["bias-weight-decay_multiplier"] = Parameter(0.01, 10, is_int=False)
        params["bias-weight-decay_multiplier"] = Parameter(0.01, 10, is_int=False)


        last_layer = layer_idx == self.max_fc_layers-1
        if not last_layer:
            params["num_output"] = Parameter(10, 1000, is_int=True)
            params["activation"] = "relu"
            params["dropout"] = [{"type": "dropout",
                                  "use_dropout": True,
                                  "dropout_ratio": Parameter(0.05, 0.95, is_int=False)},
                                 {"type": "no_dropout",
                                  "use_dropout": False}]
        else:
            params["num_output"] = self.num_classes
            params["activation"] = "none"
            params["dropout"] = {"use_dropout": False}

        return params

