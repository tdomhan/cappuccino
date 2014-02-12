
#TODO: make batch size a parameter!


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


#TODO: make the depth a hyperparameter that can also be set to be constant
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
        #note: the input_dimensions are not used right now
        self.input_dimensions = input_dimensions

    def get_network_parameter_subspace(self):
        params = {}
        params["num_conv_layers"] = Parameter(0, self.max_conv_layers, is_int=True)
        #note we need at least one fc layer
        params["num_fc_layers"] = Parameter(1, self.max_fc_layers, is_int=True)
        params["lr"] = Parameter(1e-16, 0.1, is_int=False, log_scale=True)
        params["momentum"] = Parameter(0, 0.99, is_int=False)
        params["weight_decay"] = Parameter(0.000005, 0.005, is_int=False, log_scale=True)
        fixed_policy = {"type": "fixed"}
        exp_policy = {"type": "exp",
                      "gamma": Parameter(0.8, 1., is_int=False)}
        step_policy = {"type": "step",
                      "gamma": Parameter(0.8, 1., is_int=False),
                      "stepsize": Parameter(2, 20, is_int=True)}
        inv_policy = {"type": "inv",
                      "gamma": Parameter(0.001, 100, is_int=False),
                      "power": Parameter(0.000001, 1, is_int=False)}
        params["lr_policy"] = [fixed_policy,
                               exp_policy,
                               step_policy,
                               inv_policy]
        return params

    def get_conv_layer_subspace(self, layer_idx):
        params = {}
        params["type"] = "conv"
        params["kernelsize"] = Parameter(2, 6, is_int=True)
        params["num_output"] = Parameter(5, 500, is_int=True)
        #params["stride"] = Parameter(1, 5, is_int=True)
        params["stride"] =  1
        params["weight-filler"] = [{"type": "gaussian",
                                    "std": Parameter(0.00001, 1, log_scale=True, is_int=False)},
                                   {"type": "xavier",
                                    "std": Parameter(0.00001, 1, log_scale=True, is_int=False)}]
        params["weight-lr-multiplier"] = Parameter(0.01, 10, is_int=False)
        params["bias-lr-multiplier"] = Parameter(0.01, 10, is_int=False)
        params["weight-weight-decay_multiplier"] = Parameter(0.01, 10, is_int=False)
        params["bias-weight-decay_multiplier"] = Parameter(0.01, 10, is_int=False)
        no_pooling = {"type": "none"}
        max_pooling = {"type": "max",
                       "stride": Parameter(1, 3, is_int=True),
                       "kernelsize": Parameter(2, 5, is_int=True)}
        #average pooling:
        ave_pooling = {"type": "ave",
                       "stride": Parameter(1, 3, is_int=True),
                       "kernelsize": Parameter(2, 5, is_int=True)}

        #        stochastic_pooling = {"type": "stochastic"}
        params["pooling"] = [no_pooling,
                             max_pooling,
                             ave_pooling]
        return params

    def get_fc_layer_subspace(self, layer_idx):
        """
            Get the subspace of fully connect layer parameters.

            layer_idx: the zero-based index of the layer
        """
        params = {}
        params["type"] = "fc"
        params["weight-filler"] = [{"type": "gaussian",
                                    "std": Parameter(0.00001, 1, log_scale=True, is_int=False)},
                                   {"type": "xavier",
                                    "std": Parameter(0.00001, 1, log_scale=True, is_int=False)}]
        params["weight-lr-multiplier"] = Parameter(0.01, 10, is_int=False)
        params["bias-lr-multiplier"] = Parameter(0.01, 10, is_int=False)
        params["weight-weight-decay_multiplier"] = Parameter(0.01, 10, is_int=False)
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

    def get_parameter_count(self):
        #TODO: calculate the parameter count
        pass


class LeNet5(ConvNetSearchSpace):
    """
        A search space, where the architecture is fixed
        and only the network parameters, like the learning rate,
        are tuned.

        For the definition of LeNet-5 see:
        http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
    """

    def __init__(self):
        super(LeNet5, self).__init__(max_conv_layers=2,
                                     max_fc_layers=2,
                                     num_classes=10,
                                     input_dimensions=(32,1,1))

    def get_network_parameter_subspace(self):
        #we don't change the network parameters
        return super(LeNet5, self).get_network_parameter_subspace()

    def get_conv_layer_subspace(self, layer_idx):
        params = super(LeNet5, self).get_conv_layer_subspace(layer_idx)
        if layer_idx == 0:
            params["kernelsize"] = 5
            params["stride"] = 1
            params["num_output"] = 6
            params["pooling"] = {"type": "max",
                                 "stride": 2,
                                 "kernelsize": 2}
        elif layer_idx == 1:
            params["kernelsize"] = 5
            params["stride"] = 1
            params["num_output"] = 16
            params["pooling"] = {"type": "max",
                                 "stride": 2,
                                 "kernelsize": 2}
 
        return params

    def get_fc_layer_subspace(self, layer_idx):
        params = super(LeNet5, self).get_fc_layer_subspace(layer_idx)
        if layer_idx == 0:
            params["num_output"] = 120
        elif layer_idx == 1:
            params["num_output"] = 84

        return params

