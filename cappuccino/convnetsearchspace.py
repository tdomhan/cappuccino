from math import log, exp


class Parameter:
    def __init__(self, min_val, max_val,
                 default_val = None,
                 is_int = False,
                 log_scale = False):
        assert(min_val < max_val)
        self.min_val = min_val
        self.max_val = max_val
        if default_val is None:
            if log_scale:
                self.default_val = exp(0.5 * log(min_val) + 0.5 * log(max_val))
            else:
                self.default_val = 0.5 * min_val + 0.5 * max_val
        else:
            assert(default_val <= max_val)
            assert(min_val <= default_val)
            self.default_val = default_val
        self.is_int = is_int
        if self.is_int:
            assert type(self.min_val) is int
            assert type(self.min_val) is int
            self.default_val = int(self.default_val)
        self.log_scale = log_scale

    def __str__(self):
        return "Parameter(min: %d, max: %d, default: %d, is_int: %d, log_scale: %d)" % (self.min_val,
                                                            self.max_val,
                                                            self.default_val,
                                                            self.is_int,
                                                            self.log_scale)
    
    def __repr__(self):
        return self.__str__()


class ConvNetSearchSpace(object):
    KERNEL_RELATIVE_MAX_SIZE = 0.25 # relative to the input image, how big can the kernel be?
    KERNEL_ABSOLUTE_MIN_SIZE = 2

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
                 input_dimension,
                 max_conv_layers=3,
                 max_fc_layers=3,
                 fc_layer_max_num_output_x_128=10,
                 conv_layer_max_num_output_x_128=5,
                 num_classes=10):
        """
            input_dimension: dimension of the data input
                             in case of image data: channels x width x height
            max_conv_layers: maximum number of convolutional layers
            max_fc_layers: maximum number of fully connected layers
            num_classes: the number of output classes
        """
        assert all([dim > 0 for dim in input_dimension]), "input dimension not >= 0"
        assert max_conv_layers >= 0
        self.max_conv_layers = max_conv_layers
        assert max_fc_layers >= 1
        self.max_fc_layers = max_fc_layers
        self.fc_layer_max_num_output_x_128 = fc_layer_max_num_output_x_128
        self.conv_layer_max_num_output_x_128 = conv_layer_max_num_output_x_128
        self.num_classes = num_classes
        self.input_dimension = input_dimension

        if (self.input_dimension[0] > 1 and
            self.input_dimension[1] == 1 and
            self.input_dimension[2] == 1):
            assert self.max_conv_layers == 0, "conv layers can only used when width/height are > 1"

    def get_preprocessing_parameter_subspace(self):
        params  = {}

        #the following is only possible in 2D/3D for square images
        if self.input_dimension[1] > 1 and self.input_dimension[2] == self.input_dimension[1]:
            augment_params = {"type": "augment"}
            im_size = self.input_dimension[1]
            # the size of the image after cropping
            max_crop_size = im_size / 2
            augment_params["crop_size"] = Parameter(0, int(0.8*max_crop_size), is_int=True)
            params["augment"] = [{"type": "none"},
                                 augment_params]
        else:
            print "note: not a square image, will not use data augmentation."
            params["augment"] = [{"type": "none"}]

        params["input_dropout"] = [{"type": "no_dropout"},
                                   {"type": "dropout",
                                    "dropout_ratio": Parameter(0.05, 0.95,
                                                          is_int=False)}]
 

        return params

    def get_network_parameter_subspace(self):
        params = {}
        if self.max_conv_layers == 0:
            params["num_conv_layers"] = 0
        else:
            params["num_conv_layers"] = Parameter(0,
                                                  self.max_conv_layers,
                                                  #default_val=self.max_conv_layers,
                                                  default_val=0,
                                                  is_int=True)
        #note we need at least one fc layer
        if self.max_fc_layers == 1:
            params["num_fc_layers"] = 1
        else:
            params["num_fc_layers"] = Parameter(1, # at least one layer that generates our output 
                                                self.max_fc_layers,
                                                #default_val=self.max_fc_layers,
                                                default_val=1,
                                                is_int=True)
        params["lr"] = Parameter(1e-10, 0.9, default_val=0.01,
                                 is_int=False, log_scale=True)
        params["momentum"] = Parameter(0, 0.99, default_val=0.6, is_int=False)
        params["weight_decay"] = Parameter(0.000005, 0.05, default_val=0.0005,
                                           is_int=False, log_scale=True)
        params["batch_size_train"] = Parameter(10, 1000, default_val=100, is_int=True)
        fixed_policy = {"type": "fixed"}
        """
            exp_policy:
                lr = base_lr * gamma^iteration
        """
        exp_policy = {"type": "exp",
                      "gamma": Parameter(0.8, 0.99999, is_int=False)}
        """
            step_policy:
            lr = base_lr * gamma^epoch
        """
        step_policy = {"type": "step",
                       "gamma": Parameter(0.05, 0.99, is_int=False),
                       "epochcount": Parameter(1, 50, is_int=True)}
        #TODO: make gamma relative to the epoch count??
        """
            inv_policy:
            lr = base_lr * (1+gamma*iter)^-power
        """
        inv_policy = {"type": "inv",
                      "gamma": Parameter(0.00000001, 10000,
                                         is_int=False,
                                         log_scale=True),
                      "power": Parameter(0.01, 1,
                                         is_int=False, log_scale = True)}
        """
            inv_bergstra_bengio:

            lr = base_lr * epochcount / max(epochcount, epoch)
        """
        inv_bergstra_bengio_policy = {"type": "inv_bergstra_bengio",
                                       "epochcount": Parameter(1, 50, is_int=True)}
        params["lr_policy"] = [fixed_policy,
                               exp_policy,
                               step_policy,
                               inv_policy,  
                               inv_bergstra_bengio_policy]
        return params

    def get_conv_layer_subspace(self, layer_idx):
        """Get the parameters for the given layer.

        layer_idx: 1 based layer idx
        """
        assert layer_idx > 0 and layer_idx <= self.max_conv_layers

        assert self.input_dimension[2] == self.input_dimension[1], "only square kernels supported right now."

        max_kernel_size = max(int(ConvNetSearchSpace.KERNEL_RELATIVE_MAX_SIZE * self.input_dimension[1]),
                              ConvNetSearchSpace.KERNEL_ABSOLUTE_MIN_SIZE)

        params = {}
        params["padding"] = [{"type": "none"},
                             {"type": "zero-padding",
                              #percentage of the size of the kernel
                              "size": Parameter(0., 1.0, is_int=False)}]

        params["type"] = "conv"
        #TODO: make dependent on the image size
        #TODO: try to alternatively parametrize relative to the input image size (percentage)
        params["kernelsize"] = Parameter(ConvNetSearchSpace.KERNEL_ABSOLUTE_MIN_SIZE,
            max_kernel_size, is_int=True)
        #reducing the search spacing by only allowing multiples of 128
        params["num_output_x_128"] = Parameter(1, self.conv_layer_max_num_output_x_128,
            is_int=True)
        #params["stride"] = Parameter(1, 5, is_int=True)
        params["stride"] =  1
        params["weight-filler"] = [{"type": "gaussian",
                                    "std": Parameter(0.00001,.1,
                                                     default_val=0.01,
                                                     log_scale=True,
                                                     is_int=False)},
                                   {"type": "xavier",
                                    "std": Parameter(0.00001,.1,
                                                     default_val=0.01,
                                                     log_scale=True,
                                                     is_int=False)}]
        params["bias-filler"] = [{"type": "const-zero"},
                                 {"type": "const-one"}]
        params["weight-lr-multiplier"] = Parameter(0.01, 10, default_val=1,
                                                   is_int=False,
                                                   log_scale=True)
        params["bias-lr-multiplier"] = Parameter(0.01, 10, default_val=2,
                                                 is_int=False, log_scale=True)
        params["weight-weight-decay_multiplier"] = Parameter(0.01, 10,
                                                             default_val=1,
                                                             is_int=False,
                                                             log_scale=True)
        params["bias-weight-decay_multiplier"] = Parameter(0.001, 10,
                                                           default_val=0.001,
                                                           is_int=False,
                                                           log_scale=True)

        #TODO: check lrn parameter ranges
        normalization_params = {"type": "lrn",
                                "local_size": Parameter(2, 6, is_int=True),
                                "alpha": Parameter(0.00001, 0.001,
                                                   default_val=0.0001,
                                                   log_scale=True),
                                "beta": Parameter(0.6, 0.9, default_val=0.75)}
        params["norm"] = [normalization_params,
                          {"type": "none"}]

        no_pooling = {"type": "none"}
        max_pooling = {"type": "max",
                       "stride": Parameter(1, max_kernel_size/2, is_int=True),
                       "kernelsize": Parameter(ConvNetSearchSpace.KERNEL_ABSOLUTE_MIN_SIZE,
                            max_kernel_size, is_int=True)}
        #average pooling:
        ave_pooling = {"type": "ave",
                       "stride": Parameter(1, max_kernel_size/2, is_int=True),
                       "kernelsize": Parameter(ConvNetSearchSpace.KERNEL_ABSOLUTE_MIN_SIZE,
                            max_kernel_size, is_int=True)}

        #        stochastic_pooling = {"type": "stochastic"}
        params["pooling"] = [no_pooling,
                             max_pooling,
                             ave_pooling]

        params["dropout"] = [{"type": "dropout",
                              "dropout_ratio": Parameter(0.05, 0.95,
                                                         is_int=False)},
                             {"type": "no_dropout"}]
 
        return params

    def get_fc_layer_subspace(self, layer_idx):
        """
            Get the subspace of fully connect layer parameters.

            layer_idx: the one-based index of the layer
        """
        assert layer_idx > 0 and layer_idx <= self.max_fc_layers
        params = {}
        params["type"] = "fc"
        params["weight-filler"] = [{"type": "gaussian",
                                    "std": Parameter(0.000001, 0.1,
                                                     default_val=0.005,
                                                     log_scale=True,
                                                     is_int=False)},
                                   {"type": "xavier",
                                    "std": Parameter(0.000001, 0.1,
                                                     default_val=0.005,
                                                     log_scale=True,
                                                     is_int=False)}]
        params["bias-filler"] = [{"type": "const-zero"},
                                 {"type": "const-one"}]
        params["weight-lr-multiplier"] = Parameter(0.01, 10, default_val=1,
                                                   is_int=False,
                                                   log_scale=True)
        params["bias-lr-multiplier"] = Parameter(0.01, 10, default_val=2,
                                                 is_int=False, log_scale=True)
#        params["weight-weight-decay_multiplier"] = Parameter(0.01, 10, is_int=False)
#        params["bias-weight-decay_multiplier"] = Parameter(0.01, 10, is_int=False)


        last_layer = layer_idx == self.max_fc_layers
        if not last_layer:
            params["num_output_x_128"] = Parameter(1, self.fc_layer_max_num_output_x_128,
                is_int=True)
            params["activation"] = "relu"
            params["dropout"] = [{"type": "dropout",
                                  "dropout_ratio": Parameter(0.05, 0.95,
                                                             default_val=0.5,
                                                             is_int=False)},
                                 {"type": "no_dropout"}]
                                  
        else:
            params["num_output"] = self.num_classes
            params["activation"] = "none"
            params["dropout"] = {"type": "no_dropout"}

        return params

    def get_parameter_count(self):
        """
            How many parameters are there to be optimized.
        """
        count = 0
        def count_params(subspace):
            if isinstance(subspace, Parameter):
                return 1
            elif isinstance(subspace, dict):
                c = 0
                for key, value in subspace.iteritems():
                    c += count_params(value)
                return c
            elif isinstance(subspace, list):
                c = 1 #each list is a choice parameter
                for value in subspace:
                    c += count_params(value)
                return c
            else:
                return 0

        for layer_id in range(self.max_conv_layers):
            count += count_params(self.get_conv_layer_subspace(layer_id))

        for layer_id in range(self.max_fc_layers):
            count += count_params(self.get_fc_layer_subspace(layer_id))

        count += count_params(self.get_network_parameter_subspace())

        count += count_params(self.get_preprocessing_parameter_subspace())

        print "Total: ", count
        return count


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
                                     input_dimension=(32,1,1))

    def get_network_parameter_subspace(self):
        #we don't change the network parameters
        network_params = super(LeNet5, self).get_network_parameter_subspace()
        network_params["num_conv_layers"] = 2
        network_params["num_fc_layers"] = 2
        return network_params

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


class Cifar10CudaConvnet(ConvNetSearchSpace):
    """
        A search space, where the architecture is fixed
        and only the network parameters, like the learning rate,
        are tuned.

        The definition is inspired (but only loosly based on) by the cuda-convnet:
        https://code.google.com/p/cuda-convnet/
    """

    def __init__(self):
        super(Cifar10CudaConvnet, self).__init__(max_conv_layers=3,
                                                 max_fc_layers=1,
                                                 num_classes=10,
                                                 input_dimension=(3, 32, 32))

    def get_preprocessing_parameter_subspace(self):
        params = super(Cifar10CudaConvnet, self).get_preprocessing_parameter_subspace()
        params["augment"] = {"type": "augment",
                             "crop_size": 24}

        return params

    def get_network_parameter_subspace(self):
        #we don't change the network parameters
        network_params = super(Cifar10CudaConvnet, self).get_network_parameter_subspace()
        #fix the number of layers
        network_params["num_conv_layers"] = self.max_conv_layers
        network_params["num_fc_layers"] = self.max_fc_layers
        network_params["momentum"] = 0.8
        network_params["lr"] = 0.01
        network_params["weight_decay"] = 0.004

        #TODO: change this!
        network_params["lr_policy"] = {}
        network_params["lr_policy"]["type"] = "step"
        network_params["lr_policy"]["epochcount"] = 4
        network_params["lr_policy"]["gamma"] = 0.1

        return network_params

    def get_conv_layer_subspace(self, layer_idx):
        params = super(Cifar10CudaConvnet, self).get_conv_layer_subspace(layer_idx)
        #first parameters that are common to all layers:
        params["bias-filler"] = {"type": "const-zero"}

        params["weight-lr-multiplier"] = 1
        params["bias-lr-multiplier"] = 2
        params["bias-weight-decay_multiplier"] = 0

        params["norm"] = {"type": "none"}

        params["dropout"] = {"type": "no_dropout"} 

        if layer_idx == 0:
            params["weight-filler"] = {"type": "gaussian",
                                       "std": 0.0001}
 
            params["padding"] = {"type": "zero-padding",
                                 "size": 2}
            params["kernelsize"] = 5
            params["stride"] = 1
            params["num_output_x_128"] = 2
            params["pooling"] = {"type": "max",
                                 "stride": 2,
                                 "kernelsize": 3}

            params["weight-weight-decay_multiplier"] = 0  

        elif layer_idx == 1:
            params["weight-filler"] = {"type": "gaussian",
                                       "std": 0.01}
 
            params["padding"] = {"type": "zero-padding",
                                 "size": 1}
            params["kernelsize"] = 5
            params["stride"] = 1
            params["num_output_x_128"] = 2
            params["pooling"] = {"type": "max",
                                 "stride": 2,
                                 "kernelsize": 3}

            params["weight-weight-decay_multiplier"] = 0  
        elif layer_idx == 2:
            params["weight-filler"] = {"type": "gaussian",
                                       "std": 0.04}
 
            params["padding"] = {"type": "zero-padding",
                                 "size": 1}
            params["kernelsize"] = 3
            params["stride"] = 1
            params["num_output_x_128"] = 1
            params["pooling"] = {"type": "none"}

            params["weight-weight-decay_multiplier"] = 1  
        elif layer_idx == 3:
            params["weight-filler"] = {"type": "gaussian",
                                       "std": 0.04}
            params["padding"] = {"type": "zero-padding",
                                 "size": 1}
            params["kernelsize"] = 3
            params["stride"] = 1
            params["num_output_x_128"] = 1
            params["pooling"] = {"type": "none"}
 
            params["weight-weight-decay_multiplier"] = 1  

 
        return params

    def get_fc_layer_subspace(self, layer_idx):
        params = super(Cifar10CudaConvnet, self).get_fc_layer_subspace(layer_idx)
        assert layer_idx == 0
        params["weight-filler"] = {"type": "gaussian",
                                   "std": 0.01}
        params["dropout"] = {"use_dropout": "no_dropout"} 
        params["num_output"] = self.num_classes

        params["bias-filler"] = {"type": "const-zero"}

        params["weight-lr-multiplier"] = 1
        params["bias-lr-multiplier"] = 2
        #params["weight-weight-decay_multiplier"] = 1  
        #params["bias-weight-decay_multiplier"] = 0

        return params


class NoDataAugmentationConvNetSearchSpace(ConvNetSearchSpace):
    """Convnet search space, but with data augmentation disabled."""
    def __init__(self, **kwargs):
        super(NoDataAugmentationConvNetSearchSpace, self).__init__(**kwargs)

    def get_preprocessing_parameter_subspace(self):
        params = super(NoDataAugmentationConvNetSearchSpace, self).get_preprocessing_parameter_subspace()
        params["augment"] = {"type": "none"}
        return params


class Pylearn2Convnet(ConvNetSearchSpace):
    """
        A search space, where the architecture is fixed
        and only the network parameters, like the learning rate,
        are tuned.

        The definition is inspired (but only loosly based on) by the pylearn2 maxout paper architecture.
    """

    def __init__(self):
        super(Pylearn2Convnet, self).__init__(max_conv_layers=3,
                                                 max_fc_layers=2,
                                                 num_classes=10,
                                                 input_dimension=(3, 32, 32))


    def get_preprocessing_parameter_subspace(self):
        params = super(Pylearn2Convnet, self).get_preprocessing_parameter_subspace()
#        params["augment"] = {"type": "augment",
#                             "crop_size": 24}
        params["augment"] = {"type": "none"}

        return params


    def get_network_parameter_subspace(self):
        #we don't change the network parameters
        network_params = super(Pylearn2Convnet, self).get_network_parameter_subspace()
        #fix the number of layers
        network_params["num_conv_layers"] = self.max_conv_layers
        network_params["num_fc_layers"] = self.max_fc_layers
#        network_params["momentum"] = 0.55
#        network_params["lr"] = 0.17
#        network_params["weight_decay"] = 0.004

#        network_params["lr_policy"] = {}
#        network_params["lr_policy"]["type"] = "step"
#        network_params["lr_policy"]["epochcount"] = 1
#        network_params["lr_policy"]["gamma"] = 0.8

        return network_params


    def get_conv_layer_subspace(self, layer_idx):
        params = super(Pylearn2Convnet, self).get_conv_layer_subspace(layer_idx)
        #first parameters that are common to all layers:
        params["bias-filler"] = {"type": "const-zero"}

        params["weight-lr-multiplier"] = 0.05
        params["bias-lr-multiplier"] = 0.05
        params["weight-weight-decay_multiplier"] = 1  
        params["bias-weight-decay_multiplier"] = 0

        params["norm"] = {"type": "none"}

        params["weight-filler"] = {"type": "gaussian",
                                   "std": 0.005}

        params["dropout"] = {"type": "dropout",
                             "dropout_ratio": 0.5}

        if layer_idx == 0:
            params["padding"] = {"type": "zero-padding",
                                 "size": 4}
            params["kernelsize"] = 8
            params["stride"] = 1
            params["num_output_x_128"] = 1
            params["pooling"] = {"type": "max",
                                 "stride": 2,
                                 "kernelsize": 4}
        elif layer_idx == 1:
            params["padding"] = {"type": "zero-padding",
                                 "size": 3}
            params["kernelsize"] = 8
            params["stride"] = 1
            params["num_output_x_128"] = 2
            params["pooling"] = {"type": "max",
                                 "stride": 2,
                                 "kernelsize": 4}
        elif layer_idx == 2:
            params["padding"] = {"type": "zero-padding",
                                 "size": 3}
            params["kernelsize"] = 5
            params["stride"] = 1
            params["num_output_x_128"] = 2
            params["pooling"] = {"type": "max",
                                 "stride": 2,
                                 "kernelsize": 2}
 
        return params

    def get_fc_layer_subspace(self, layer_idx):
        params = super(Pylearn2Convnet, self).get_fc_layer_subspace(layer_idx)
        params["weight-filler"] = {"type": "gaussian",
                                   "std": 0.005}
 
        if layer_idx == 0:
            params["dropout"] = {"type": "dropout",
                                 "dropout_ratio": 0.5} 
            params["num_output_x_128"] = 4
        elif layer_idx == 1:
            pass

        params["bias-filler"] = {"type": "const-zero"}

        params["weight-lr-multiplier"] = 1
        params["bias-lr-multiplier"] = 1
        #params["weight-weight-decay_multiplier"] = 1  
        #params["bias-weight-decay_multiplier"] = 0

        return params

