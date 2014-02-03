from caffe.proto import caffe_pb2

import pprint
import copy

#smac:

#hyperopt:
import hyperopt.pyll.stochastic
from hyperopt import fmin, tpe, hp, STATUS_OK

"""
    TODO: Create random caffe configuration
"""


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
                 input_dimensions=(100, 1, 1)):
        """
            max_conv_layers: maximum number of convolutional layers
            max_fc_layers: maximum number of fully connected layers
            input_dimensions: dimensions of the data input
                              in case of image data: channels x width x height
        """
        self.max_conv_layers = max_conv_layers
        self.max_fc_layers = max_fc_layers
        self.input_dimensions = input_dimensions

        for layer_id in range(self.max_conv_layers):
            pass

    def get_network_parameter_subspace(self):
        params = {}
        params["lr"] = Parameter(0, 0.8, is_int=False)
        params["momentum"] = Parameter(0, 1, is_int=False)
        params["weight_decay"] = Parameter(0, 0.1, is_int=False)
        return params

    def get_conv_layer_subspace(self):
        params = {}
        params["type"] = "conv"
        params["kernelsize"] = Parameter(2, 10, is_int=True)
        params["num_output"] = Parameter(5, 500, is_int=True)
        params["stride"] = Parameter(1, 10, is_int=True)
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

    def get_fc_layer_subspace(self):
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
        #TODO: dropout on/off
        params["dropout"] = Parameter(0.1, 0.9, is_int=False)
        return params


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
        fc_layer_params = self.get_fc_layer_subspace()

        for layer_id in range(self.max_fc_layers):
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

class CaffeConvNet(object):
    def __init__(self, params, train_file, valid_file, mean_file=None, batch_size=128):
        """
            Parameters of the network as defined by ConvNetSearchSpace.

            params: parameters to the network
            train_file: the training data leveldb file
            valid_file: the validation data leveldb file
            mean_file: mean per dimesnion leveldb file

        """
        self.train_file = train_file
        self.valid_file = valid_file
        self.mean_file = mean_file
        self.batch_size = batch_size
        self.convert_params_to_caffe_network(params)

    def convert_params_to_caffe_network(self, params):
        """
            Converts the given parameters into a caffe network configuration.
        """
        self.caffe_net = caffe_pb2.NetParameter()
        self.solver = caffe_pb2.SolverParameter()

        self.create_data_layer()

        all_conv_layers_params, all_fc_layers_params, network_params = params

        prev_layer_name = "data"
        for i, conv_layer_params in enumerate(all_conv_layers_params):
            current_layer_base_name = "conv_layer%d_" % i
            prev_layer_name = self.create_conv_layer(current_layer_base_name,
                                                     prev_layer_name,
                                                     conv_layer_params)

        for fc_layer_params in all_fc_layers_params:
            current_layer_base_name = "fc_layer%d_" % i
            prev_layer_name = self.create_fc_layer(current_layer_base_name,
                                                   prev_layer_name,
                                                   fc_layer_params)

        self.create_network_parameters(network_params)

        self.create_train_valid_networks()

        print str(self.caffe_net_train)

    def create_train_valid_networks(self):
        """
            Given the self.caffe_net base network we create
            different version for training, testing and predicting.
        """
        self.caffe_net_train = copy.deepcopy(self.caffe_net)
        self.caffe_net_train.layers[0].layer.source = self.train_file
        if self.mean_file:
            self.caffe_net_train.layers[0].layer.meanfile = self.mean_file

        self.caffe_net_validation = copy.deepcopy(self.caffe_net)
        self.caffe_net_validation.layers[0].layer.source = self.valid_file
        if self.mean_file:
            self.caffe_net_validation.layers[0].layer.meanfile = self.mean_file


    def create_data_layer(self):
        data_layer = self.caffe_net.layers.add()
        data_layer.layer.name = "data"
        data_layer.layer.type = "data"
        data_layer.layer.batchsize = self.batch_size
        data_layer.top.append("data")
        data_layer.top.append("label")

    def create_conv_layer(self, current_layer_base_name, prev_layer_name, params):
        """
            Generate a caffe layer from the given parameters.

            Note: one logical layer will be converted to multiple caffe layers.
        """

        #Convolution
        assert params.pop("type") == "conv"
        caffe_conv_layer = self.caffe_net.layers.add()
        current_layer_name = current_layer_base_name + "conv"
        caffe_conv_layer.layer.name = current_layer_name
        caffe_conv_layer.layer.kernelsize = int(params.pop("kernelsize")) 
        caffe_conv_layer.layer.num_output = int(params.pop("num_output"))
        caffe_conv_layer.layer.stride = int(params.pop("stride"))
        weight_filler_params = params.pop("weight-filler")
        weight_filler = caffe_conv_layer.layer.weight_filler
        for param, param_val in weight_filler_params.iteritems():
            setattr(weight_filler, param, param_val)

        caffe_conv_layer.layer.bias_filler.type = "constant"
        caffe_conv_layer.layer.bias_filler.value = 0
#        bias_filler_params = params.pop("bias-filler")
#        bias_filler = caffe_conv_layer.layer.bias_filler
#        for param, param_val in bias_filler_params.iteritems():
#            setattr(bias_filler, param, param_val)
        caffe_conv_layer.layer.blobs_lr.append(params.pop("weight-lr-multiplier"))
        caffe_conv_layer.layer.blobs_lr.append(params.pop("bias-lr-multiplier"))

        caffe_conv_layer.layer.weight_decay.append(params.pop("weight-weight-decay_multiplier"))
        caffe_conv_layer.layer.weight_decay.append(params.pop("bias-weight-decay_multiplier"))

        caffe_conv_layer.bottom.append(prev_layer_name)
        caffe_conv_layer.top.append(current_layer_name)

        prev_layer_name = current_layer_name

        # Relu
        caffe_relu_layer = self.caffe_net.layers.add()

        current_layer_name = current_layer_base_name + "relu"
        caffe_relu_layer.layer.name = current_layer_name
        caffe_relu_layer.bottom.append(prev_layer_name)
        caffe_relu_layer.top.append(current_layer_name)

        prev_layer_name = current_layer_name

        # Pooling
        caffe_pool_layer = self.caffe_net.layers.add()

        current_layer_name = current_layer_base_name + "pool"
        caffe_pool_layer.layer.name = current_layer_name
        caffe_pool_layer.bottom.append(prev_layer_name)
        caffe_pool_layer.top.append(current_layer_name)
        caffe_pool_layer.layer.pool = caffe_pool_layer.layer.MAX
        caffe_pool_layer.layer.kernelsize = 3
        caffe_pool_layer.layer.stride = 2

        prev_layer_name = current_layer_name

        #TODO; normlization layer
        #TODO; padding layer
 
        assert len(params) == 0, "More convolution parameters given than needed: " + str(params)

        return prev_layer_name

    def create_fc_layer(self, current_layer_base_name, prev_layer_name, params):
        caffe_fc_layer = self.caffe_net.layers.add()

        current_layer_name = current_layer_base_name + "fc"
        caffe_fc_layer.layer.name = current_layer_name
        caffe_fc_layer.bottom.append(prev_layer_name)
        caffe_fc_layer.top.append(current_layer_name)

        weight_filler_params = params.pop("weight-filler")
        weight_filler = caffe_fc_layer.layer.weight_filler
        for param, param_val in weight_filler_params.iteritems():
            setattr(weight_filler, param, param_val)

        caffe_fc_layer.layer.bias_filler.type = "constant"
        caffe_fc_layer.layer.bias_filler.value = 0

        prev_layer_name = current_layer_name

        return prev_layer_name

    def create_network_parameters(self, params):
        self.solver.base_lr = params.pop("lr")
        self.solver.momentum = params.pop("momentum")
        self.solver.weight_decay = params.pop("weight_decay")

        assert len(params) == 0, "More solver parameters given than needed: " + str(params)




space = TPEConvNetSearchSpace()
tpe_space = TPEConvNetSearchSpace().get_tpe_search_space()
print "TPE search space"
print tpe_space
print "Search space samples:"
for i in range(0,10):
    print hyperopt.pyll.stochastic.sample(tpe_space)

def test_fun(kwargs):
    print "Test fun called, parameters:"
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(kwargs)
    caffe = CaffeConvNet(kwargs, "train-leveldb", "test-leveldb", "mean-leveldb", 128)
    return {'loss': 0, 'status': STATUS_OK}

best = fmin(test_fun, space=tpe_space, algo=tpe.suggest, max_evals=10)

if False:
    caffe_net = caffe_pb2.NetParameter()
    caffe_net.name = "CaffeConvNet"
    prev_layer_name = "data"

#Notice: each logical layer consists of multiple caffe layers

#conv layers
    for i in xrange(0,3):
        current_layer_base_name = "layer%d_" % i

        #Convolution
        caffe_conv_layer = caffe_net.layers.add()
        current_layer_name = current_layer_base_name + "conv"
        caffe_conv_layer.layer.name = current_layer_name
        caffe_conv_layer.layer.num_output = 64
        caffe_conv_layer.layer.kernelsize = 5
        caffe_conv_layer.layer.stride = 1

        caffe_conv_layer.bottom.append(prev_layer_name)
        caffe_conv_layer.top.append(current_layer_name)

        prev_layer_name = current_layer_name

        #ReLU
        caffe_relu_layer = caffe_net.layers.add()

        current_layer_name = current_layer_base_name + "relu"
        caffe_relu_layer.layer.name = current_layer_name
        caffe_relu_layer.bottom.append(prev_layer_name)
        caffe_relu_layer.top.append(current_layer_name)

        prev_layer_name = current_layer_name

        #Pooling
        caffe_pool_layer = caffe_net.layers.add()

        current_layer_name = current_layer_base_name + "pool"
        caffe_pool_layer.layer.name = current_layer_name
        caffe_pool_layer.bottom.append(prev_layer_name)
        caffe_pool_layer.top.append(current_layer_name)
        caffe_pool_layer.layer.pool = caffe_pool_layer.layer.MAX
        caffe_pool_layer.layer.kernelsize = 3
        caffe_pool_layer.layer.stride = 2

        prev_layer_name = current_layer_name
        #TODO: norm layer
        #TODO: padding layer

#fc layers
    for i in xrange(0,2):
        caffe_ip_layer = caffe_net.layers.add()

        current_layer_name = current_layer_base_name + "ip"
        caffe_ip_layer.layer.name = current_layer_name
        caffe_ip_layer.bottom.append(prev_layer_name)
        caffe_ip_layer.top.append(current_layer_name)

        caffe_ip_layer.layer.weight_filler.type = "gaussian"
        caffe_ip_layer.layer.weight_filler.std = 0.01

        caffe_ip_layer.layer.bias_filler.type = "constant"
        caffe_ip_layer.layer.bias_filler.value = 0

        prev_layer_name = current_layer_name

#print caffe_net.SerializeToString()
    print str(caffe_net)

