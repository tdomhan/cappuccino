from caffe.proto import caffe_pb2
import copy

class CaffeConvNet(object):
    """
        Runs a caffe convnet with the given parameters
    """
    def __init__(self, params, train_file, valid_file, mean_file=None, batch_size=128):
        """
            Parameters of the network as defined by ConvNetSearchSpace.

            params: parameters to the network
            train_file: the training data leveldb file
            valid_file: the validation data leveldb file
            mean_file: mean per dimesnion leveldb file

        """
        self._train_file = train_file
        self._valid_file = valid_file
        self._mean_file = mean_file
        self._batch_size = batch_size

        self._convert_params_to_caffe_network(params)
        self._create_train_valid_networks()

    def _convert_params_to_caffe_network(self, params):
        """
            Converts the given parameters into a caffe network configuration.
        """
        self._caffe_net = caffe_pb2.NetParameter()
        self._solver = caffe_pb2.SolverParameter()

        self._create_data_layer()

        all_conv_layers_params, all_fc_layers_params, network_params = params

        prev_layer_name = "data"
        for i, conv_layer_params in enumerate(all_conv_layers_params):
            current_layer_base_name = "conv_layer%d_" % i
            prev_layer_name = self._create_conv_layer(current_layer_base_name,
                                                     prev_layer_name,
                                                     conv_layer_params)

        for i, fc_layer_params in enumerate(all_fc_layers_params):
            current_layer_base_name = "fc_layer%d_" % i
            prev_layer_name = self._create_fc_layer(current_layer_base_name,
                                                   prev_layer_name,
                                                   fc_layer_params)

        self._create_network_parameters(network_params)


    def _create_train_valid_networks(self):
        """
            Given the self._caffe_net base network we create
            different version for training, testing and predicting.
        """
        self._caffe_net_train = copy.deepcopy(self._caffe_net)
        self._caffe_net_train.layers[0].layer.source = self._train_file
        if self._mean_file:
            self._caffe_net_train.layers[0].layer.meanfile = self._mean_file

        self._caffe_net_validation = copy.deepcopy(self._caffe_net)
        self._caffe_net_validation.layers[0].layer.source = self._valid_file
        if self._mean_file:
            self._caffe_net_validation.layers[0].layer.meanfile = self._mean_file


    def _create_data_layer(self):
        data_layer = self._caffe_net.layers.add()
        data_layer.layer.name = "data"
        data_layer.layer.type = "data"
        data_layer.layer.batchsize = self._batch_size
        data_layer.top.append("data")
        data_layer.top.append("label")

    def _create_conv_layer(self, current_layer_base_name, prev_layer_name, params):
        """
            Generate a caffe layer from the given parameters.

            Note: one logical layer will be converted to multiple caffe layers.
        """

        #Convolution
        assert params.pop("type") == "conv"
        caffe_conv_layer = self._caffe_net.layers.add()
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
        caffe_relu_layer = self._caffe_net.layers.add()

        current_layer_name = current_layer_base_name + "relu"
        caffe_relu_layer.layer.name = current_layer_name
        caffe_relu_layer.bottom.append(prev_layer_name)
        caffe_relu_layer.top.append(current_layer_name)

        prev_layer_name = current_layer_name

        # Pooling
        caffe_pool_layer = self._caffe_net.layers.add()

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

    def _create_fc_layer(self, current_layer_base_name, prev_layer_name, params):
        caffe_fc_layer = self._caffe_net.layers.add()

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

    def _create_network_parameters(self, params):
        self._solver.base_lr = params.pop("lr")
        self._solver.momentum = params.pop("momentum")
        self._solver.weight_decay = params.pop("weight_decay")
        self._solver.train_net = self._train_file
        self._solver.test_net = self._valid_file
        self._solver.test_iter = 500
        self._solver.test_interval = 10
        self._solver.display = 10
        self._solver.snapshot = 10000000
        self._solver.snapshot_prefix = "caffenet"

        assert len(params) == 0, "More solver parameters given than needed: " + str(params)


