from caffe.proto import caffe_pb2
from subprocess import check_output
import copy

class CaffeConvNet(object):
    """
        Runs a caffe convnet with the given parameters
    """
    def __init__(self, params,
                 train_file,
                 valid_file,
                 num_validation_set_batches,
                 mean_file=None,
                 batch_size_train = 128,
                 batch_size_valid = 100):
        """
            Parameters of the network as defined by ConvNetSearchSpace.

            params: parameters to the network
            train_file: the training data leveldb file
            valid_file: the validation data leveldb file
            num_validation_set_batches: num_validation_set_batches * batch_size_test = number of training examples
            mean_file: mean per dimesnion leveldb file
            batch_size_train: the batch size during training
            batch_size_test: the batch size during testing
        """
        self._train_file = train_file
        self._valid_file = valid_file
        self._mean_file = mean_file
        self._batch_size_train = batch_size_train
        self._batch_size_valid = batch_size_valid
        self._num_validation_set_batches = num_validation_set_batches

        self._base_name = "imagenet"

        self._train_network_file = self._base_name + "_train.prototxt"
        self._valid_network_file = self._base_name + "_valid.prototxt"
        self._solver_file = self._base_name + "_solver.prototxt"

        self._convert_params_to_caffe_network(params)
        self._create_train_valid_networks()

        self._serialize()

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
        # train network
        self._caffe_net_train = copy.deepcopy(self._caffe_net)
        self._caffe_net_train.layers[0].layer.source = self._train_file
        self._caffe_net_train.layers[0].layer.batchsize = self._batch_size_train
        if self._mean_file:
            self._caffe_net_train.layers[0].layer.meanfile = self._mean_file

        # add train loss:
        last_layer_top = self._caffe_net_train.layers[-1].top[0]
        loss_layer = self._caffe_net_train.layers.add()
        loss_layer.layer.name = "loss"
        loss_layer.layer.type = "softmax_loss"
        loss_layer.bottom.append(last_layer_top)
        loss_layer.bottom.append("label")

        # validation network:
        self._caffe_net_validation = copy.deepcopy(self._caffe_net)
        self._caffe_net_validation.layers[0].layer.source = self._valid_file
        self._caffe_net_validation.layers[0].layer.batchsize = self._batch_size_valid
        if self._mean_file:
            self._caffe_net_validation.layers[0].layer.meanfile = self._mean_file

        #softmax layer:
        last_layer_top = self._caffe_net_validation.layers[-1].top[0]
        prob_layer = self._caffe_net_validation.layers.add()
        prob_layer.layer.name = "prob"
        prob_layer.layer.type = "softmax"
        prob_layer.bottom.append(last_layer_top)
        prob_layer.top.append("prob")

        #accuracy layer:
        prob_layer = self._caffe_net_validation.layers.add()
        prob_layer.layer.name = "accuracy"
        prob_layer.layer.type = "accuracy"
        prob_layer.bottom.append("prob")
        prob_layer.bottom.append("label")
        prob_layer.top.append("accuracy")


    def _create_data_layer(self):
        data_layer = self._caffe_net.layers.add()
        data_layer.layer.name = "data"
        data_layer.layer.type = "data"
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
        caffe_conv_layer.layer.type = "conv"
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
        caffe_relu_layer.layer.type = "relu"
        caffe_relu_layer.bottom.append(prev_layer_name)
        caffe_relu_layer.top.append(current_layer_name)

        prev_layer_name = current_layer_name

        # Pooling
        caffe_pool_layer = self._caffe_net.layers.add()

        current_layer_name = current_layer_base_name + "pool"
        caffe_pool_layer.layer.name = current_layer_name
        caffe_pool_layer.layer.type = "pool"
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
        caffe_fc_layer.layer.type = "innerproduct"
        caffe_fc_layer.bottom.append(prev_layer_name)
        caffe_fc_layer.top.append(current_layer_name)

        weight_filler_params = params.pop("weight-filler")
        weight_filler = caffe_fc_layer.layer.weight_filler
        for param, param_val in weight_filler_params.iteritems():
            setattr(weight_filler, param, param_val)

        caffe_fc_layer.layer.bias_filler.type = "constant"
        caffe_fc_layer.layer.bias_filler.value = 0

        prev_layer_name = current_layer_name

        #TODO: add RELU
        #TODO: add dropout

        return prev_layer_name

    def _create_network_parameters(self, params):
        self._solver.base_lr = params.pop("lr")
        self._solver.momentum = params.pop("momentum")
        self._solver.weight_decay = params.pop("weight_decay")
        self._solver.train_net = self._train_network_file
        self._solver.test_net = self._valid_network_file
        self._solver.test_iter = self._num_validation_set_batches
        #TODO: make parameter
        self._solver.test_interval = 100
        self._solver.display = 100
        self._solver.snapshot = 10000000
        self._solver.snapshot_prefix = "caffenet"

        assert len(params) == 0, "More solver parameters given than needed: " + str(params)

    def _serialize(self):
        """
            Serialize the network to protobuf files.
        """
        with open(self._solver_file, "w") as solver_file:
            solver_file.write(str(self._solver))

        with open(self._train_network_file, "w") as train_network:
            train_network.write(str(self._caffe_net_train))

        with open(self._valid_network_file, "w") as valid_network:
            valid_network.write(str(self._caffe_net_validation))

    def run(self):
        """
            Run the given network and return the best validation performance.
        """
        return run_caffe()


def run_caffe():
    """
        Runs caffe and returns the accuracy.
    """
    try:
        output = check_output(("GLOG_logtostderr=1 train_net.bin "
                               "imagenet_solver.prototxt"),
                               shell=True)
                               #, stderr=STDOUT)

        accuracy = output.split("\n")[-1]
        if "Accuracy:" in accuracy:
            accuracy = float(accuracy.split("Accuracy: ", 1)[1])
            print "accuracy: ", accuracy
            return {'loss': 1.0-accuracy, 'status': "ok"}
        else:
            #Failed?
            return {'loss': 1.0, 'status': "fail"}
    except:
        print "ERROR!"
        return {'loss': 1.0, 'status': "fail"}

