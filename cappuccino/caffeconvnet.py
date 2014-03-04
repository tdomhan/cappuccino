from caffe.proto import caffe_pb2
from subprocess import check_output, call, STDOUT, CalledProcessError
import traceback
import copy

class CaffeConvNet(object):
    """
        Runs a caffe convnet with the given parameters
    """
    def __init__(self, params,
                 train_file,
                 valid_file,
                 num_train,
                 num_valid,
                 mean_file=None,
                 scale_factor=None,
                 batch_size_train = 128,
                 batch_size_valid = 100,
                 device = "GPU",
                 device_id = 0,
                 snapshot_on_exit = 0):
        """
            Parameters of the network as defined by ConvNetSearchSpace.

            params: parameters to the network
            train_file: the training data leveldb file
            valid_file: the validation data leveldb file
            num_train: number of examples in the train set
            num_valid: number of examples in the validation set
            mean_file: mean per dimesnion leveldb file
            scale_factor: a factor used for scaling the input data.
            batch_size_train: the batch size during training
            batch_size_test: the batch size during testing
            device: either "CPU" or "GPU"
            device_id: the id of the device to run the experiment on
            snapshot_on_exit: save network on exit?
        """
        self._train_file = train_file
        self._valid_file = valid_file
        self._mean_file = mean_file
        if scale_factor != None:
            self._scale_factor = scale_factor
        assert num_train % batch_size_train == 0, "num_train must be a multiple of the train bach size"
        assert num_valid % batch_size_valid == 0, "num_valid must be a multiple of the valid bach size"
        self._batch_size_train = batch_size_train
        self._batch_size_valid = batch_size_valid
        self._num_train = num_train
        self._num_valid = num_valid
        assert device in ["CPU", "GPU"]
        self._device = device
        self._device_id = device_id
        self._snapshot_on_exit = snapshot_on_exit

        self._base_name = "caffenet"

        self._train_network_file = self._base_name + "_train.prototxt"
        self._valid_network_file = self._base_name + "_valid.prototxt"
        self._solver_file = self._base_name + "_solver.prototxt"

        self._convert_params_to_caffe_network(copy.deepcopy(params))
        self._create_train_valid_networks()

        self._serialize()

    def _convert_params_to_caffe_network(self, params):
        """
            Converts the given parameters into a caffe network configuration.
        """
        self._caffe_net = caffe_pb2.NetParameter()
        self._solver = caffe_pb2.SolverParameter()

        preproc_params, all_conv_layers_params, all_fc_layers_params, network_params = params

        self._create_data_layer(preproc_params)

        assert len(all_conv_layers_params) == network_params['num_conv_layers']
        assert len(all_fc_layers_params) == network_params['num_fc_layers']

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
        #turn off mirroring:
        self._caffe_net_validation.layers[0].layer.mirror = False

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


    def _create_data_layer(self, params):
        data_layer = self._caffe_net.layers.add()
        data_layer.layer.name = "data"
        data_layer.layer.type = "data"
        if hasattr(self, "_scale_factor"):
            data_layer.layer.scale = self._scale_factor
        data_layer.top.append("data")
        data_layer.top.append("label")

        augment_params = params["augment"]
        if augment_params["type"] == "augment":
            #note: will be turned off for the validation layer
            data_layer.layer.mirror = True

            #note: will be turned off for the validation layer
            data_layer.layer.cropsize = int(augment_params["crop_size"])

    def _create_conv_layer(self, current_layer_base_name, prev_layer_name, params):
        """
            Generate a caffe layer from the given parameters.

            Note: one logical layer will be converted to multiple caffe layers.
        """

        #Convolution
        assert params.pop("type") == "conv"

        padding_params = params.pop("padding")
        if padding_params["type"] == "zero-padding":
            caffe_pad_layer = self._caffe_net.layers.add()

            current_layer_name = current_layer_base_name + "pad"
            caffe_pad_layer.layer.name = current_layer_name
            caffe_pad_layer.layer.type = "padding"
            caffe_pad_layer.layer.pad = int(padding_params["size"])
            caffe_pad_layer.bottom.append(prev_layer_name)
            #Note: the operation is made in-place by using the same name twice
            caffe_pad_layer.top.append(current_layer_name)

            prev_layer_name = current_layer_name


        caffe_conv_layer = self._caffe_net.layers.add()
        current_layer_name = current_layer_base_name + "conv"
        caffe_conv_layer.layer.name = current_layer_name
        caffe_conv_layer.layer.type = "conv"
        caffe_conv_layer.layer.kernelsize = int(params.pop("kernelsize")) 
        caffe_conv_layer.layer.num_output = int(params.pop("num_output_x_128")) * 128
        caffe_conv_layer.layer.stride = int(params.pop("stride"))
        weight_filler_params = params.pop("weight-filler")
        weight_filler = caffe_conv_layer.layer.weight_filler
        for param, param_val in weight_filler_params.iteritems():
            setattr(weight_filler, param, param_val)

        caffe_conv_layer.layer.bias_filler.type = "constant"
        bias_filler_params = params.pop("bias-filler")
        if bias_filler_params["type"] == "const-zero":
            caffe_conv_layer.layer.bias_filler.value = 0.
        elif bias_filler_params["type"] == "const-one":
            caffe_conv_layer.layer.bias_filler.value = 1.
        else:
            raise RuntimeError("unknown bias-filler %s" % (bias_filler_params["type"]))

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
        #Note: the operation is made in-place by using the same name twice
        caffe_relu_layer.top.append(prev_layer_name)

        # Pooling
        pooling_params = params.pop("pooling")
        if pooling_params["type"] == "none":
            pass
        elif pooling_params["type"] == "max":
            caffe_pool_layer = self._caffe_net.layers.add()

            current_layer_name = current_layer_base_name + "pool"
            caffe_pool_layer.layer.name = current_layer_name
            caffe_pool_layer.layer.type = "pool"
            caffe_pool_layer.bottom.append(prev_layer_name)
            caffe_pool_layer.top.append(current_layer_name)
            caffe_pool_layer.layer.pool = caffe_pool_layer.layer.MAX
            caffe_pool_layer.layer.kernelsize = int(pooling_params["kernelsize"])
            caffe_pool_layer.layer.stride = int(pooling_params["stride"])

            prev_layer_name = current_layer_name
        elif pooling_params["type"] == "ave":
            caffe_pool_layer = self._caffe_net.layers.add()

            current_layer_name = current_layer_base_name + "pool"
            caffe_pool_layer.layer.name = current_layer_name
            caffe_pool_layer.layer.type = "pool"
            caffe_pool_layer.bottom.append(prev_layer_name)
            caffe_pool_layer.top.append(current_layer_name)
            caffe_pool_layer.layer.pool = caffe_pool_layer.layer.AVE
            caffe_pool_layer.layer.kernelsize = int(pooling_params["kernelsize"])
            caffe_pool_layer.layer.stride = int(pooling_params["stride"])

            prev_layer_name = current_layer_name

        normalization_params = params.pop("norm")
        if normalization_params["type"] == "lrn":
            caffe_norm_layer = self._caffe_net.layers.add()

            current_layer_name = current_layer_base_name + "norm"
            caffe_norm_layer.layer.name = current_layer_name
            caffe_norm_layer.layer.type = "lrn"
            caffe_norm_layer.bottom.append(prev_layer_name)
            caffe_norm_layer.top.append(current_layer_name)
            caffe_norm_layer.layer.local_size = int(normalization_params["local_size"])
            caffe_norm_layer.layer.alpha = int(normalization_params["alpha"])
            caffe_norm_layer.layer.beta = int(normalization_params["beta"])

            prev_layer_name = current_layer_name


        #Dropout
        dropout_params = params.pop("dropout")
        if dropout_params["type"] == "dropout":
            caffe_dropout_layer = self._caffe_net.layers.add()

            current_layer_name = current_layer_base_name + "dropout"
            caffe_dropout_layer.layer.name = current_layer_name
            caffe_dropout_layer.layer.type = "dropout"
            caffe_dropout_layer.layer.dropout_ratio = dropout_params.pop("dropout_ratio")
            caffe_dropout_layer.bottom.append(prev_layer_name)
            caffe_dropout_layer.top.append(current_layer_name)

            prev_layer_name = current_layer_name
 
        assert len(params) == 0, "More convolution parameters given than needed: " + str(params)

        return prev_layer_name

    def _create_fc_layer(self, current_layer_base_name, prev_layer_name, params):
        caffe_fc_layer = self._caffe_net.layers.add()

        assert params.pop("type") == "fc"

        current_layer_name = current_layer_base_name + "fc"
        caffe_fc_layer.layer.name = current_layer_name
        caffe_fc_layer.layer.type = "innerproduct"
        caffe_fc_layer.bottom.append(prev_layer_name)
        caffe_fc_layer.top.append(current_layer_name)

        if "num_output_x_128" in params:
            caffe_fc_layer.layer.num_output = int(params.pop("num_output_x_128")) * 128
        elif "num_output" in params:
            caffe_fc_layer.layer.num_output = int(params.pop("num_output")) 

        caffe_fc_layer.layer.blobs_lr.append(params.pop("weight-lr-multiplier"))
        caffe_fc_layer.layer.blobs_lr.append(params.pop("bias-lr-multiplier"))

        caffe_fc_layer.layer.weight_decay.append(1)
        caffe_fc_layer.layer.weight_decay.append(0)

        weight_filler_params = params.pop("weight-filler")
        weight_filler = caffe_fc_layer.layer.weight_filler
        for param, param_val in weight_filler_params.iteritems():
            setattr(weight_filler, param, param_val)

        caffe_fc_layer.layer.bias_filler.type = "constant"
        bias_filler_params = params.pop("bias-filler")
        if bias_filler_params["type"] == "const-zero":
            caffe_fc_layer.layer.bias_filler.value = 0.
        elif bias_filler_params["type"] == "const-one":
            caffe_fc_layer.layer.bias_filler.value = 1.
        else:
            raise RuntimeError("unknown bias-filler %s" % (bias_filler_params["type"]))


        prev_layer_name = current_layer_name

        #RELU
        if params.pop("activation") == "relu":
            caffe_relu_layer = self._caffe_net.layers.add()

            current_layer_name = current_layer_base_name + "relu"
            caffe_relu_layer.layer.name = current_layer_name
            caffe_relu_layer.layer.type = "relu"
            caffe_relu_layer.bottom.append(prev_layer_name)
            #Note: the operation is made in-place by using the same name twice
            caffe_relu_layer.top.append(prev_layer_name)


        #Dropout
        dropout_params = params.pop("dropout")
        if dropout_params["type"] == "dropout":
            caffe_dropout_layer = self._caffe_net.layers.add()

            current_layer_name = current_layer_base_name + "dropout"
            caffe_dropout_layer.layer.name = current_layer_name
            caffe_dropout_layer.layer.type = "dropout"
            caffe_dropout_layer.layer.dropout_ratio = dropout_params.pop("dropout_ratio")
            caffe_dropout_layer.bottom.append(prev_layer_name)
            caffe_dropout_layer.top.append(current_layer_name)

            prev_layer_name = current_layer_name

        assert len(params) == 0, "More fc layer parameters given than needed: " + str(params)

        return prev_layer_name

    def _create_network_parameters(self, params):
        params.pop("num_conv_layers")
        params.pop("num_fc_layers")
        self._solver.base_lr = params.pop("lr")
        lr_policy_params = params.pop("lr_policy")
        lr_policy = lr_policy_params.pop("type")
        self._solver.lr_policy = lr_policy
        if lr_policy == "fixed":
            pass
        elif lr_policy == "exp":
            self._solver.gamma = lr_policy_params.pop("gamma")
        elif lr_policy == "step":
            self._solver.gamma = lr_policy_params.pop("gamma")
            if "stepsize" in lr_policy_params:
                self._solver.stepsize = int(lr_policy_params.pop("stepsize"))
            elif "epochcount" in lr_policy_params:
                self._solver.stepsize = int((self._num_train / self._batch_size_train) * lr_policy_params.pop("epochcount"))
            else:
                assert False, "Neither stepsize nor epochcount given."
        elif lr_policy == "inv":
            self._solver.gamma = lr_policy_params.pop("gamma")
            self._solver.power = lr_policy_params.pop("power")
        assert len(lr_policy_params) == 0, "More learning policy arguments given, than needed, " + str(lr_policy_params)

        self._solver.momentum = params.pop("momentum")
        self._solver.weight_decay = params.pop("weight_decay")
        self._solver.train_net = self._train_network_file
        self._solver.test_net = self._valid_network_file
        self._solver.test_iter = int(self._num_valid / self._batch_size_valid)

        self._solver.termination_criterion = self._solver.TEST_ACCURACY
        #stop, when no improvement for X epoches
        self._solver.test_accuracy_stop_countdown = 3 * 10

        #test 10 times per epoch:
        self._solver.test_interval = int((0.1 * self._num_train) / self._batch_size_train)
        self._solver.display = int((0.01 * self._num_train) / self._batch_size_train)
        self._solver.snapshot = 10000000
        self._solver.snapshot_prefix = "caffenet"
        if self._device == "CPU":
            self._solver.solver_mode = 0
        elif self._device == "GPU":
            self._solver.solver_mode = 1
            self._solver.device_id = self._device_id

        self._solver.snapshot_on_exit = self._snapshot_on_exit

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

    def run(self, hide_output=True):
        """
            Run the given network and return the best validation performance.
        """
        return run_caffe(self._solver_file, hide_output=hide_output)


def run_caffe(solver_file, hide_output=True):
    """
        Runs caffe and returns the accuracy.
    """
    try:
        if hide_output:
            output = check_output(["train_net.sh", solver_file], stderr=STDOUT)
        else:
            output = check_output(["train_net.sh", solver_file])

        accuracy = output.split("\n")[-1]
        if "Accuracy:" in accuracy:
            accuracy = float(accuracy.split("Accuracy: ", 1)[1])
            return 1.0 - accuracy
        else:
            #Failed?
            raise RuntimeError("Failed running caffe: didn't find accuracy in output")
    except CalledProcessError as e:
        raise RuntimeError("Failed running caffe. Return code: %s Output: %s" % (str(e.returncode), str(e.output)))
    except:
        print "UNKNOWN ERROR!"
        raise RuntimeError("Unknown exception, when running caffe." + traceback.format_exc())

