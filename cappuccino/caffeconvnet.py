from caffe.proto import caffe_pb2
from subprocess import check_output, STDOUT, CalledProcessError
import traceback
import copy

class TerminationCriterion(object):
    def __init__(self):
        pass

    def add_to_solver_param(self, solver, iter_per_epoch):
        raise NotImplementedError("this is just a base class..")


class TerminationCriterionMaxIter(TerminationCriterion):
    def __init__(self, max_epochs):
        self.max_epochs = max_epochs

    def add_to_solver_param(self, solver, iter_per_epoch):
        solver.termination_criterion = self.caffe_pb2.SolverParameter.MAX_ITER
        solver.max_iter = iter_per_epoch * self.max_epochs

class TerminationCriterionTestAccuracy(TerminationCriterion):
    def __init__(self, test_accuracy_stop_countdown):
        """
            test_accuracy_stop_countdown: countdown in epochs
        """
        self.test_accuracy_stop_countdown = test_accuracy_stop_countdown

    def add_to_solver_param(self, solver, iter_per_epoch):
        solver.termination_criterion = caffe_pb2.SolverParameter.TEST_ACCURACY
        #stop, when no improvement for X epoches
        solver.test_accuracy_stop_countdown = self.test_accuracy_stop_countdown * 10


class CaffeConvNet(object):
    """
        Runs a caffe convnet with the given parameters
    """
    def __init__(self, params,
                 train_file,
                 valid_file,
                 test_file,
                 num_train,
                 num_valid,
                 num_test,
                 mean_file=None,
                 scale_factor=None,
                 batch_size_train = 128,
                 batch_size_valid = 100,
                 batch_size_test = 100,
                 termination_criterion = TerminationCriterionTestAccuracy(5),
                 device = "GPU",
                 device_id = 0,
                 snapshot_on_exit = 0):
        """
            Parameters of the network as defined by ConvNetSearchSpace.

            params: parameters to the network
            train_file: the training data hdf5 file
            valid_file: the validation data hdf5 file
            test_file: the test data hdf5 file
            num_train: number of examples in the train set
            num_valid: number of examples in the validation set
            num_test: number of examples in the test set
            mean_file: DEPRECATED FOR NOW mean per dimension leveldb file
            scale_factor: a factor used for scaling the input data.
            batch_size_train: the batch size during training
            batch_size_valid: the batch size during validation
            batch_size_test: the batch size during testing
            termination_criterion: either "accuracy" or "max_iter"
            device: either "CPU" or "GPU"
            device_id: the id of the device to run the experiment on
            snapshot_on_exit: save network on exit?
        """
        self._train_file = train_file
        self._valid_file = valid_file
        self._test_file = test_file
        self._mean_file = mean_file
        if scale_factor != None:
            self._scale_factor = scale_factor
        assert num_train % batch_size_train == 0, "num_train must be a multiple of the train bach size"
        assert num_valid % batch_size_valid == 0, "num_valid must be a multiple of the valid bach size"
        assert num_test % batch_size_test == 0, "num_test must be a multiple of the test bach size"
        self._batch_size_train = batch_size_train
        self._batch_size_valid = batch_size_valid
        self._batch_size_test = batch_size_valid
        self._num_train = num_train
        self._num_valid = num_valid
        self._num_test = num_valid
        assert isinstance(termination_criterion, TerminationCriterion)
        self._termination_criterion = termination_criterion
        assert device in ["CPU", "GPU"]
        self._device = device
        self._device_id = device_id
        self._snapshot_on_exit = snapshot_on_exit

        self._base_name = "caffenet"

        self._train_network_file = self._base_name + "_train.prototxt"
        self._valid_network_file = self._base_name + "_valid.prototxt"
        self._test_network_file = self._base_name + "_test.prototxt"
        self._solver_file = self._base_name + "_solver.prototxt"

        self._convert_params_to_caffe_network(copy.deepcopy(params))
        self._create_train_valid_networks()


    def _convert_params_to_caffe_network(self, params):
        """
            Converts the given parameters into a caffe network configuration.
        """
        self._caffe_net = caffe_pb2.NetParameter()
        self._solver = caffe_pb2.SolverParameter()

        preproc_params, all_conv_layers_params, all_fc_layers_params, network_params = params

        prev_layer_name = self._create_data_layer(preproc_params)

        assert len(all_conv_layers_params) == network_params['num_conv_layers']
        assert len(all_fc_layers_params) == network_params['num_fc_layers']

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
        self._caffe_net_train.name = "train"
        self._caffe_net_train.layers[0].hdf5_data_param.source = self._train_file
        self._caffe_net_train.layers[0].hdf5_data_param.batch_size = self._batch_size_train
        #if self._mean_file:
        #    self._caffe_net_train.layers[0].layer.meanfile = self._mean_file

        # add train loss:
        last_layer_top = self._caffe_net_train.layers[-1].top[0]
        loss_layer = self._caffe_net_train.layers.add()
        loss_layer.name = "loss"
        loss_layer.type = caffe_pb2.LayerParameter.SOFTMAX_LOSS
        loss_layer.bottom.append(last_layer_top)
        loss_layer.bottom.append("label")

        # validation network:
        self._caffe_net_validation = copy.deepcopy(self._caffe_net)
        self._add_softmax_accuray_layers(self._caffe_net_validation)
        self._caffe_net_validation.name = "valid"
        self._caffe_net_validation.layers[0].hdf5_data_param.source = self._valid_file
        self._caffe_net_validation.layers[0].hdf5_data_param.batch_size = self._batch_size_valid
        #if self._mean_file:
        #    self._caffe_net_validation.layers[0].layer.meanfile = self._mean_file

        self._caffe_net_test = copy.deepcopy(self._caffe_net)
        self._add_softmax_accuray_layers(self._caffe_net_test)
        self._caffe_net_test.name = "test"
        self._caffe_net_test.layers[0].hdf5_data_param.source = self._test_file
        self._caffe_net_test.layers[0].hdf5_data_param.batch_size = self._batch_size_test
 

    def _add_softmax_accuray_layers(self, caffe_net):
        """add a softmax and an accuracy layer to the net."""
        #softmax layer:
        last_layer_top = caffe_net.layers[-1].top[0]
        prob_layer = caffe_net.layers.add()
        prob_layer.name = "prob"
        prob_layer.type = caffe_pb2.LayerParameter.SOFTMAX
        prob_layer.bottom.append(last_layer_top)
        prob_layer.top.append("prob")

        #accuracy layer:
        prob_layer = caffe_net.layers.add()
        prob_layer.name = "accuracy"
        prob_layer.type = caffe_pb2.LayerParameter.ACCURACY
        prob_layer.bottom.append("prob")
        prob_layer.bottom.append("label")
        prob_layer.top.append("accuracy")


    def _create_data_layer(self, params):
        data_layer = self._caffe_net.layers.add()
        data_layer.name = "data"
        data_layer.type = caffe_pb2.LayerParameter.HDF5_DATA
        if hasattr(self, "_scale_factor"):
            data_layer.layer.scale = self._scale_factor
        data_layer.top.append("data")
        data_layer.top.append("label")
        prev_layer_name = "data"

        augment_params = params["augment"]
        if augment_params["type"] == "augment":
            augmentation_layer = self._caffe_net.layers.add()
            augmentation_layer.type = caffe_pb2.LayerParameter.DATA_AUGMENTATION
            augmentation_layer.data_param.mirror = True
            augmentation_layer.data_param.crop_size = int(augment_params["crop_size"])
            augmentation_layer.bottom.append("data")
            prev_layer_name = "augmented_data"
            augmentation_layer.top.append(prev_layer_name)

        return prev_layer_name



    def _create_conv_layer(self, current_layer_base_name, prev_layer_name, params):
        """
            Generate a caffe layer from the given parameters.

            Note: one logical layer will be converted to multiple caffe layers.
        """

        #Convolution
        assert params.pop("type") == "conv"

        caffe_conv_layer = self._caffe_net.layers.add()
        current_layer_name = current_layer_base_name + "conv"
        caffe_conv_layer.name = current_layer_name
        caffe_conv_layer.type = caffe_pb2.LayerParameter.CONVOLUTION
        caffe_conv_layer.convolution_param.kernel_size = int(params.pop("kernelsize")) 
        caffe_conv_layer.convolution_param.num_output = int(params.pop("num_output_x_128")) * 128
        caffe_conv_layer.convolution_param.stride = int(params.pop("stride"))
        weight_filler_params = params.pop("weight-filler")
        weight_filler = caffe_conv_layer.convolution_param.weight_filler
        for param, param_val in weight_filler_params.iteritems():
            setattr(weight_filler, param, param_val)

        caffe_conv_layer.convolution_param.bias_filler.type = "constant"
        bias_filler_params = params.pop("bias-filler")
        if bias_filler_params["type"] == "const-zero":
            caffe_conv_layer.convolution_param.bias_filler.value = 0.
        elif bias_filler_params["type"] == "const-one":
            caffe_conv_layer.convolution_param.bias_filler.value = 1.
        else:
            raise RuntimeError("unknown bias-filler %s" % (bias_filler_params["type"]))

#        bias_filler_params = params.pop("bias-filler")
#        bias_filler = caffe_conv_layer.layer.bias_filler
#        for param, param_val in bias_filler_params.iteritems():
#            setattr(bias_filler, param, param_val)
        caffe_conv_layer.blobs_lr.append(params.pop("weight-lr-multiplier"))
        caffe_conv_layer.blobs_lr.append(params.pop("bias-lr-multiplier"))

        caffe_conv_layer.weight_decay.append(params.pop("weight-weight-decay_multiplier"))
        caffe_conv_layer.weight_decay.append(params.pop("bias-weight-decay_multiplier"))

        padding_params = params.pop("padding")
        if padding_params["type"] == "zero-padding":
            caffe_conv_layer.convolution_param.pad = int(padding_params["size"])

        caffe_conv_layer.bottom.append(prev_layer_name)
        caffe_conv_layer.top.append(current_layer_name)

        prev_layer_name = current_layer_name

        # Relu
        caffe_relu_layer = self._caffe_net.layers.add()

        current_layer_name = current_layer_base_name + "relu"
        caffe_relu_layer.name = current_layer_name
        caffe_relu_layer.type = caffe_pb2.LayerParameter.RELU
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
            caffe_pool_layer.name = current_layer_name
            caffe_pool_layer.type = caffe_pb2.LayerParameter.POOLING
            caffe_pool_layer.bottom.append(prev_layer_name)
            caffe_pool_layer.top.append(current_layer_name)
            caffe_pool_layer.pooling_param.pool = caffe_pb2.PoolingParameter.MAX
            caffe_pool_layer.pooling_param.kernel_size = int(pooling_params["kernelsize"])
            caffe_pool_layer.pooling_param.stride = int(pooling_params["stride"])

            prev_layer_name = current_layer_name
        elif pooling_params["type"] == "ave":
            caffe_pool_layer = self._caffe_net.layers.add()

            current_layer_name = current_layer_base_name + "pool"
            caffe_pool_layer.name = current_layer_name
            caffe_pool_layer.type = caffe_pb2.LayerParameter.POOLING
            caffe_pool_layer.bottom.append(prev_layer_name)
            caffe_pool_layer.top.append(current_layer_name)
            caffe_pool_layer.pooling_param.pool = caffe_pb2.PoolingParameter.AVE
            caffe_pool_layer.pooling_param.kernel_size = int(pooling_params["kernelsize"])
            caffe_pool_layer.pooling_param.stride = int(pooling_params["stride"])

            prev_layer_name = current_layer_name
        #TODO: add stochastic pooling

        normalization_params = params.pop("norm")
        if normalization_params["type"] == "lrn":
            caffe_norm_layer = self._caffe_net.layers.add()

            #TODO: add across and within channel pooling!

            current_layer_name = current_layer_base_name + "norm"
            caffe_norm_layer.name = current_layer_name
            caffe_norm_layer.type = caffe_pb2.LayerParameter.LRN
            caffe_norm_layer.bottom.append(prev_layer_name)
            caffe_norm_layer.top.append(current_layer_name)
            caffe_norm_layer.lrn_param.local_size = int(normalization_params["local_size"])
            caffe_norm_layer.lrn_param.alpha = int(normalization_params["alpha"])
            caffe_norm_layer.lrn_param.beta = int(normalization_params["beta"])

            prev_layer_name = current_layer_name


        #Dropout
        dropout_params = params.pop("dropout")
        if dropout_params["type"] == "dropout":
            caffe_dropout_layer = self._caffe_net.layers.add()

            current_layer_name = current_layer_base_name + "dropout"
            caffe_dropout_layer.name = current_layer_name
            caffe_dropout_layer.type = caffe_pb2.LayerParameter.DROPOUT
            caffe_dropout_layer.dropout_param.dropout_ratio = dropout_params.pop("dropout_ratio")
            caffe_dropout_layer.bottom.append(prev_layer_name)
            caffe_dropout_layer.top.append(current_layer_name)

            prev_layer_name = current_layer_name
 
        assert len(params) == 0, "More convolution parameters given than needed: " + str(params)

        return prev_layer_name

    def _create_fc_layer(self, current_layer_base_name, prev_layer_name, params):
        caffe_fc_layer = self._caffe_net.layers.add()

        assert params.pop("type") == "fc"

        current_layer_name = current_layer_base_name + "fc"
        caffe_fc_layer.name = current_layer_name
        caffe_fc_layer.type = caffe_pb2.LayerParameter.INNER_PRODUCT
        caffe_fc_layer.bottom.append(prev_layer_name)
        caffe_fc_layer.top.append(current_layer_name)

        if "num_output_x_128" in params:
            caffe_fc_layer.inner_product_param.num_output = int(params.pop("num_output_x_128")) * 128
        elif "num_output" in params:
            caffe_fc_layer.inner_product_param.num_output = int(params.pop("num_output")) 

        caffe_fc_layer.blobs_lr.append(params.pop("weight-lr-multiplier"))
        caffe_fc_layer.blobs_lr.append(params.pop("bias-lr-multiplier"))

        caffe_fc_layer.weight_decay.append(1)
        caffe_fc_layer.weight_decay.append(0)

        weight_filler_params = params.pop("weight-filler")
        weight_filler = caffe_fc_layer.inner_product_param.weight_filler
        for param, param_val in weight_filler_params.iteritems():
            setattr(weight_filler, param, param_val)

        caffe_fc_layer.inner_product_param.bias_filler.type = "constant"
        bias_filler_params = params.pop("bias-filler")
        if bias_filler_params["type"] == "const-zero":
            caffe_fc_layer.inner_product_param.bias_filler.value = 0.
        elif bias_filler_params["type"] == "const-one":
            caffe_fc_layer.inner_product_param.bias_filler.value = 1.
        else:
            raise RuntimeError("unknown bias-filler %s" % (bias_filler_params["type"]))


        prev_layer_name = current_layer_name

        #RELU
        if params.pop("activation") == "relu":
            caffe_relu_layer = self._caffe_net.layers.add()

            current_layer_name = current_layer_base_name + "relu"
            caffe_relu_layer.name = current_layer_name
            caffe_relu_layer.type = caffe_pb2.LayerParameter.RELU
            caffe_relu_layer.bottom.append(prev_layer_name)
            #Note: the operation is made in-place by using the same name twice
            caffe_relu_layer.top.append(prev_layer_name)


        #Dropout
        dropout_params = params.pop("dropout")
        if dropout_params["type"] == "dropout":
            caffe_dropout_layer = self._caffe_net.layers.add()

            current_layer_name = current_layer_base_name + "dropout"
            caffe_dropout_layer.name = current_layer_name
            caffe_dropout_layer.type = caffe_pb2.LayerParameter.DROPOUT
            caffe_dropout_layer.dropout_param.dropout_ratio = dropout_params.pop("dropout_ratio")
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
        self._solver.test_net.append(self._valid_network_file)
        self._solver.test_iter.append(int(self._num_valid / self._batch_size_valid))
        self._solver.test_net.append(self._test_network_file)
        self._solver.test_iter.append(int(self._num_test / self._batch_size_test))
        #TODO: add both the validation aaaand the test set

        self._termination_criterion.add_to_solver_param(self._solver, self._num_train / self._batch_size_train)
            
        #test 10 times per epoch:
        self._solver.test_interval = int((0.1 * self._num_train) / self._batch_size_train)
        self._solver.display = int((0.01 * self._num_train) / self._batch_size_train)
        self._solver.snapshot = 0
        self._solver.snapshot_prefix = "caffenet"
        if self._device == "CPU":
            self._solver.solver_mode = caffe_pb2.SolverParameter.CPU
        elif self._device == "GPU":
            self._solver.solver_mode = caffe_pb2.SolverParameter.GPU
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

        with open(self._test_network_file, "w") as test_network:
            test_network.write(str(self._caffe_net_test))

    def run(self):
        """
            Run the given network and return the logging output.
        """
        self._serialize()
        return run_caffe(self._solver_file)


def run_caffe(solver_file, hide_output=True):
    """
        Runs caffe and returns the logging output.
    """
    try:
        os.environ["GLOG_logtostderr"] = 1
        output = check_output(["train_net.sh", solver_file], stderr=STDOUT)
        return output
#        if hide_output:
#            output = check_output(["train_net.sh", solver_file], stderr=STDOUT)
#        else:
#            output = check_output(["train_net.sh", solver_file])
#
#        accuracy = output.split("\n")[-1]
#        if "Accuracy:" in accuracy:
#            accuracy = float(accuracy.split("Accuracy: ", 1)[1])
#            return 1.0 - accuracy
#        else:
#            #Failed?
#            raise RuntimeError("Failed running caffe: didn't find accuracy in output")
    except CalledProcessError as e:
        raise RuntimeError("Failed running caffe. Return code: %s Output: %s" % (str(e.returncode), str(e.output)))
    except:
        print "UNKNOWN ERROR!"
        raise RuntimeError("Unknown exception, when running caffe." + traceback.format_exc())

