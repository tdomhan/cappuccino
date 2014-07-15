from caffe.proto import caffe_pb2
from subprocess import check_output, STDOUT, CalledProcessError
import numpy as np
import math
import traceback
import copy
import os


class TerminationCriterion(object):
    def __init__(self):
        pass

    def add_to_solver_param(self, solver, iter_per_epoch, tests_per_epoch):
        raise NotImplementedError("this is just a base class..")


class TerminationCriterionMaxEpoch(TerminationCriterion):
    def __init__(self, max_epochs):
        self.max_epochs = max_epochs

    def add_to_solver_param(self, solver, iter_per_epoch, tests_per_epoch):
        solver.termination_criterion.append(caffe_pb2.SolverParameter.MAX_ITER)
        solver.max_iter = iter_per_epoch * self.max_epochs


class TerminationCriterionTestAccuracy(TerminationCriterion):
    def __init__(self, test_accuracy_stop_countdown):
        """
            test_accuracy_stop_countdown: countdown in epochs
        """
        self.test_accuracy_stop_countdown = test_accuracy_stop_countdown

    def add_to_solver_param(self, solver, iter_per_epoch, tests_per_epoch):
        solver.termination_criterion.append(caffe_pb2.SolverParameter.TEST_ACCURACY)
        #stop, when no improvement for X epoches
        solver.test_accuracy_stop_countdown = self.test_accuracy_stop_countdown * tests_per_epoch

class TerminationCriterionDivergenceDetection(TerminationCriterion):
    def __init__(self):
        """
            test_accuracy_stop_countdown: countdown in epochs
        """
        pass

    def add_to_solver_param(self, solver, iter_per_epoch, tests_per_epoch):
        solver.termination_criterion.append(caffe_pb2.SolverParameter.DIVERGENCE_DETECTION)

class TerminationCriterionExternal(TerminationCriterion):

    def __init__(self, external_cmd, run_every_x_epochs):
        self.external_cmd = external_cmd
        self.run_every_x_epochs = run_every_x_epochs

    def add_to_solver_param(self, solver, iter_per_epoch, tests_per_epoch):
        solver.termination_criterion.append(caffe_pb2.SolverParameter.EXTERNAL)
        solver.external_term_criterion_cmd = self.external_cmd
        solver.external_term_criterion_num_iter = self.run_every_x_epochs * iter_per_epoch


class TerminationCriterionExternalInBackground(TerminationCriterion):

    def __init__(self, external_cmd, run_every_x_epochs):
        self.external_cmd = external_cmd
        self.run_every_x_epochs = run_every_x_epochs

    def add_to_solver_param(self, solver, iter_per_epoch, tests_per_epoch):
        solver.termination_criterion.append(caffe_pb2.SolverParameter.EXTERNAL_IN_BACKGROUND)
        solver.external_term_criterion_cmd = self.external_cmd
        solver.external_term_criterion_num_iter = self.run_every_x_epochs * iter_per_epoch

class CaffeConvNet(object):
    """
        Runs a caffe convnet with the given parameters
    """
    def __init__(self, params,
                 train_file,
                 num_train,
                 valid_file,
                 num_valid,
                 test_file=None,
                 num_test=None,
                 mean_file=None,
                 scale_factor=None,
                 batch_size_valid = 100,
                 batch_size_test = 100,
                 test_every_x_epoch = 0.1,
                 termination_criterions = [TerminationCriterionTestAccuracy(5)],
                 input_image_size=None,
                 min_image_size=None,
	         snapshot_prefix="caffenet",
                 restrict_to_legal_configurations=False,
                 device = "GPU",
                 device_id = 0,
                 seed=13,
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
            test_every_x_epoch: epoch count or fraction thereof after which the test network is run.
            termination_criterions: either "accuracy" or "max_iter"
            min_image_size: needs to be set if restrict_to_legal_configurations is True
            restrict_to_legal_configurations: The input configuration will be corrected to form a legal configuration.
                that is, if the size of the input shrinks too much due to pooling
                operations, some pooling layers will be left out.
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
        assert num_valid % batch_size_valid == 0, "num_valid must be a multiple of the valid bach size"
        if num_test is not None or test_file is not None:
            assert num_test % batch_size_test == 0, "num_test must be a multiple of the test bach size"
        self._test_every_x_epoch = test_every_x_epoch
        #batch_size_train is a network parameter and set during network configuration
        self._batch_size_valid = batch_size_valid
        self._batch_size_test = batch_size_valid
        self._num_train = num_train
        self._num_valid = num_valid
        self._num_test = num_test
        for termination_criterion in termination_criterions:
            assert isinstance(termination_criterion, TerminationCriterion)
        self._termination_criterions = termination_criterions
        self._restrict_to_legal_configurations = restrict_to_legal_configurations
	self._snapshot_prefix = snapshot_prefix
        if restrict_to_legal_configurations:
            assert min_image_size is not None, "min_image_size needs to be set."
            assert input_image_size is not None, "min_image_size needs to be set."
        self._min_image_size = min_image_size
        self._input_image_size = input_image_size

        assert device in ["CPU", "GPU"]
        self._device = device
        self._device_id = device_id
        self._seed = seed
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

        prev_layer_name, image_size = self._create_data_layer(preproc_params)

        assert len(all_conv_layers_params) == network_params['num_conv_layers']
        assert len(all_fc_layers_params) == network_params['num_fc_layers']

        for i, conv_layer_params in enumerate(all_conv_layers_params):
            current_layer_base_name = "conv_layer%d_" % i
            prev_layer_name, image_size = self._create_conv_layer(current_layer_base_name,
                                                                  prev_layer_name,
                                                                  conv_layer_params,
                                                                  image_size)
        self._output_image_size = image_size

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

        if self._test_file is not None:
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
            augmentation_layer.name = "data_augmentation"
            augmentation_layer.type = caffe_pb2.LayerParameter.DATA_AUGMENTATION
            augmentation_layer.data_param.mirror = True
            augmentation_layer.data_param.crop_size = int(augment_params["crop_size"])
            augmentation_layer.bottom.append("data")
            prev_layer_name = "augmented_data"
            augmentation_layer.top.append(prev_layer_name)

            if self._input_image_size is not None:
                output_size = self._input_image_size - 2 * augmentation_layer.data_param.crop_size
            else:
                output_size = None
        else:
            output_size = self._input_image_size

        dropout_params = params["input_dropout"]
        if dropout_params["type"] == "dropout":
            caffe_dropout_layer = self._caffe_net.layers.add()

            current_layer_name = "input_dropout"
            caffe_dropout_layer.name = current_layer_name
            caffe_dropout_layer.type = caffe_pb2.LayerParameter.DROPOUT
            caffe_dropout_layer.dropout_param.dropout_ratio = dropout_params.pop("dropout_ratio")
            caffe_dropout_layer.bottom.append(prev_layer_name)
            caffe_dropout_layer.top.append(current_layer_name)

            prev_layer_name = current_layer_name


        return prev_layer_name, output_size



    def _create_conv_layer(self, current_layer_base_name, prev_layer_name, params, image_size):
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
        if "kernelsize" in params:
            kernelsize = int(params.pop("kernelsize"))
        elif "kernelsize_odd" in params:
            kernelsize = int(params.pop("kernelsize_odd")) * 2 + 1
        else:
            assert False, "kernelsize missing for conv layer"
        caffe_conv_layer.convolution_param.kernel_size = kernelsize

        if "num_output_x_128" in params:
            num_output = int(params.pop("num_output_x_128")) * 128
        elif "num_output" in params:
            num_output = int(params.pop("num_output"))
        else:
            assert False, "num_output missing for conv layer"
        caffe_conv_layer.convolution_param.num_output = num_output

        kernelstride = int(params.pop("stride"))
        caffe_conv_layer.convolution_param.stride = kernelstride
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
        elif bias_filler_params["type"] == "const-value":
            caffe_conv_layer.convolution_param.bias_filler.value = float(bias_filler_params["value"])
        else:
            raise RuntimeError("unknown bias-filler %s" % (bias_filler_params["type"]))

#        bias_filler_params = params.pop("bias-filler")
#        bias_filler = caffe_conv_layer.layer.bias_filler
#        for param, param_val in bias_filler_params.iteritems():
#            setattr(bias_filler, param, param_val)
        caffe_conv_layer.blobs_lr.append(params.pop("weight-lr-multiplier"))
        caffe_conv_layer.blobs_lr.append(params.pop("bias-lr-multiplier"))

        if "weight-weight-decay_multiplier" in params:
            caffe_conv_layer.weight_decay.append(params.pop("weight-weight-decay_multiplier"))
        else:
            caffe_conv_layer.weight_decay.append(1.0)
        print ""
        if "bias-weight-decay_multiplier" in params:
            caffe_conv_layer.weight_decay.append(params.pop("bias-weight-decay_multiplier"))
        else:
            caffe_conv_layer.weight_decay.append(.0)

        padding_params = params.pop("padding")
        pad_size = 0
        if padding_params["type"] == "zero-padding":
            if "relative_size" in padding_params:
                pad_size = int(math.floor(float(padding_params["relative_size"]) * 0.5 * kernelsize))
                if pad_size > 0:
                    caffe_conv_layer.convolution_param.pad = pad_size
            elif "absolute_size" in padding_params:
                caffe_conv_layer.convolution_param.pad = int(padding_params["absolute_size"])
        elif padding_params["type"] == "implicit":
            #set it implicitly dependent on the kernel size:
            pad_size = int(math.floor(0.5 * kernelsize))
            caffe_conv_layer.convolution_param.pad = pad_size

        caffe_conv_layer.bottom.append(prev_layer_name)
        caffe_conv_layer.top.append(current_layer_name)

        #calculate the output image size after the convolution operation:
        if image_size is not None:
            output_image_size = (image_size + 2 * pad_size - kernelsize) / kernelstride + 1;
        else:
            output_image_size = None

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
            kernelsize = int(pooling_params["kernelsize"])
            stride = int(pooling_params["stride"])
            skip_layer = False
            if output_image_size is not None:
                new_output_image_size = (image_size - kernelsize) / stride + 1
                if self._restrict_to_legal_configurations and new_output_image_size < self._min_image_size:
                    skip_layer = True
                else:
                    output_image_size = new_output_image_size

            if not skip_layer:
                caffe_pool_layer = self._caffe_net.layers.add()

                current_layer_name = current_layer_base_name + "pool"
                caffe_pool_layer.name = current_layer_name
                caffe_pool_layer.type = caffe_pb2.LayerParameter.POOLING
                caffe_pool_layer.bottom.append(prev_layer_name)
                caffe_pool_layer.top.append(current_layer_name)
                caffe_pool_layer.pooling_param.pool = caffe_pb2.PoolingParameter.MAX
                caffe_pool_layer.pooling_param.kernel_size = kernelsize
                caffe_pool_layer.pooling_param.stride = stride

                prev_layer_name = current_layer_name
            else:
                #we don't do pooling, because we are restricted to only use legal configurations
                pass
        elif pooling_params["type"] == "ave":
            kernelsize = int(pooling_params["kernelsize"])
            stride = int(pooling_params["stride"])
            skip_layer = False
            if output_image_size is not None:
                new_output_image_size = (image_size - kernelsize) / stride + 1
                if self._restrict_to_legal_configurations and new_output_image_size < self._min_image_size:
                    skip_layer = True
                else:
                    output_image_size = new_output_image_size

            if not skip_layer:
                caffe_pool_layer = self._caffe_net.layers.add()

                current_layer_name = current_layer_base_name + "pool"
                caffe_pool_layer.name = current_layer_name
                caffe_pool_layer.type = caffe_pb2.LayerParameter.POOLING
                caffe_pool_layer.bottom.append(prev_layer_name)
                caffe_pool_layer.top.append(current_layer_name)
                caffe_pool_layer.pooling_param.pool = caffe_pb2.PoolingParameter.AVE
                caffe_pool_layer.pooling_param.kernel_size = kernelsize
                caffe_pool_layer.pooling_param.stride = stride

                prev_layer_name = current_layer_name
            else:
                #we don't do pooling in this layer, because we are restricted to only use legal configurations
                pass
        #TODO: add stochastic pooling

        normalization_params = params.pop("norm")
        if normalization_params["type"] == "lrn":
            caffe_norm_layer = self._caffe_net.layers.add()

            current_layer_name = current_layer_base_name + "norm"
            caffe_norm_layer.name = current_layer_name
            caffe_norm_layer.type = caffe_pb2.LayerParameter.LRN
            caffe_norm_layer.bottom.append(prev_layer_name)
            caffe_norm_layer.top.append(current_layer_name)
            caffe_norm_layer.lrn_param.local_size = int(normalization_params["local_size"])
            caffe_norm_layer.lrn_param.alpha = float(normalization_params["alpha"])
            caffe_norm_layer.lrn_param.beta = float(normalization_params["beta"])
            assert "norm_region" in  normalization_params
            norm_region_type = normalization_params["norm_region"]["type"]
            if norm_region_type == "across-channels":
                caffe_norm_layer.lrn_param.norm_region = caffe_pb2.LRNParameter.ACROSS_CHANNELS
            elif norm_region_type == "within-channels":
                caffe_norm_layer.lrn_param.norm_region = caffe_pb2.LRNParameter.WITHIN_CHANNEL

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

        return prev_layer_name, output_image_size

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

        if "weight-weight-decay_multiplier" in params:
            caffe_fc_layer.weight_decay.append(params.pop("weight-weight-decay_multiplier"))
        else:
            caffe_fc_layer.weight_decay.append(1.0)
        if "bias-weight-decay_multiplier" in params:
            caffe_fc_layer.weight_decay.append(params.pop("bias-weight-decay_multiplier"))
        else:
            caffe_fc_layer.weight_decay.append(.0)

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
        elif bias_filler_params["type"] == "const-value":
            caffe_fc_layer.inner_product_param.bias_filler.value = float(bias_filler_params["value"])
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

        assert "batch_size_train" in params, "batch size for training missing"
        self._batch_size_train = int(params.pop("batch_size_train"))

        train_iter_per_epoch = float(self._num_train) / self._batch_size_train

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
            half_life = train_iter_per_epoch * float(lr_policy_params.pop("half_life"))
            power = lr_policy_params.pop("power")
            self._solver.gamma = (2.**(1./power) - 1.) / half_life
            self._solver.power = power
        elif lr_policy == "inv_bergstra_bengio":
            #is this parametrization correct?
            half_life = train_iter_per_epoch * float(lr_policy_params.pop("half_life"))
            tau = 0.5 * half_life
            self._solver.stepsize = int(train_iter_per_epoch  * lr_policy_params.pop("epochcount"))
        assert len(lr_policy_params) == 0, "More learning policy arguments given, than needed, " + str(lr_policy_params)

        self._solver.momentum = params.pop("momentum")
        self._solver.weight_decay = params.pop("weight_decay")
        self._solver.train_net = self._train_network_file
        self._solver.test_net.append(self._valid_network_file)
        self._solver.test_iter.append(max(1, int(self._num_valid / self._batch_size_valid)))
        if self._test_file is not None:
            self._solver.test_net.append(self._test_network_file)
            self._solver.test_iter.append(max(1, int(self._num_test / self._batch_size_test)))

        for termination_criterion in self._termination_criterions:
            termination_criterion.add_to_solver_param(self._solver,
                iter_per_epoch=self._num_train / self._batch_size_train,
                tests_per_epoch=int(1./self._test_every_x_epoch))
            
        self._solver.random_seed = self._seed
        #test X times per epoch:
        self._solver.test_interval = max(1, int(self._test_every_x_epoch * train_iter_per_epoch))
        self._solver.display = max(1, int(0.01 * train_iter_per_epoch))
        self._solver.snapshot = 0
        self._solver.snapshot_prefix = self._snapshot_prefix
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

        if self._test_file is not None:
            with open(self._test_network_file, "w") as test_network:
                test_network.write(str(self._caffe_net_test))

    def run(self):
        """
            Run the given network and return the logging output.
        """
        self._serialize()
        return run_caffe(self._solver_file)


class ImagenetConvNet(CaffeConvNet):

    def __init__(self, **kwargs):
        super(ImagenetConvNet, self).__init__(**kwargs)

    def _convert_params_to_caffe_network(self, params):
        """
            Converts the given parameters into a caffe network configuration.
        """
        self._caffe_net = caffe_pb2.NetParameter()
        self._solver = caffe_pb2.SolverParameter()

        preproc_params, all_conv_layers_params, all_fc_layers_params, network_params = params

	if "conv_norm_constraint" in network_params:
	    self._conv_norm_constraint = network_params.pop("conv_norm_constraint")
	else:
	    self._conv_norm_constraint = None

        prev_layer_name, image_size = self._create_data_layer(preproc_params)

        assert len(all_conv_layers_params) == network_params['num_conv_layers']
        assert len(all_fc_layers_params) == network_params['num_fc_layers']

        for i, conv_layer_params in enumerate(all_conv_layers_params):
            current_layer_base_name = "conv_layer%d_" % i
            prev_layer_name, image_size = self._create_conv_layer(current_layer_base_name,
                                                                  prev_layer_name,
                                                                  conv_layer_params,
                                                                  image_size)
        self._output_image_size = image_size

        if "global_average_pooling" in network_params:
            global_average_pooling = network_params.pop("global_average_pooling")
            if global_average_pooling["type"] == "on":
                caffe_pool_layer = self._caffe_net.layers.add()

                current_layer_name = "global_pooling"
                caffe_pool_layer.name = current_layer_name
                caffe_pool_layer.type = caffe_pb2.LayerParameter.POOLING
                caffe_pool_layer.bottom.append(prev_layer_name)
                caffe_pool_layer.top.append(current_layer_name)
                caffe_pool_layer.pooling_param.pool = caffe_pb2.PoolingParameter.AVE
                caffe_pool_layer.pooling_param.kernel_size = 6
                caffe_pool_layer.pooling_param.stride = 6
                caffe_pool_layer.pooling_param.pad = 0

                prev_layer_name = current_layer_name

        for i, fc_layer_params in enumerate(all_fc_layers_params):
            current_layer_base_name = "fc_layer%d_" % i
            prev_layer_name = self._create_fc_layer(current_layer_base_name,
                                                   prev_layer_name,
                                                   fc_layer_params)

        self._create_network_parameters(network_params)
        if self._conv_norm_constraint is not None:
            self._solver.weight_constraint = True

    def _create_train_valid_networks(self):
        """
            Given the self._caffe_net base network we create
            different version for training, testing and predicting.
        """
        # train network
        self._caffe_net_train = copy.deepcopy(self._caffe_net)
        self._caffe_net_train.name = "train"
        self._caffe_net_train.layers[0].data_param.source = self._train_file
        self._caffe_net_train.layers[0].data_param.batch_size = self._batch_size_train
        self._caffe_net_train.layers[0].data_param.mean_file = self._mean_file

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
        self._caffe_net_validation.layers[0].data_param.source = self._valid_file
        self._caffe_net_validation.layers[0].data_param.batch_size = self._batch_size_valid
        self._caffe_net_validation.layers[0].data_param.mean_file = self._mean_file
        #if self._mean_file:
        #    self._caffe_net_validation.layers[0].layer.meanfile = self._mean_file

        if self._test_file is not None:
            self._caffe_net_test = copy.deepcopy(self._caffe_net)
            self._add_softmax_accuray_layers(self._caffe_net_test)
            self._caffe_net_test.name = "test"
            self._caffe_net_test.layers[0].data_param.source = self._test_file
            self._caffe_net_test.layers[0].data_param.batch_size = self._batch_size_test
            self._caffe_net_test.layers[0].data_param.mean_file = self._mean_file


    def _create_data_layer(self, params):
        data_layer = self._caffe_net.layers.add()
        data_layer.name = "data"
        data_layer.type = caffe_pb2.LayerParameter.DATA
        data_layer.data_param.backend = caffe_pb2.DataParameter.LMDB
        data_layer.top.append("data")
        data_layer.top.append("label")
        prev_layer_name = "data"

        augment_params = params["augment"]
        if augment_params["type"] == "augment":
            augmentation_layer = self._caffe_net.layers.add()
            augmentation_layer.name = "data_augmentation"
            augmentation_layer.type = caffe_pb2.LayerParameter.DATA_AUGMENTATION
            augmentation_layer.augmentation_param.mirror.rand_type = "bernoulli"
            augmentation_layer.augmentation_param.mirror.prob = 0.5
            augmentation_layer.augmentation_param.translate.rand_type = "uniform"
            augmentation_layer.augmentation_param.translate.mean = 0.
            augmentation_layer.augmentation_param.translate.spread = 0.1 # specified in percent
            augmentation_layer.augmentation_param.rotate.rand_type = "uniform"
            augmentation_layer.augmentation_param.rotate.mean = 0.
            augmentation_layer.augmentation_param.rotate.spread = np.deg2rad(float(augment_params["rotation_angle"]))
            zoom_coeff = float(augment_params["zoom_coeff"])
	    augmentation_layer.augmentation_param.zoom.rand_type = "uniform"
	    augmentation_layer.augmentation_param.zoom.spread = 0.5 * zoom_coeff
	    augmentation_layer.augmentation_param.zoom.mean = 0.5 * zoom_coeff
	    augmentation_layer.augmentation_param.zoom.exp = True
            # set color_distort and contrast parameters
            color_coeff = augment_params["color_distort"]
            contrast_coeff = augment_params["contrast"]
            brightness_coeff = augment_params["brightness"]

            if color_coeff > 0.:
                augmentation_layer.augmentation_param.col_add.rand_type = "uniform"
                augmentation_layer.augmentation_param.col_add.mean = 0.2 * color_coeff
                augmentation_layer.augmentation_param.col_add.spread = 0.3 * color_coeff

                augmentation_layer.augmentation_param.col_mult.rand_type = "gaussian"
                augmentation_layer.augmentation_param.col_mult.mean = 0.
                augmentation_layer.augmentation_param.col_mult.spread = 0.7 * color_coeff
                augmentation_layer.augmentation_param.col_mult.exp = True
            if contrast_coeff > 0.:
                augmentation_layer.augmentation_param.sat_add.rand_type = "uniform"
                augmentation_layer.augmentation_param.sat_add.mean = 0.3 * contrast_coeff
                augmentation_layer.augmentation_param.sat_add.spread = 0.5 * contrast_coeff

                augmentation_layer.augmentation_param.sat_mult.rand_type = "gaussian"
                augmentation_layer.augmentation_param.sat_mult.mean = 0.
                augmentation_layer.augmentation_param.sat_mult.spread = contrast_coeff
                augmentation_layer.augmentation_param.sat_mult.exp = True
            if brightness_coeff > 0.:
                augmentation_layer.augmentation_param.lmult_add.rand_type = "uniform"
                augmentation_layer.augmentation_param.lmult_add.mean = 0.3 * brightness_coeff
                augmentation_layer.augmentation_param.lmult_add.spread = 0.3 * brightness_coeff

                augmentation_layer.augmentation_param.lmult_mult.rand_type = "gaussian"
                augmentation_layer.augmentation_param.lmult_mult.mean = 0.
                augmentation_layer.augmentation_param.lmult_mult.spread = brightness_coeff
                augmentation_layer.augmentation_param.lmult_mult.exp = True
                                

            augmentation_layer.augmentation_param.crop_size = int(augment_params["crop_size"])
            augmentation_layer.bottom.append("data")
            prev_layer_name = "augmented_data"
            augmentation_layer.top.append(prev_layer_name)
        return prev_layer_name, None

    def _create_conv_layer(self, current_layer_base_name, prev_layer_name, params, image_size):
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
        if "kernelsize" in params:
            kernelsize = int(params.pop("kernelsize"))
        elif "kernelsize_odd" in params:
            kernelsize = int(params.pop("kernelsize_odd")) * 2 + 1
        else:
            assert False, "kernelsize missing for conv layer"
        caffe_conv_layer.convolution_param.kernel_size = kernelsize

        if "num_output_x_128" in params:
            num_output = int(params.pop("num_output_x_128")) * 128
        elif "num_output" in params:
            num_output = int(params.pop("num_output"))
        else:
            assert False, "num_output missing for conv layer"
        caffe_conv_layer.convolution_param.num_output = num_output

        caffe_conv_layer.max_weight_norm = self._conv_norm_constraint

        kernelstride = int(params.pop("stride"))
        caffe_conv_layer.convolution_param.stride = kernelstride
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
        elif bias_filler_params["type"] == "const-value":
            caffe_conv_layer.convolution_param.bias_filler.value = float(bias_filler_params["value"])
        else:
            raise RuntimeError("unknown bias-filler %s" % (bias_filler_params["type"]))

#        bias_filler_params = params.pop("bias-filler")
#        bias_filler = caffe_conv_layer.layer.bias_filler
#        for param, param_val in bias_filler_params.iteritems():
#            setattr(bias_filler, param, param_val)
        caffe_conv_layer.blobs_lr.append(params.pop("weight-lr-multiplier"))
        caffe_conv_layer.blobs_lr.append(params.pop("bias-lr-multiplier"))

        if "weight-weight-decay_multiplier" in params:
            caffe_conv_layer.weight_decay.append(params.pop("weight-weight-decay_multiplier"))
        else:
            caffe_conv_layer.weight_decay.append(1.0)
        print ""
        if "bias-weight-decay_multiplier" in params:
            caffe_conv_layer.weight_decay.append(params.pop("bias-weight-decay_multiplier"))
        else:
            caffe_conv_layer.weight_decay.append(.0)

        padding_params = params.pop("padding")
        pad_size = 0
        if padding_params["type"] == "zero-padding":
            if "relative_size" in padding_params:
                pad_size = int(math.floor(float(padding_params["relative_size"]) * 0.5 * kernelsize))
                if pad_size > 0:
                    caffe_conv_layer.convolution_param.pad = pad_size
            elif "absolute_size" in padding_params:
                caffe_conv_layer.convolution_param.pad = int(padding_params["absolute_size"])
        elif padding_params["type"] == "implicit":
            #set it implicitly dependent on the kernel size:
            pad_size = int(math.floor(0.5 * kernelsize))
            caffe_conv_layer.convolution_param.pad = pad_size

        caffe_conv_layer.bottom.append(prev_layer_name)
        caffe_conv_layer.top.append(current_layer_name)

        #calculate the output image size after the convolution operation:
        if image_size is not None:
            output_image_size = (image_size + 2 * pad_size - kernelsize) / kernelstride + 1;
        else:
            output_image_size = None

        prev_layer_name = current_layer_name

        activation = params.pop("activation")

        if activation["type"]  == "relu":
            # Relu
            caffe_relu_layer = self._caffe_net.layers.add()

            current_layer_name = current_layer_base_name + "relu"
            caffe_relu_layer.name = current_layer_name
            caffe_relu_layer.type = caffe_pb2.LayerParameter.RELU
            caffe_relu_layer.bottom.append(prev_layer_name)
            #Note: the operation is made in-place by using the same name twice
            caffe_relu_layer.top.append(prev_layer_name)
        elif activation["type"]  == "subspace-pooling":
            # subspace pooling
            caffe_subspace_pooling_layer = self._caffe_net.layers.add()

            current_layer_name = current_layer_base_name + "subspace_pooling"
            caffe_subspace_pooling_layer.name = current_layer_name
            caffe_subspace_pooling_layer.type = caffe_pb2.LayerParameter.SUBSPACEPOOLING
            caffe_subspace_pooling_layer.pooling_param.pool = caffe_pb2.PoolingParameter.MAX
            caffe_subspace_pooling_layer.pooling_param.kernel_size = 2
            caffe_subspace_pooling_layer.pooling_param.stride = 2
            caffe_subspace_pooling_layer.pooling_param.pad = 0
            caffe_subspace_pooling_layer.bottom.append(prev_layer_name)
            #Note: the operation is made in-place by using the same name twice
            caffe_subspace_pooling_layer.top.append(current_layer_name)

            prev_layer_name = current_layer_name
        else:
            assert False, "unkown activation"

        # Pooling
        pooling_params = params.pop("pooling")
        if pooling_params["type"] == "none":
            pass
        elif pooling_params["type"] == "max":
            kernelsize = int(pooling_params["kernelsize"])
            stride = int(pooling_params["stride"])
            caffe_pool_layer = self._caffe_net.layers.add()

            current_layer_name = current_layer_base_name + "pool"
            caffe_pool_layer.name = current_layer_name
            caffe_pool_layer.type = caffe_pb2.LayerParameter.POOLING
            caffe_pool_layer.bottom.append(prev_layer_name)
            caffe_pool_layer.top.append(current_layer_name)
            caffe_pool_layer.pooling_param.pool = caffe_pb2.PoolingParameter.MAX
            caffe_pool_layer.pooling_param.kernel_size = kernelsize
            caffe_pool_layer.pooling_param.stride = stride

            prev_layer_name = current_layer_name
        elif pooling_params["type"] == "ave":
            kernelsize = int(pooling_params["kernelsize"])
            stride = int(pooling_params["stride"])
            caffe_pool_layer = self._caffe_net.layers.add()

            current_layer_name = current_layer_base_name + "pool"
            caffe_pool_layer.name = current_layer_name
            caffe_pool_layer.type = caffe_pb2.LayerParameter.POOLING
            caffe_pool_layer.bottom.append(prev_layer_name)
            caffe_pool_layer.top.append(current_layer_name)
            caffe_pool_layer.pooling_param.pool = caffe_pb2.PoolingParameter.AVE
            caffe_pool_layer.pooling_param.kernel_size = kernelsize
            caffe_pool_layer.pooling_param.stride = stride

            prev_layer_name = current_layer_name
        else:
            assert False, "unkown pooling layer type"
        #TODO: add stochastic pooling

        normalization_params = params.pop("norm")
        if normalization_params["type"] == "lrn":
            caffe_norm_layer = self._caffe_net.layers.add()

            current_layer_name = current_layer_base_name + "norm"
            caffe_norm_layer.name = current_layer_name
            caffe_norm_layer.type = caffe_pb2.LayerParameter.LRN
            caffe_norm_layer.bottom.append(prev_layer_name)
            caffe_norm_layer.top.append(current_layer_name)
            caffe_norm_layer.lrn_param.local_size = int(normalization_params["local_size"])
            caffe_norm_layer.lrn_param.alpha = float(normalization_params["alpha"])
            caffe_norm_layer.lrn_param.beta = float(normalization_params["beta"])
            assert "norm_region" in  normalization_params
            norm_region_type = normalization_params["norm_region"]["type"]
            if norm_region_type == "across-channels":
                caffe_norm_layer.lrn_param.norm_region = caffe_pb2.LRNParameter.ACROSS_CHANNELS
            elif norm_region_type == "within-channels":
                caffe_norm_layer.lrn_param.norm_region = caffe_pb2.LRNParameter.WITHIN_CHANNEL

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

        return prev_layer_name, output_image_size


def run_caffe(solver_file, hide_output=True):
    """
        Runs caffe and returns the logging output.
    """
    try:
        os.environ["GLOG_logtostderr"] = "1"
        output = check_output(["train_net.sh", solver_file], stderr=STDOUT)
        return output
    except CalledProcessError as e:
        raise RuntimeError("Failed running caffe. Return code: %s Output: %s" % (str(e.returncode), str(e.output)))
    except:
        print "UNKNOWN ERROR!"
        raise RuntimeError("Unknown exception, when running caffe." + traceback.format_exc())





