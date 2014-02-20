import pprint

import random

#hyperopt:
import hyperopt.pyll.stochastic
from hyperopt import fmin, tpe, hp, STATUS_OK

import cappuccino
from cappuccino.convnetsearchspace import ConvNetSearchSpace
from cappuccino.tpesearchspace import convnet_space_to_tpe
from cappuccino.tpesearchspace import tpe_sample_to_caffenet
from cappuccino.caffeconvnet import CaffeConvNet

def test_fun(kwargs):
    params = tpe_sample_to_caffenet(kwargs)
    print "Test fun called, parameters:"
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(params)
    caffe = CaffeConvNet(params,
                         train_file="/home/domhant/data/cifar10/caffe/cifar-train-leveldb-2",
                         valid_file="/home/domhant/data/cifar10/caffe/cifar-test-leveldb-2",
                         mean_file="/home/domhant/data/cifar10/caffe/cifar-train-mean.binaryproto",
                         num_validation_set_batches=100,
                         batch_size_train=128,
                         batch_size_valid=100)
    return caffe.run()

def test():
    space = ConvNetSearchSpace((3, 32, 32))
    tpe_space = convnet_space_to_tpe(space)
    print "TPE search space"
    print tpe_space
    print "Search space samples:"
#    for i in range(0,10):
#        print hyperopt.pyll.stochastic.sample(tpe_space)

    best = fmin(test_fun, space=tpe_space, algo=tpe.suggest, max_evals=10)


if __name__ == "__main__":
    test()
