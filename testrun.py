import pprint

#hyperopt:
import hyperopt.pyll.stochastic
from hyperopt import fmin, tpe, hp, STATUS_OK

from caffeconvnet import CaffeConvNet
from tpesearchspace import TPEConvNetSearchSpace

def test_fun(kwargs):
    print "Test fun called, parameters:"
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(kwargs)
    caffe = CaffeConvNet(kwargs,
                         train_file="/home/domhant/data/cifar10/caffe/cifar-train-leveldb",
                         valid_file="/home/domhant/data/cifar10/caffe/cifar-test-leveldb",
                         mean_file="/home/domhant/data/cifar10/caffe/cifar-train-mean.binaryproto",
                         num_validation_set_batches=100,
                         batch_size_train=128,
                         batch_size_valid=100)
    return caffe.run()

def test():
    space = TPEConvNetSearchSpace()
    tpe_space = TPEConvNetSearchSpace().get_tpe_search_space()
    print "TPE search space"
    print tpe_space
    print "Search space samples:"
    for i in range(0,10):
        print hyperopt.pyll.stochastic.sample(tpe_space)

    best = fmin(test_fun, space=tpe_space, algo=tpe.suggest, max_evals=1)


if __name__ == "__main__":
    test()
