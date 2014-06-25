import fcntl
import json
import os
import re
import sys
import traceback
import numpy as np
from collections import defaultdict
from cappuccino.paramutil import hpolib_to_caffenet


def get_current_ybest():
    ybest_curr = None
    if os.path.exists("ybest.txt"):
        ybest_curr = float(open("ybest.txt").read())
    return ybest_curr

def update_ybest(y_candidate):
   """ 
        y_candidate: latest accuracy

        Set ybest in ybest.txt if y_candidate is lower than the previous ybest.

        returns the current ybest.
   """
   ybest_curr = get_current_ybest()
   if ybest_curr is None or y_candidate > ybest_curr:
        with open("ybest.txt", "w") as ybest_file:
            ybest_file.write(str(y_candidate))
        return y_candidate
   else:
        return ybest_curr

def store_result(dirname, params, loss, total_time, learning_curves,
    learning_curve_timestamps, predicted_loss=None, extra={}):
    """
        Store the results in a central file, one line of json per experiment.

    """
    result_file_name = os.path.join(dirname, "results.json")
    with open(result_file_name, "a") as result_file:
        #lock file:
        fcntl.lockf(result_file.fileno(), fcntl.LOCK_EX)
        result_line = {"loss": loss,
                       "total_time": total_time,
                       "params": params,
                       "learning_curves": learning_curves,
                       "learning_curve_timestamps": learning_curve_timestamps}
        if predicted_loss is not None:
            result_line["predicted_loss"] = predicted_loss
        result_line.update(extra)
        result_file.write(json.dumps(result_line))
        result_file.write("\n")

def log_error(dirname, error_msg):
    """
        Store the errors that occur in a central file
    """
    error_log_file_name = os.path.join(dirname, "errors.txt")
    with open(error_log_file_name, "a") as error_log_file:
        #lock file:
        fcntl.lockf(error_log_file.fileno(), fcntl.LOCK_EX)

        error_log_file.write(error_msg)
        error_log_file.write("\n")
        error_log_file.write("\n")


def read_learning_curve():
    """
        Read the learning curve from a file.
        Expects the file learning_curve.txt in the current folder.
    """
    with open("learning_curve.txt") as f:
        learning_curve = f.read().split(",")[:-1]
        return learning_curve
    return []


def learning_curve_from_log(lines):
    """
    lines: the line by line output of caffe

    returns learning curves for each network, timestamps
    """
    #example test accuracy:
    #I0512 15:43:21.701407 13354 solver.cpp:183] valid test score #0: 0.0792
    line_regex = "[^]]+]\s(\w+)\stest score\s#0:\s(\d+\.?\d*)"
    #test timestamp line
    #I0512 16:29:38.952080 13854 solver.cpp:141] Test timestamp 1399904978
    time_regex = "[^]]+] Test timestamp (\d+)"

    network_learning_curves = defaultdict(list)

    learning_curve_timestamps = []

    mday = 1
    for line in lines:
        m = re.match(time_regex, line.strip())
        if m:
            learning_curve_timestamps.append(int(m.group(1)))
        m = re.match(line_regex, line.strip())
        if m:
            network_name = m.group(1)
            accuracy = float(m.group(2))

            network_learning_curves[network_name].append(accuracy)

    return network_learning_curves, learning_curve_timestamps


def hpolib_experiment_main(params, construct_caffeconvnet,
    experiment_dir, working_dir, mean_performance_on_last, **kwargs):
    """
        params: parameters coming directly from hpolib
        construct_caffeconvnet: a function that takes caffeconvnet parameters and constructs a CaffeConvNet
        mean_performance_on_last: take average of the last x values from the validation network as the reported performance.
    """
    try:
        caffe_convnet_params = hpolib_to_caffenet(params)

        output_log = construct_caffeconvnet(caffe_convnet_params).run()
        learning_curves, learning_curve_timestamps = learning_curve_from_log(output_log.split("\n"))
        if "valid" not in learning_curves:
            log_error(experiment_dir, output_log)
            raise Exception("no learning curve found")
        valid_learning_curve = learning_curves["valid"]
        best_accuracy = np.mean(valid_learning_curve[-mean_performance_on_last:])
        if not np.isfinite(best_accuracy):
            best_accuracy = .0
        lowest_error = 1.0 - best_accuracy
        total_time = learning_curve_timestamps[-1] - learning_curve_timestamps[0]

        #read output from termination_criterion
        best_predicted_accuracy = None
        lowest_predicted_error = None
        if os.path.exists("y_predict.txt"):
            best_predicted_accuracy = float(open("y_predict.txt").read())
            lowest_predicted_error = 1.0 - best_predicted_accuracy
            #make sure we don't use it in the next run as well..
            os.remove("y_predict.txt")

        try:
            current_ybest = get_current_ybest()
            store_result(experiment_dir, caffe_convnet_params, lowest_error, total_time,
                         learning_curves, learning_curve_timestamps, predicted_loss=lowest_predicted_error,
                         extra={"current_ybest": current_ybest})
            store_result(working_dir, caffe_convnet_params, lowest_error, total_time,
                         learning_curves, learning_curve_timestamps, predicted_loss=lowest_predicted_error,
                         extra={"current_ybest": current_ybest})
            update_ybest(best_accuracy)
        except Exception as e:
            print "Unexpected error:", sys.exc_info()[0]
            print "Trackback: ", traceback.format_exc()
            log_error(experiment_dir, str(sys.exc_info()[0]))
            log_error(experiment_dir, str(traceback.format_exc()))
        finally:
            if np.isfinite(lowest_error):
                return lowest_error
            else:
                raise Exception("RESULT NOT FINITE!")
    except Exception:
        print "Unexpected error:", sys.exc_info()[0]
        print "Trackback: ", traceback.format_exc()
        log_error(experiment_dir, str(sys.exc_info()[0]))
        log_error(experiment_dir, str(traceback.format_exc()))
        #raise e
        #maximum loss:
        return 1.0
