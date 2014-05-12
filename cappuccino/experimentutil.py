import fcntl
import json
import os
import re
import defaultdict


def store_result(dirname, params, result, learning_curve):
    """
        Store the results in a central file, one line of json per experiment.

    """
    result_file_name = os.path.join(dirname, "results.json")
    with open(result_file_name, "a") as result_file:
        #lock file:
        fcntl.lockf(result_file.fileno(), fcntl.LOCK_EX)

        result_file.write(json.dumps({"loss": result,
                                      "params": params,
                                      "learning_curve": learning_curve}))
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
            learning_curve_timestamps.append(m.group(1))
        m = re.match(line_regex, line.strip())
        if m:
            network_name = m.group(1)
            accuracy = m.group(2)

            network_learning_curves[network_name].append(accuracy)

    return network_learning_curve, learning_curve_timestamps
