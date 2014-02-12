import fcntl
import json

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

def read_learning_curve():
    """
        Read the learning curve from a file.
        Expects the file learning_curve.txt in the current folder.
    """
    with open("learning_curve.txt") as f:
        learning_curve = f.read().split(",")[:-1]
        return learning_curve
    return []

