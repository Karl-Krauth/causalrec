import os

import numpy as np


data_dir = os.path.join(os.path.dirname(__file__), "../..", "data")


def save_experiment(recommendations, predictions, experiment_name):
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    file_path = os.path.join(data_dir, experiment_name + ".npz")
    np.savez(file_path, recommendations=recommendations, predictions=predictions)


def load_experiment(experiment_name):
    file_path = os.path.join(data_dir, experiment_name + ".npz")
    print(file_path)
    if os.path.isfile(file_path):
        result = np.load(file_path, allow_pickle=True)
        return result["recommendations"], result["predictions"]
    else:
        return None, None


def save_statistic(name, stat):
    file_path = os.path.join(data_dir, name + ".npz")
    np.savez(file_path, **stat)


def load_statistic(name):
    file_path = os.path.join(data_dir, name + ".npz")
    if os.path.isfile(file_path):
        return np.load(file_path, allow_pickle=True)
    else:
        return None
