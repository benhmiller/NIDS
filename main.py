#   Author        : *** INSERT YOUR NAME ***
#   Last Modified : *** DATE ***

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
import argparse
import helper
from sklearn import tree
from sklearn import metrics
from sklearn import neural_network
from sklearn import model_selection


def exp1(data, labels):
    """STUDENT CODE BELOW"""
    # define model architecure
    model = tree.DecisionTreeClassifier(max_depth=1)
    """STUDENT CODE ABOVE"""
    return model


def exp2(data, labels):
    """STUDENT CODE BELOW"""
    # define model architecture
    model = tree.DecisionTreeClassifier(max_depth=1)
    """STUDENT CODE ABOVE"""
    return model


def exp3(data, labels):
    """STUDENT CODE BELOW"""
    # define model architecture
    model = neural_network.MLPClassifier(hidden_layer_sizes=(50,))
    """STUDENT CODE ABOVE"""
    return model


def exp4(data, labels):
    """STUDENT CODE BELOW"""
    # define model architecture
    model = neural_network.MLPClassifier(hidden_layer_sizes=(50,))
    """STUDENT CODE ABOVE"""
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, default=1, help="Experiment number")
    args = parser.parse_args()
    save_name = f"exp{args.exp}_model" + (".pkl")
    if args.exp == 1:
        model = exp1(*helper.load_dataset(multiclass=False, normalize=False))
    elif args.exp == 2:
        model = exp2(*helper.load_dataset(multiclass=True, normalize=False))
    elif args.exp == 3:
        model = exp3(*helper.load_dataset(multiclass=False, normalize=True))
    elif args.exp == 4:
        model = exp4(*helper.load_dataset(multiclass=True, normalize=True))
    else:
        print("Invalid experiment number")
    helper.save_model(model, save_name)
