#   Author        : Benjamin Miller
#   Last Modified : 04 / 18 / 2024

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
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV


def exp1(data, labels):
    """STUDENT CODE BELOW"""
    # define model architecure
    # model = tree.DecisionTreeClassifier(max_depth=1)
    # print(data.shape, labels.shape)

    # Prepare local variables
    X = data
    y = labels
    tree_sizes = range(10,16)

    # Split Data Into Train, Test, and Validation Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Determine optimal tree depth using cross validation
    cv_scores = []
    for depth in tree_sizes:
        # Create decision tree classifier with current size
        clf = tree.DecisionTreeClassifier(max_depth=depth, random_state=42)

        # Perform 5-fold cross-validation on the training set
        scores = cross_val_score(clf, X_train, y_train, cv=5)

        # Compute average accuracy across flds
        avg_score = np.mean(scores)

        # Append score to list
        cv_scores.append(avg_score)

    # Determine Optimal Tree Size and Score
    optimal_size_index = np.argmax(cv_scores)
    optimal_size = tree_sizes[optimal_size_index]
    optimal_score = cv_scores[optimal_size_index]

    print("Optimal Tree Size:", optimal_size)
    print("Average Cross-Validation Accuracy:", optimal_score)

    # Plot Accuracy vs. Tree Depth
    plt.plot(tree_sizes, cv_scores, marker='o')
    plt.title('Accuracy vs. Tree Depth')
    plt.xlabel('Tree Depth')
    plt.ylabel('Cross-Validation Accuracy')

    # Save the plot as an image file
    plt.savefig('figures/accuracy_vs_tree_depth.png')

    # Show the plot (optional)
    plt.show()

    # Train final model using the optimal tree size on the entire training set
    model = tree.DecisionTreeClassifier(max_depth=optimal_size)
    model.fit(X_train, y_train)

    # Evaluate final model on the test set
    test_accuracy = model.score(X_test, y_test)
    print("Test Set Accuracy:", test_accuracy)
    """STUDENT CODE ABOVE"""
    return model


def exp2(data, labels):
    """STUDENT CODE BELOW"""
    # define model architecture
    #model = tree.DecisionTreeClassifier(max_depth=1)

    # Prepare local variables
    X = data
    y = labels
    tree_sizes = range(5,15)

    # Split Data Into Train, Test, and Validation Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Determine optimal tree depth using cross validation
    cv_scores = []
    for depth in tree_sizes:
        # Create decision tree classifier with current size
        clf = tree.DecisionTreeClassifier(max_depth=depth)

        # Perform 5-fold cross-validation on the training set
        scores = cross_val_score(clf, X_train, y_train, cv=5)

        # Compute average accuracy across flds
        avg_score = np.mean(scores)

        # Append score to list
        cv_scores.append(avg_score)

    # Determine Optimal Tree Size and Score
    optimal_size_index = np.argmax(cv_scores)
    optimal_size = tree_sizes[optimal_size_index]
    optimal_score = cv_scores[optimal_size_index]

    print("Optimal Tree Size:", optimal_size)
    print("Average Cross-Validation Accuracy:", optimal_score)

    # Train final model using the optimal tree size on the entire training set
    model = tree.DecisionTreeClassifier(max_depth=optimal_size)
    model.fit(X_train, y_train)

    # Evaluate final model on the test set
    test_accuracy = model.score(X_test, y_test)
    print("Test Set Accuracy:", test_accuracy)
    """STUDENT CODE ABOVE"""
    return model


def exp3(data, labels):
    """STUDENT CODE BELOW"""
    # define model architecture
    model = neural_network.MLPClassifier(hidden_layer_sizes=(50,))

    # Prepare local variables
    X = data
    y = labels

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train Model (with default values)
    model.fit(X_train, y_train)

    # Evaluate final model on the test set
    test_accuracy = model.score(X_test, y_test)
    print("Test Set Accuracy:", test_accuracy)

    """STUDENT CODE ABOVE"""
    return model


def exp4(data, labels):
    """STUDENT CODE BELOW"""
    
    # define model architecture
    model = neural_network.MLPClassifier(hidden_layer_sizes=(100,))
    #'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],

    # Prepare local variables
    X = data
    y = labels

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train Model (with default values)
    model.fit(X_train, y_train)

    # Evaluate final model on the test set
    test_accuracy = model.score(X_test, y_test)
    print("Test Set Accuracy:", test_accuracy)
    '''
    # Prepare local variables
    X = data
    y = labels

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Define MLPClassifier
    mlp = neural_network.MLPClassifier()

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        #'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        #'activation': ['relu', 'tanh', 'logistic'],
        #'alpha': [0.0001, 0.001, 0.01, 0.1],
        #'learning_rate_init': [0.001, 0.01, 0.1]
    }

    # Perform GridSearchCV
    grid_search = GridSearchCV(mlp, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get best model and its parameters
    model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Evaluate final model on the test set
    test_accuracy = model.score(X_test, y_test)
    print("Test Set Accuracy:", test_accuracy)
    '''
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
