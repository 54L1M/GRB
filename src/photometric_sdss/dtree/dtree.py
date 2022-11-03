# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import pandas as pd

# Importing the dataset


def dtree(data):
    features = np.zeros(shape=(len(data), 4))
    features[:, 0] = data["u"] - data["g"]
    features[:, 1] = data["g"] - data["r"]
    features[:, 2] = data["r"] - data["i"]
    features[:, 3] = data["i"] - data["z"]
    targets = data["redshift"]
    # Splitting the dataset into the Training and Test set

    features_train, features_test, targets_train, targets_test = train_test_split(
        features, targets, test_size=0.2, random_state=0
    )
    # initialize model
    regressor = DecisionTreeRegressor(max_depth=17, random_state=0)
    regressor.fit(features_train, targets_train)
    # get the predicted_redshifts
    y_pred = regressor.predict(features_test)

    accuracies = cross_val_score(
        estimator=regressor, X=features_train, y=targets_train, cv=10
    )
    
    df = pd.DataFrame(y_pred)
    df.to_csv('y_pred.csv',index=False)
    df = pd.DataFrame(targets_test)
    df.to_csv("targets_test.csv",index=False)
    
    return [y_pred, targets_test, accuracies]


def median_diff(predicted, actual):
    return np.median(np.abs(predicted[:] - actual[:]))


def plot_tree(data):
    y_pred, targets_test, accuracies = dtree(data)
    cmap = plt.get_cmap("hot")
    xy = np.vstack([targets_test, y_pred])
    z = gaussian_kde(xy)(xy)
    plot = plt.scatter(targets_test, y_pred, c=z, cmap=cmap, s=0.4)
    plt.colorbar(plot)
    plt.xlim((0, 3))
    plt.ylim((0, 3))
    plt.xlabel("Measured Redshift")
    plt.ylabel("Predicted Redshift")
    plt.savefig("output/plot/Tree_Result", dpi=1200)
    plt.show()


def R2(targets_test, y_pred):
    R2 = r2_score(targets_test, y_pred)
    return R2


def main_tree(data):
    y_pred, targets_test, accuracies = dtree(data)
    diff = median_diff(y_pred, targets_test)
    znorm = []
    znorm = (targets_test[:] - y_pred[:]) / (targets_test[:] + 1)
    deltaz=(targets_test[:] - y_pred[:])
    df = pd.DataFrame(deltaz)
    df.to_csv('deltaz_dtree_total.csv',index=False)
    df1 = pd.DataFrame(targets_test)
    df1.to_csv('specz_dtree_total.csv',index=False)
    print(np.mean(znorm))
    print(f"Median difference of decision tree: {diff}")
    print("Accuracy decision tree: {} %".format(accuracies.mean() * 100))
    print("Standard Deviation decision tree: {} %".format(accuracies.std() * 100))
    delta_tree = y_pred - targets_test
    return delta_tree


def run(data):
    y_pred, targets_test, accuracies = dtree(data)
    diff = median_diff(y_pred, targets_test)
    print(f"Median difference of decision tree: {diff}")
    print("Accuracy decision tree: {} %".format(accuracies.mean() * 100))
    print("Standard Deviation decision tree: {} %".format(accuracies.std() * 100))
    cmap = plt.get_cmap("hot")
    xy = np.vstack([targets_test, y_pred])
    z = gaussian_kde(xy)(xy)
    plot = plt.scatter(targets_test, y_pred, c=z, cmap=cmap, s=0.4)
    plt.colorbar(plot)
    plt.xlim((0, 7))
    plt.ylim((0, 7))
    plt.clim(0, 10)
    plt.xlabel("Measured Redshift")
    plt.ylabel("Predicted Redshift")
    plt.savefig("Tree_Result_total", dpi=800)
    plt.show()
    return [y_pred, targets_test]
