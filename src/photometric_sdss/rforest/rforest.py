# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Importing the dataset
def rforest(data):
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
    # Training the Random Forest Regression model on the whole dataset
    regressor = RandomForestRegressor(n_estimators=600, max_depth=13, random_state=0)
    regressor.fit(features_train, targets_train)

    y_pred = regressor.predict(features_test)
    return [y_pred, targets_test]


def median_diff(predicted, actual):
    return np.median(np.abs(predicted[:] - actual[:]))


def plot_forest(y_pred,targets_test):
    # plot the results to see how well our model looks
    
    cmap = plt.get_cmap("plasma")
    xy = np.vstack([targets_test, y_pred])
    z = gaussian_kde(xy)(xy)
    # Create the plot with plt.scatter
    plot = plt.scatter(targets_test, y_pred, c=z, cmap=cmap, s=0.4)

    cb = plt.colorbar(plot)
    # plt.scatter(targets_test, y_pred, s=0.4)
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.xlabel("Measured Redshift")
    plt.ylabel("Predicted Redshift")
    plt.savefig("Forest_Result_under2", dpi=800)
    plt.show()


def main_forest(y_pred,targets_test):
    
    diff = median_diff(y_pred, targets_test)
    znorm = []
    znorm = (targets_test[:] - y_pred[:]) / (targets_test[:] + 1)
    print(np.mean(znorm))
    deltaz=(targets_test[:] - y_pred[:])
    df = pd.DataFrame(deltaz)
    df.to_csv('deltaz_rforest_total.csv',index=False)
    df1 = pd.DataFrame(targets_test)
    df1.to_csv('specz_rforest_total.csv',index=False)
    print(np.mean(znorm))
    print(f"Median difference of random forest: {diff}")
    delta_forest = y_pred - targets_test
    return delta_forest
