# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt


# train\valid
# n_estimator_sample = [10, 70, 100, 300, 500, 600, 700]
# max_depths_sample = [10, 14, 15, 16, 17, 19, 25, 70, 150]
n_estimator_sample = [10, 20, 30]
max_depths_sample = [10, 14, 16]
# Importing the dataset
data = np.load("data\sdss_galaxy_colors.npy")

features = np.zeros(shape=(len(data), 4))
features[:, 0] = data["u"] - data["g"]
features[:, 1] = data["g"] - data["r"]
features[:, 2] = data["r"] - data["i"]
features[:, 3] = data["i"] - data["z"]
targets = data["redshift"]
# Splitting the dataset into the Training and Test set
from sklearn.model_selection import train_test_split

features_train, features_test, targets_train, targets_test = train_test_split(
    features, targets, test_size=0.2, random_state=0
)


def median_diff(predicted, actual):
    return np.median(np.abs(predicted[:] - actual[:]))


median_train = []
median_test = []
mdeplist = []
# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor

for est in n_estimator_sample:
    lx = []
    ly = []
    lz = []
    for mdep in max_depths_sample:
        regressor = RandomForestRegressor(
            n_estimators=est, max_depth=(mdep), random_state=0
        )
        regressor.fit(features_train, targets_train)
        y_pred_train = regressor.predict(features_train)
        y_pred_test = regressor.predict(features_test)
        diff_train = median_diff(y_pred_train, targets_train)
        diff_test = median_diff(y_pred_test, targets_test)
        lx.append(mdep)
        ly.append(diff_test)
        lz.append(diff_train)

    mdeplist.append(lx)
    median_test.append(ly)
    median_train.append(lz)


for i in range(len(median_train)):
    plt.plot(mdeplist[i], median_test[i], label=f"Test, N: {n_estimator_sample[i]}")
    plt.plot(mdeplist[i], median_train[i], label=f"Train, N: {n_estimator_sample[i]}")
# plt.savefig("train-val", dpi=1200)
plt.legend()
plt.show()
