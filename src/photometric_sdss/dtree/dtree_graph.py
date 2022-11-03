import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import tree
from dtreeviz.trees import dtreeviz

data = np.load("../data/sdss_galaxy_450000.npy")


features = np.zeros(shape=(len(data), 4))
features[:, 0] = data["u"] - data["g"]
features[:, 1] = data["g"] - data["r"]
features[:, 2] = data["r"] - data["i"]
features[:, 3] = data["i"] - data["z"]
targets = data["redshift"]
features_train, features_test, targets_train, targets_test = train_test_split(
    features, targets, test_size=0.2, random_state=0
)
regressor = DecisionTreeRegressor(max_depth=3, random_state=0)
regressor = regressor.fit(features_train, targets_train)
fig = plt.figure(figsize=(100, 80))
_ = tree.plot_tree(
    regressor,
    filled=True,
    feature_names=["u-g", "g-r", "r-i", "i-z"],
)
fig.savefig("../output/plot/decistion_tree_viz.png")
