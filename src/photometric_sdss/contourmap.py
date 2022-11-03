import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Complete the following to make the plot
if __name__ == "__main__":
    data = pd.read_csv("data/output_under2.csv")
    # Get a colour map
    cmap = plt.get_cmap("Greens")

    # Define our colour indexes u-g and r-i
    u_g = data["u"] - data["g"]
    r_i = data["r"] - data["i"]

    # Make a redshift array
    redshift = data["redshift"]

    # Create the plot with plt.scatter
    plot = plt.scatter(u_g, r_i, s=0.5, lw=0, c=redshift, cmap=cmap)

    cb = plt.colorbar(plot)
    cb.set_label("Redshift")

    # Define your axis labels and plot title
    plt.clim(0, 2.0)
    plt.xlabel("Color index  u-g")
    plt.ylabel("Color index  r-i")
    plt.title("Redshift (color) u-g versus r-i")

    # Set any axis limits
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 1)
    plt.savefig("Contourmap_under2", dpi=600)
    plt.show()
