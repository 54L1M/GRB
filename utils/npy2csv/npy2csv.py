import pandas as pd
import numpy as np
import sys


def npy2csv(file):
    data = np.load(file)
    data_frame = pd.DataFrame(data)
    return data_frame.to_csv(str(file).replace(".npy", ".csv"), index=None)


if __name__ == "__main__":
    npy2csv(sys.argv[1])
