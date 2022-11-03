"""
A simple script that converts text file to csv file

pass the text file address as a system argument and

the rest will happen!
"""
import pandas as pd
import sys


def txt2csv(file):
    """convert text file to csv file

    Args:
        file (text file): a text file which you want to convert

    Returns:
        csv file: converted csv file
    """
    dataframe = pd.read_csv(file, sep="\t")

    return dataframe.to_csv(str(file).replace(".txt", ".csv"), index=None)


if __name__ == "__main__":
    txt2csv(sys.argv[1])
