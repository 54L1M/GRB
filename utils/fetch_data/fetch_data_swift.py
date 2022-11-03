#!/usr/local/bin/python3
import os
import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import UnivariateSpline
import seaborn as sns

# sns.set_style("darkgrid", {'xtick.minor.width': 3,'xtick.minor.size': 5,'ytick.minor.size': 0,'xtick.direction': u'in'})
sns.set_style("darkgrid", {"xtick.minor.size": 0, "ytick.minor.size": 0})

plt.figure(figsize=(10, 6.5))

plt.interactive(False)

#################################

GRB = "151027A"
# if len(sys.argv): GRB = sys.argv[1]
autoSmoothX = False
smoothX = 0.4  # smooth of spline for X-ray
smoothO = 0.05  # smooth of spline for Optical
timeCut = 1e7

# flare information
flareInfoGRB = "total1021.csv"

################################

# x-ray files
path = "./XRays"
filePath = os.path.join(path, GRB)

# optical files
path2 = "./Opticals"
filePath2 = os.path.join(path2, GRB + "_O.txt")


def readInfo(filename):
    tPeaks = {}
    redshifts = {}
    deltaTs = {}
    energys = {}
    with open(filename) as fd:
        data = csv.DictReader(fd, delimiter=",")
        for row in data:
            key = row.pop("GRB")
            redshifts[key] = float(row.pop("redshift"))
            tPeaks[key] = float(row.pop("middle"))
            deltaTs[key] = float(row.pop("deltaT"))
            energys[key] = float(row.pop("energy"))
    return redshifts, tPeaks, deltaTs, energys


def readX(filename, cutoff=1e8):
    times = []
    lumins = []
    luminErrPoss = []
    luminErrNegs = []
    with open(filename) as fd:
        data = csv.DictReader(
            fd,
            delimiter=" ",
            fieldnames=(
                "time",
                "lumin",
                "timeErrPos",
                "timeErrNeg",
                "luminErrPos",
                "luminErrNeg",
                "PI",
                "PIErrNeg",
                "PIErrPos",
            ),
        )
        # sort data by a given column
        sortedData = sorted(data, key=lambda d: float(d["time"]))
        for row in sortedData:
            if float(row["time"]) < cutoff:
                # Widths will be our inputs, include intercept
                times.append(float(row["time"]))
                lumins.append(float(row["lumin"]))
                luminErrPoss.append(float(row["luminErrPos"]))
                luminErrNegs.append(float(row["luminErrNeg"]))
    return times, lumins, luminErrPoss, luminErrNegs


def readO(filename, cutoff=1e8):
    times = []
    lumins = []
    luminErrPoss = []
    luminErrNegs = []
    bands = []
    try:
        with open(filename) as fd:
            data = csv.DictReader(
                fd,
                delimiter=",",
                fieldnames=("time", "lumin", "luminErrPos", "luminErrNeg", "band"),
            )
            # sort data by a given column
            sortedData = sorted(data, key=lambda d: float(d["time"]))
            for row in sortedData:
                if float(row["time"]) < cutoff:
                    # Widths will be our inputs, include intercept
                    times.append(float(row["time"]))
                    lumins.append(float(row["lumin"]))
                    luminErrPoss.append(float(row["luminErrPos"]))
                    luminErrNegs.append(float(row["luminErrNeg"]))
                    bands.append(str(row["band"]))
    except:
        times = 0
        lumins = 0
        luminErrPoss = 0
        luminErrNegs = 0
        bands = 0
    finally:
        return times, lumins, luminErrPoss, luminErrNegs, bands


# plot title and labels
# plt.title('GRB '+GRB,fontsize=18)
plt.xlabel("Time (s)", fontsize=18)
plt.ylabel("Luminosity (erg/s)", fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=18)

# plot X-ray light curve (data and spline)
xX, yX, yErrPosX, yErrNegX = readX(filePath, timeCut)
if autoSmoothX:
    smoothX = np.log10(len(yX))
splX = UnivariateSpline(np.log10(xX), np.log10(yX))
splX.set_smoothing_factor(smoothX)
plt.errorbar(
    xX,
    yX,
    yerr=[yErrNegX, yErrPosX],
    fmt=".",
    elinewidth=0.5,
    markersize=8,
    capsize=0,
    label="X-ray: " + "0.3-10 KeV",
)
plt.loglog(
    10 ** np.log10(xX), 10 ** splX(np.log10(xX)), "--", color="steelblue", lw=0.5
)
# plt.loglog(10**np.log10(xX), 10**splX(np.log10(xX)), '-', color='red', lw=1)
axes = plt.gca()
plt.gca().set_xlim([0.8 * min(xX), 1.3 * max(xX)])
plt.legend(loc=1, title="GRB " + GRB)


# plot Optical light curve (data and spline)
xO, yO, yErrPosO, yErrNegO, bands = readO(filePath2, timeCut)
if xO != 0:
    splO = UnivariateSpline(np.log10(xO), np.log10(yO))
    splO.set_smoothing_factor(smoothO)
    plt.loglog(10 ** np.log10(xO), 10 ** splO(np.log10(xO)), "--g", lw=0.5)
    plt.errorbar(
        xO,
        yO,
        yerr=[yErrNegO, yErrPosO],
        fmt="^",
        elinewidth=0.5,
        markersize=8,
        capsize=0,
        label="Optical: " + bands[0] + " band",
    )
    plt.gca().set_xlim([0.8 * min(min(xO), min(xX)), 1.3 * max(max(xO), max(xX))])
    plt.legend(loc=1, title="GRB " + GRB)

# plot vertical line
redshifts, tPeaks, deltaTs, energys = readInfo(flareInfoGRB)
# plt.axvline(44563,linewidth=0.5, linestyle = 'solid', color='r', alpha = 1, label = 'Peak Time: '+ str(44563)+'s')
plt.axvline(
    tPeaks[GRB],
    linewidth=0.5,
    linestyle="solid",
    color="r",
    alpha=1,
    label="Peak Time: "
    + str(tPeaks[GRB])
    + "s"
    + "\n"
    + "Duration: "
    + str(deltaTs[GRB])
    + "s",
)
legend = plt.legend(
    loc=1,
    title="GRB "
    + GRB
    + "\n"
    + "\n"
    + "Redshift:"
    + str(redshifts[GRB])
    + "\n"
    + "Eiso:"
    + str(energys[GRB])
    + "erg",
)
legend.get_title().set_fontsize("14")  # legend 'Title' fontsize
plt.setp(plt.gca().get_legend().get_texts(), fontsize="14")  # legend 'list' fontsize

# output to file
output = GRB + "_LC.pdf"
plt.savefig(output, format="pdf")

if len(sys.argv) == 0:
    plt.show()
plt.show()
