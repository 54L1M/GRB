import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

z = []
z_spec = []
znorm = []

spec = pd.read_csv("../dtree/targets_test.csv")
photo = pd.read_csv("../dtree/y_pred.csv")

for i in range(87000):
    z.append(spec.iloc[i, 0] - photo.iloc[i, 0])
    z_spec.append(spec.iloc[i, 0])
    znorm.append(z[i] / (z_spec[i] + 1))
deltazaverage = np.mean(znorm)
print(deltazaverage)
dfOut = pd.DataFrame(znorm)
dfOut.to_csv("znormdata.csv", index=False)


size, scale = 1000, 10
df = pd.read_excel("znormdata.xlsx")

# _,ax,_=plt.hist(df['znormdata'], bins=200, rwidth=0.9,
#                    color='#607c8e')

plt.xlim(-0.5, 0.5)
list1 = np.arange(-1, 1, 87000)
width = 1 / 1.5
plt.title("\u0394 z distribution")
plt.xlabel("\u0394 z")
plt.ylabel("Number")
# ax.set_yscale('log')
# plt.grid(axis='y', alpha=0.75)
ax = sb.histplot(df["znormdata"], bins=400, color="cyan")
# ax.lines[0].set_color('crimson')
plt.savefig("deltaznormunder2", dpi=1200)
plt.show()
"""mu, sigma = norm.fit(df['znormdata'])
best_fit_line = norm.pdf(ax, mu, sigma)
plt.plot(ax, best_fit_line)
plt.savefig("deltaznorm", dpi=1200)
plt.show()"""
# gausaisn plot
# fit bar plot data using curve_fit
# def func(x, a, b, c):
#     # a Gaussian distribution
#     return a * np.exp(-(x-b)**2/(2*c**2))

# popt, pcov = curve_fit(func, list1, znorm)

# x = np.linspace(-0.5, 0.5, 100)
# y = func(x, *popt)

# plt.plot(x , y, c='g')
# plt.savefig("deltaznorm", dpi=1200)
# plt.show()
