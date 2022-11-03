import pandas as pd
import numpy as np
import math
sum=0
data=pd.read_csv("deltaz_dtree_under2.csv")
for i in range (87938):
    sum=sum+(data.iloc[i, 0]**2)
sd=math.sqrt(sum/i)
