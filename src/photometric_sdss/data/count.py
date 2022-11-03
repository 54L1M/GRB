import pandas as pd
dfIn = pd.read_csv("sdss_galaxy_450000.csv")

        
dfOut = dfIn.drop(dfIn[dfIn['redshift'] > 2].index)
# j=0
# gg=dfOut['redshift']
# for i in gg:
#     if i>=2.25:
#         j+=1

outputFilePath = 'output_under2.csv'
    
dfOut.to_csv(outputFilePath, index=False)
       