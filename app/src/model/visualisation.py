import pandas as pd

def resetStringColumnsDataframe(dataframe: pd.DataFrame, initialDf: pd.DataFrame):
    for columnN in range(len(initialDf.dtypes)):
        if initialDf.dtypes[columnN] == 'object':
            dataframe[dataframe.columns[columnN]] = initialDf[initialDf.columns[columnN]]
    return dataframe
