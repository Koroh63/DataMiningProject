import pandas as pd

def loadCSVBase(file):
    file = pd.read_csv(file);
    return file;
