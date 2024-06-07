from src.model.load import loadCSVBase
from src.model import norm_stand as norm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #baseView()
    dataset = loadCSVBase('./trees.csv');
    for ds in dataset:
        plt.hist(dataset[ds])
        plt.title("répartition des données de : " + ds)
        plt.show()