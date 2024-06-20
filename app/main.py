
from src.model.load import loadCSVBase
from src.model.clustering import clustering, apport


if __name__ == "__main__":
    dataset = loadCSVBase('./trees.csv');
    dataset.columns = dataset.columns.str.strip()
    bestColumns = apport(dataset)
    clustering(dataset, bestColumns)