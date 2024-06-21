
from src.model.load import loadCSVBase
from src.model.clustering import clusteringKMeans, bestApport, bestNbCluster, clusteringHierarchique


if __name__ == "__main__":
    dataset = loadCSVBase('./trees.csv');
    dataset.columns = dataset.columns.str.strip()
    bestColumns = bestApport(dataset)
    bestNbCluster = bestNbCluster(dataset)
    clusteringKMeans(dataset, bestColumns, bestNbCluster)
    #clusteringHierarchique(dataset, bestColumns, bestNbCluster)