from src.model.load import loadCSVBase
from src.model import norm_stand as norm
from src.view.index import baseView
from src.model.clustering import clusteringKMeans, bestApport, bestNbCluster, clusteringHierarchique

if __name__ == "__main__":
    baseView()