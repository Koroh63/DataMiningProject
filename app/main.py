import pandas as pd
from sklearn.decomposition import PCA
from src.model import norm_stand as norm
from src.model import loading
from src.model.clustering import clusteringKMeans, bestApport, bestNbClusterKmeans, bestNbClusterHiAsc, clusteringHierarchique, buildBestDataSetRep, dendrogramme

if __name__ == "__main__":
    dataset = loading.loadCSVBase('./trees.csv');

    # Nettoyage des données
    dataset.columns = dataset.columns.str.strip()
    if 'Index' in dataset.columns:
        dataset = dataset.drop(columns=['Index'])

    datasetStandard = norm.standardisationZScore(dataset) # choix de la stardardisation 
    bestColumns = bestApport(dataset, datasetStandard) # choix des 2 meilleurs colonnes qui représentent au mieux la variance
    X_principal = buildBestDataSetRep(datasetStandard, bestColumns) # création du dataset avec seultement les deux colonnes 
    bestNbCluster = bestNbClusterKmeans(X_principal) # choix du meilleur nombrede cluster
    bestNbClusterHiAsc = bestNbClusterHiAsc(X_principal)
    clusteringKMeans(X_principal, bestNbCluster) # cluster k-mean
    clusteringHierarchique(X_principal, bestColumns, bestNbClusterHiAsc) # cluster hierarchique

    dendrogramme(X_principal)