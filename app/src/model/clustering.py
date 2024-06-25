
import numpy as np
from src.model import norm_stand as norm
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
import scipy.cluster.hierarchy as shc
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

def bestApport(dataset_standard):

    pca = PCA()
    pca.fit(dataset_standard)

    # Récupérer les noms des colonnes après standardisation
    columns_after_drop = dataset_standard.columns

    # Variance expliquée par chaque composant principal
    explained_variance = pca.explained_variance_ratio_

    sorted_indices = sorted(range(len(explained_variance)), key=lambda k: explained_variance[k], reverse=True)

    # Récupérer les noms des deux colonnes avec les variances les plus élevées
    top_column_names = [columns_after_drop[idx] for idx in sorted_indices[:2]]

    print("Variance expliquée par chaque composant principal :")
    for col, var in zip(columns_after_drop, explained_variance):
        print(f"{col}: {var}")

    print(f"\nLes deux colonnes avec les variances les plus élevées sont : {top_column_names}")
    print("Représentation de la variance pour les clusters : ", explained_variance[0] + explained_variance[1])

    return top_column_names, explained_variance

def bestNbCluster(datasetStandard):

    # Paramètres du modèle KMeans
    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42
    }

    # Liste pour stocker les sommes des carrés des erreurs (SSE)
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(datasetStandard)
        sse.append(kmeans.inertia_)

    # Utilisation de la méthode du coude pour trouver le nombre optimal de clusters
    f = plt.figure()
    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel("Nombre de Clusters")
    plt.ylabel("SSE")
    plt.title("Elbow Method")
    # plt.show()

    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    print(f"Nombre optimal de clusters : {kl.elbow}")
    return kl.elbow, f

def bestNbClusterHiAsc(datasetStandard):
    # Calculer les distances entre chaque point
    dist_matrix = pdist(datasetStandard)

    # Clustering hiérarchique ascendant
    Z = linkage(dist_matrix, method='ward')

    # Extraire les distances de fusion des 10 dernières fusions
    # La colonne 2 de la matrice de liaison (Z) contient les distances de fusion
    last = Z[-10:, 2]

    # Inverser l'ordre des distances pour avoir les plus grandes distances en premier
    last_rev = last[::-1]

    # Générer une séquence d'indices correspondant au nombre de clusters
    # Ici, nous avons 10 dernières distances de fusion, donc les indices seront de 1 à 10
    idxs = np.arange(1, len(last) + 1)

    # Utilisation de la méthode du coude pour trouver le nombre optimal de clusters
    f = plt.figure()
    plt.plot(idxs, last_rev)
    plt.title("Méthode du coude pour déterminer le nombre optimal de clusters")
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Distance de fusion")
    plt.show()

    kl = KneeLocator(idxs, last_rev, curve="convex", direction="decreasing")
    optimal_clusters = kl.elbow

    return optimal_clusters, f

def buildBestDataSetRep(datasetStandard, bestColumns):
    pca = PCA()
    pca.fit(datasetStandard)
    pca = PCA(n_components = 2) 
    X_principal = pca.fit_transform(datasetStandard) 
    X_principal = pd.DataFrame(X_principal) 
    X_principal.columns = [bestColumns[0], bestColumns[1]] 
    return X_principal

def clusteringKMeans(datasetStandard, bestNbCluster):
    # Application du KMeans avec le nombre de clusters
    kmeans = KMeans(n_clusters=bestNbCluster, random_state=0, n_init='auto')
    kmeans.fit(datasetStandard)

    # Évaluation du clustering avec le score de silhouette
    kmeans_silhouette = silhouette_score(datasetStandard, kmeans.labels_).round(2)
    yield f"Score de silhouette : {kmeans_silhouette}  \n"

    # calcul des coordonnées des centres des clusters
    yield f"Centre des clusters : {kmeans.cluster_centers_}  \n"

    # Calcul du nombre de points par cluster
    counts = np.bincount(kmeans.labels_)
    for cluster_idx, count in enumerate(counts):
        yield f"Cluster {cluster_idx}: {count} points  \n"

    # Visualisation des clusters sur les données
    f = plt.figure()
    sns.scatterplot(data = datasetStandard, x = (datasetStandard.columns)[0], y = (datasetStandard.columns)[1], hue = kmeans.labels_)
    plt.xlabel((datasetStandard.columns)[0])
    plt.ylabel((datasetStandard.columns)[1])
    plt.title('Clustering KMeans')
    # plt.show()
    yield f

def clusteringHierarchique(datasetStandard, bestColumns, bestNbCluster):
    # Application du AgglomerativeClustering avec le nombre de clusters
    ac2 = AgglomerativeClustering(n_clusters=bestNbCluster) 
    labels = ac2.fit_predict(datasetStandard)

    # Calcul du score de silhouette
    silhouette_avg = silhouette_score(datasetStandard, labels).round(2)
    yield f"Score de silhouette : {silhouette_avg}  \n"

    # Calcul du nombre de points pour chaque cluster
    counts = np.bincount(labels)
    for cluster_idx, count in enumerate(counts):
        yield f"Cluster {cluster_idx}: {count} points  \n"

    # Calcul des centres de gravité des clusters
    centers = pd.DataFrame(datasetStandard).groupby(labels).mean()
    yield f"Centre des clusters : {centers}  \n"

    # Visualisation des clusters sur les données
    f = plt.figure(figsize=(6, 6))
    sns.scatterplot(data = datasetStandard, x = (datasetStandard.columns)[0], y = (datasetStandard.columns)[1], hue = labels)
    # plt.scatter(datasetStandard[bestColumns[0]], datasetStandard[bestColumns[1]], c=labels, cmap='rainbow')
    plt.xlabel(bestColumns[0])
    plt.ylabel(bestColumns[1])
    plt.title('Clustering Hierarchique')
    yield f


def dendrogramme(datasetStandard):
    # Affichage du dendrogramme du dataset
    f = plt.figure(figsize =(8, 8))
    plt.title('Visualising the data') 
    Dendrogram = shc.dendrogram((shc.linkage(datasetStandard, method ='ward')))
    plt.title('Dendrogramme du dataset')
    return f
