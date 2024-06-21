
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

def bestNbCluster(dataset):
    datasetStandard = norm.standardisationZScore(dataset)

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
    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel("Nombre de Clusters")
    plt.ylabel("SSE")
    plt.show()

    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    print(f"Nombre optimal de clusters : {kl.elbow}")
    return kl.elbow

def clusteringKMeans(dataset, bestColumns, bestNbCluster): 

    datasetStandard = norm.standardisationZScore(dataset)
    # Application du KMeans avec le nombre optimal de clusters
    kmeans = KMeans(n_clusters=bestNbCluster, random_state=0, n_init='auto')
    kmeans.fit(datasetStandard)

    # Évaluation du clustering avec le score de silhouette
    kmeans_silhouette = silhouette_score(datasetStandard, kmeans.labels_).round(2)
    print(f"Score de silhouette : {kmeans_silhouette}")

    # Séparation des données en ensemble d'entraînement et de test
    X_train = dataset[[bestColumns[0], bestColumns[1]]]

    # Standardisation des données d'entraînement
    X_train_norm = norm.standardisationZScore(X_train)

    # Application du KMeans sur les données d'entraînement
    kmeans.fit(X_train_norm)

    # Visualisation des clusters sur les données d'entraînement
    sns.scatterplot(data = X_train, x = bestColumns[0], y = bestColumns[1], hue = kmeans.labels_)
    plt.show()

def clusteringHierarchique(dataset, bestColumns, bestNbCluster):
    dataset = dataset[[bestColumns[0], bestColumns[1]]]
    datasetStandard = norm.standardisationZScore(dataset)
    ac2 = AgglomerativeClustering(n_clusters = bestNbCluster) 
    plt.figure(figsize =(6, 6)) 
    print(datasetStandard);
    plt.scatter(datasetStandard[0], datasetStandard[1], c = ac2.fit_predict(bestColumns), cmap ='rainbow') 
    plt.show()

    plt.figure(figsize =(8, 8)) 
    plt.title('Visualising the data') 
    Dendrogram = shc.dendrogram((shc.linkage(datasetStandard, method ='ward'))) 
    plt.axhline(y=8, color='r', linestyle='--')
    plt.show() 


def bestApport(dataset):
    if 'Index' in dataset.columns:
        dataset = dataset.drop(columns=['Index'])

    dataset_standard = norm.standardisationZScore(dataset)
    dataset_standard = pd.DataFrame(dataset_standard, columns=dataset.columns)

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

    return top_column_names