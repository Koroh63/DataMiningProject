
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


def bestApport(dataset, dataset_standard):

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
    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel("Nombre de Clusters")
    plt.ylabel("SSE")
    plt.show()

    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    print(f"Nombre optimal de clusters : {kl.elbow}")
    return kl.elbow

def buildBestDataSetRep(datasetStandard, bestColumns):
    pca = PCA()
    pca.fit(datasetStandard)
    pca = PCA(n_components = 2) 
    X_principal = pca.fit_transform(datasetStandard) 
    X_principal = pd.DataFrame(X_principal) 
    X_principal.columns = [bestColumns[0], bestColumns[1]] 
    return X_principal

def clusteringKMeans(datasetStandard, bestNbCluster): 
    # Application du KMeans avec le nombre optimal de clusters
    kmeans = KMeans(n_clusters=bestNbCluster, random_state=0, n_init='auto')
    kmeans.fit(datasetStandard)

    # Évaluation du clustering avec le score de silhouette
    kmeans_silhouette = silhouette_score(datasetStandard, kmeans.labels_).round(2)
    print(f"Score de silhouette : {kmeans_silhouette}")

    # Application du KMeans sur les données d'entraînement
    kmeans.fit(datasetStandard)

    # Visualisation des clusters sur les données d'entraînement
    sns.scatterplot(data = datasetStandard, x = (datasetStandard.columns)[0], y = (datasetStandard.columns)[1], hue = kmeans.labels_)
    plt.show()

def clusteringHierarchique(datasetStandard, bestColumns, bestNbCluster):
    ac2 = AgglomerativeClustering(n_clusters = bestNbCluster) 
    plt.figure(figsize =(6, 6))
    plt.scatter(datasetStandard[bestColumns[0]], datasetStandard[bestColumns[1]], c = ac2.fit_predict(datasetStandard), cmap ='rainbow') 
    plt.show()

    plt.figure(figsize =(8, 8)) 
    plt.title('Visualising the data') 
    Dendrogram = shc.dendrogram((shc.linkage(datasetStandard, method ='ward')))
    plt.show() 