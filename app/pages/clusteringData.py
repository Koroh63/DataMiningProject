import streamlit as st
from src.model.clustering import *
from src.view.index import baseView

baseView()

datasetStandard = st.session_state.df
bestColumns, explained_variance = bestApport(datasetStandard)  # choix des 2 meilleurs colonnes qui représentent au mieux la variance

st.write(f"Les deux colonnes avec les variances les plus élevées sont : {bestColumns}")
st.write("Représentation de la variance pour les clusters : ", explained_variance[0] + explained_variance[1])

# st.write("Best columns are '" + bestColumns[0] + "' and '" + bestColumns[1] + "'")
optionFill = st.multiselect("Select 2 columns to get best cluster number.",
                            list(datasetStandard),
                            default=bestColumns,
                            max_selections=2
                            )
if len(optionFill) == 2:

    X_principal = buildBestDataSetRep(datasetStandard, optionFill)  # création du dataset avec seultement les deux colonnes
    bestNbCluster, f = bestNbCluster(X_principal)  # choix du meilleur nombrede cluster
    st.pyplot(f)
    st.write("Best cluster number is ", bestNbCluster)
    number = st.number_input("Choose a number of cluster", value=bestNbCluster, placeholder="Type a number...", min_value=2, step=1)

    optionClustering = st.selectbox("Select clustering method.",
                                    ["KMeans", "Hierarchic"],
                                    index=None,
                                    placeholder="Select clustering method...",
                                    )
    if optionClustering is not None:
        match optionClustering:
            case "KMeans":
                st.write_stream(clusteringKMeans(X_principal, number))  # cluster k-mean
            case "Hierarchic":
                st.write_stream(clusteringHierarchique(X_principal, optionFill, number))  # cluster hierarchique

        f = dendrogramme(X_principal)
        st.pyplot(f)
