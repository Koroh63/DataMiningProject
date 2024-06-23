import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.view.index import baseView
from src.model.predicting import checkTypePrediction,split_dataframe_on_column,fill_missing_values,select_highly_correlated_features,classificationLogisticRegression,regressionLinear,visualize_linear_regression

baseView()

# df = loadCSVBase("assets/trees.csv")
# df2 = loadCSVBase ("assets/palmerPinguin.csv")
dataset = st.session_state.df
initialDataset = st.session_state.initialDf


optionCol = st.selectbox("Select column to predict.",
                         list(dataset),
                         index=None,
                         placeholder="Select column to predict...",
                         )

if optionCol is not None:
    dataset[optionCol] = initialDataset[optionCol]
    st.dataframe(dataset)
    dfFull, dfEmpty = split_dataframe_on_column(dataset, optionCol)

    recommendedPredict = checkTypePrediction(dfFull, optionCol)
    st.write(f"The recommended prediction is : {recommendedPredict}")
    precdictChoice = ["classificationLogistic", "regressionLinear"]
    optionPredict = st.selectbox("Select predicting methode.",
                                 precdictChoice,
                                 index=precdictChoice.index(recommendedPredict) ,
                                 placeholder="Select predicting method...",
                                 )

    if optionPredict is not None:
        model = None
        match optionPredict:
            case "classificationLogistic":
                model, a, f = classificationLogisticRegression(dfFull, toPredict=optionCol)
                st.write(f"Accuracy score : {a}  \n")
                st.pyplot(f)
            case "regressionLinear":
                model, a, b, f = regressionLinear(dfFull, toPredict=optionCol)
                st.write(f"R2 score : {a}  \n")
                st.write(f"Mean Squared Error : {b}  \n")
                st.pyplot(f)

        corr = st.session_state.df.corr()

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
        st.write("Heatmap :")
        st.pyplot(f)

        if model is not None:
            st.write("Fill Missing values :")
            dfFilled = fill_missing_values(dataset, optionCol, model)
            st.dataframe(dfFilled)
            columns = select_highly_correlated_features(dfFilled, optionCol)
            st.write(f"List of columns that are highly correlated with the target column are {columns}")
            st.download_button(label="Download predicted file", data=dfFilled.to_csv(index=False).encode('utf-8'),
                               file_name='predictedData.csv', mime='text/csv')
