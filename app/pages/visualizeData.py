import streamlit as st
from matplotlib import pyplot as plt
from src.view.index import baseView
from src.model.visualisation import resetStringColumnsDataframe
import seaborn as sns

baseView()

plotType = st.radio("What's your favorite movie genre",
                    ["Histograms", "Boxplot", "Lineplot"],
                    )

dataframe = resetStringColumnsDataframe(st.session_state.df.copy(), st.session_state.initialDf)
optionFill = st.multiselect("Select colum to visualize.",
                            list(dataframe),
                            )

# initialDf = st.session_state.initialDf
# for columnN in range(len(initialDf.dtypes)):
#     if initialDf.dtypes[columnN] == 'object':
#         dataframe[dataframe.columns[columnN]] = initialDf[initialDf.columns[columnN]]

if optionFill is not None and len(optionFill) != 0:
    value = optionFill
    if len(optionFill) == 1:
        value = optionFill[0]
    f, axs = plt.subplots(1, 1)
    match plotType:
        case "Histograms":
            sns.histplot(dataframe[value], ax=axs)
            st.pyplot(f)
        case "Boxplot":
            sns.boxplot(dataframe[value], ax=axs)
            st.pyplot(f)
        case "Lineplot":
            sns.lineplot(dataframe[value], ax=axs)
            st.pyplot(f)
