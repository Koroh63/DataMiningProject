import streamlit as st
import pandas as pd

from src.view.index import baseView
from src.model.cleaning import *
from src.model.norm_stand import *

baseView()

optionsIgnoredColumns = st.multiselect("Select colum to ignore/delete.",
                                   list(st.session_state.df),
                                   )

optionFill = st.selectbox("Fill missing value method.",
                          ["Remove", "Mean", "Median", "Mode", "KNN", "Regression"],
                          index=None,
                          placeholder="Select fill missing value method...",
                          )

optionNorm = st.selectbox("Choose normalization/standardisation method.",
                          ["Nothing", "MinMax", "MaxAbs", "Robust", "ZScore"],
                          index=None,
                          placeholder="Select normalize data method...",
                          )

if st.button("Normalize data", disabled=(optionNorm is None or optionFill is None)):
    st.session_state.df = ignoreColumns(st.session_state.df, optionsIgnoredColumns)
    print(optionsIgnoredColumns)
    st.session_state.df = transformData(st.session_state.df)
    match optionFill:
        case "Remove":
            st.session_state.df = removeNullRows(st.session_state.df)
            st.session_state.initialDf = removeNullRows(st.session_state.initialDf)
        case "Mean":
            st.session_state.df = fillNullByMean(st.session_state.df)
        case "Median":
            st.session_state.df = fillNullByMed(st.session_state.df)
            st.session_state.initialDf = removeNullRows(st.session_state.initialDf)
        case "Mode":
            st.session_state.df = fillNullByMode(st.session_state.df)
            st.session_state.initialDf = removeNullRows(st.session_state.initialDf)
        case "KNN":
            st.session_state.df = fillNullByKNN(st.session_state.df)
            st.session_state.initialDf = removeNullRows(st.session_state.initialDf)
        case "Regression":
            st.session_state.df = fillNullByRegression(st.session_state.df)

    colums = st.session_state.df.columns
    match optionNorm:
        case "MinMax":
            st.session_state.df = normalizeMinMax(st.session_state.df)
        case "MaxAbs":
            st.session_state.df = normalizeMaxAbs(st.session_state.df)
        case "Robust":
            st.session_state.df = normalizeRobust(st.session_state.df)
        case "ZScore":
            st.session_state.df = standardisationZScore(st.session_state.df)

    st.session_state.df = pd.DataFrame(st.session_state.df, columns=colums)

    st.session_state.state = 2
    st.switch_page("pages/visualizeData.py")
