import streamlit as st
from src.model.loading import loadCSVBase
from src.view.index import baseView

baseView()

st.session_state.state = 0

# Bouton d'upload d'un fichier
file = st.file_uploader("Pick a file", type=["csv", "data"])
if file is not None:
    st.session_state.df = loadCSVBase(file)
    st.session_state.initialDf = st.session_state.df.copy()
    st.session_state.state = 1
    st.switch_page("pages/describeChargedData.py")
