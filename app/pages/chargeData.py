import streamlit as st
from src.model.loading import loadCSVBase
from src.view.index import baseView

baseView()

st.session_state.state = 0
file = st.file_uploader("Pick a file")
if file is not None:
    st.session_state.df = loadCSVBase(file)
    st.session_state.state = 1
    st.switch_page("pages/describeChargedData.py")
