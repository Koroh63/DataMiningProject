# import pandas_profiling
import streamlit as st
import pandas as pd
from src.view.index import baseView

# pip install streamlit-pandas-profiling
# import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
# pip install ydata-profiling
from ydata_profiling import ProfileReport

baseView()

# Affichage de la dataframe
st.title("Datas loaded")
st.dataframe(st.session_state.df)

# Affichage des résumés statistique de la DataFrame
pr = ProfileReport(st.session_state.df)
st_profile_report(pr)
