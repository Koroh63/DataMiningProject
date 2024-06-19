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

st.title("Datas loaded")
st.dataframe(st.session_state.df)
pr = ProfileReport(st.session_state.df)
st_profile_report(pr)
