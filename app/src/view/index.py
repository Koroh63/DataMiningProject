import streamlit as st
# import sys
#
# sys.path.append("src/view/")

def baseView():
    if "state" not in st.session_state:
        st.switch_page("main.py")

    st.set_page_config(
        page_title="BaBaMaCo",
    )

    path = "pages/"
    st.sidebar.write("## Pre-processing")
    st.sidebar.page_link(path + "chargeData.py", label="Charge data")
    st.sidebar.page_link(path + "describeChargedData.py", label="Describe charge data", disabled=st.session_state.state < 1)
    st.sidebar.page_link(path + "cleaningProcessingData.py", label="Chose fill missing value method", disabled=st.session_state.state < 1)
    st.sidebar.page_link(path + "visualizeData.py", label="Visualize data", disabled=st.session_state.state < 2)
    st.sidebar.write("## Clustering")
    st.sidebar.page_link(path + "clusteringData.py", label="Clustering  data", disabled=st.session_state.state < 2)
    st.sidebar.write("## Predicting")
    st.sidebar.page_link(path + "predictingData.py", label="Predicting columns", disabled=st.session_state.state < 2)
