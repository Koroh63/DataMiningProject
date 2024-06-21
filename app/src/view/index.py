import streamlit as st
# import sys
#
# sys.path.append("src/view/")

def baseView():
    if "state" not in st.session_state:
        st.switch_page("main.py")

    path = "pages/"
    st.sidebar.page_link(path + "chargeData.py", label="Charge data")
    st.sidebar.page_link(path + "describeChargedData.py", label="Describe charge data", disabled=st.session_state.state < 1)
    st.sidebar.page_link(path + "cleaningProcessingData.py", label="Chose fill missing value method", disabled=st.session_state.state < 1)
    st.sidebar.page_link(path + "visualizeData.py", label="Visualize data", disabled=st.session_state.state < 2)

