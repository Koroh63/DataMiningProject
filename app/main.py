import streamlit as st
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

if __name__ == "__main__":
    st.session_state.state = 0
    st.switch_page("pages/chargeData.py")
