import streamlit as st

def baseView():
    st.title("Simple Streamlit App")
    st.header("Welcome to my app")
    st.text("This is a simple Streamlit app.")
    
    st.subheader("User Input")
    name = st.text_input("Enter your name")
    age = st.number_input("Enter your age", min_value=0, max_value=120, step=1)
    
    if st.button("Submit"):
        st.write(f"Hello, {name}! You are {age} years old.")
    
    st.subheader("Data Display")
    st.write("Here you can display data, charts, and more.")
