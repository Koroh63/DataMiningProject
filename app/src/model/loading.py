import streamlit as st;
import pandas as pd;



def loadCSVBase(file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    """
    @brief Loads a CSV file into a pandas DataFrame.
    
    This function takes a CSV file uploaded via Streamlit's file uploader and 
    loads it into a pandas DataFrame. If no file is provided, it raises a ValueError.
    
    @param file The uploaded CSV file of type UploadedFile loaded by st.file_upload.
    @return DataFrame containing the data from the CSV file.
    
    @exception ValueError Raised if no file is provided.
    """
    if(file==None):
        raise ValueError
    else:
        return pd.read_csv(file, na_values=['null', 'NULL', 'nan', 'NaN', 'NA', 'na', '', 'N/A', 'n/a', '-', '--', 'None', 'none', '?', 'missing', 'MISSING', '#N/A', 'null_value', 'Not Available'])