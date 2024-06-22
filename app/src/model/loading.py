
import pandas as pd

def remove_outer_spaces_and_quotes(s: str) -> str:
    """
    @brief Removes outer spaces and quotes from a string.
    
    This function takes a string, removes leading and trailing spaces, 
    removes all double quotes, and ensures that only single spaces 
    separate words within the string.
    
    @param s The input string to be cleaned.
    @return A cleaned string with no leading/trailing spaces or quotes.
    
    @exception None
    """
    if isinstance(s, str):
        s = s.strip()  # Remove leading and trailing spaces
        s = s.replace('"', '')  # Remove all quotes
        return ' '.join(s.split())  # Remove multiple spaces
    return s

def loadCSVBase(file) -> pd.DataFrame:
    """
    @brief Loads a CSV file into a pandas DataFrame.
    
    This function takes a CSV file uploaded via Streamlit's file uploader and 
    loads it into a pandas DataFrame. If no file is provided, it raises a ValueError.
    
    @param file The uploaded CSV file of type UploadedFile loaded by st.file_upload.
    @return DataFrame containing the data from the CSV file.
    
    @exception ValueError Raised if no file is provided.
    """
    if file is None:
        raise ValueError("No file provided")
    else:
        df = pd.read_csv(file, na_values=['null', 'NULL', 'nan', 'NaN', 'NA', 'na', '', 'N/A', 'n/a', '-', '--', 'None', 'none', '?', 'missing', 'MISSING', '#N/A', 'null_value', 'Not Available'])
        
        
        # Clean column names
        
        df.columns = [remove_outer_spaces_and_quotes(col) for col in df.columns]
        
        # Remove outer spaces and quotes from all cells
        df = df.applymap(remove_outer_spaces_and_quotes)
        
        df = df.drop(columns=[col for col in df.columns if col.lower() in ['index', 'id']], errors='ignore')
        
        return df
