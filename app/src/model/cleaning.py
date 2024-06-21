import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression

def removeNullRows(df: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Removes rows with any null values from the DataFrame.
    
    This function drops all rows from the DataFrame that contain any null values.
    
    @param df The DataFrame from which null rows are to be removed.
    @return DataFrame with rows containing null values removed.
    """
    return df.dropna()

def fillNullByMean(df: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Fills null values with the mean of the respective columns.
    
    This function replaces all null values in the DataFrame with the mean 
    of their respective columns.
    
    @param df The DataFrame in which null values are to be filled with the mean.
    @return DataFrame with null values filled by mean.
    """
    return df.fillna(df.mean())

def fillNullByMed(df: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Fills null values with the median of the respective columns.
    
    This function replaces all null values in the DataFrame with the median 
    of their respective columns.
    
    @param df The DataFrame in which null values are to be filled with the median.
    @return DataFrame with null values filled by median.
    """
    return df.fillna(df.median())

def fillNullByMode(df: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Fills null values with the mode of the respective columns.
    
    This function replaces all null values in the DataFrame with the mode 
    (most frequent value) of their respective columns.
    
    @param df The DataFrame in which null values are to be filled with the mode.
    @return DataFrame with null values filled by mode.
    """
    return df.fillna(df.mode().iloc[0])

def fillNullByKNN(df: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Fills null values using the K-Nearest Neighbors (KNN) imputation method.
    
    This function uses the K-Nearest Neighbors algorithm to impute missing values 
    in the DataFrame. It replaces null values based on the values of the nearest 
    neighbors.
    
    @param df The DataFrame in which null values are to be filled using KNN imputation.
    @return DataFrame with null values filled by KNN imputation.
    """
    imputer = KNNImputer(n_neighbors=3)
    imputed = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(imputed, columns=df.columns)
    return df_imputed

def fillNullByRegression(df: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Fills null values using regression-based imputation.
    
    This function uses regression techniques to predict and fill missing values 
    in the DataFrame. For each column with null values, it uses the remaining 
    columns as features to train a linear regression model and predict the 
    missing values.
    
    @param df The DataFrame in which null values are to be filled using regression.
    @return DataFrame with null values filled by regression.
    """
    df_filled = df.copy()

    for column in df_filled.columns:
        if df_filled[column].isnull().sum() > 0:
            # Select features by excluding the target column
            features = df_filled.loc[:, df_filled.columns != column]

            # Get indices for rows with and without null values in the target column
            train_idx = df_filled[column].dropna().index
            test_idx = df_filled[column].index.difference(train_idx)

            # Initialize and train the regression model
            reg = LinearRegression()
            reg.fit(features.loc[train_idx], df_filled.loc[train_idx, column])

            # Predict the null values and fill them in the DataFrame
            predicted_values = reg.predict(features.loc[test_idx])
            df_filled.loc[test_idx, column] = predicted_values

    return df_filled


