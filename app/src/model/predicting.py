import pandas
from pandas import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
def checkTypePrediction(df, toPredict: str):
    """
    @brief Determines the type of prediction based on the target column's data type.
    
    This function checks the data type of the target column and decides whether to perform
    logistic regression (for classification) or linear regression (for regression).
    
    @param df The DataFrame containing the data.
    @param toPredict The name of the target column.
    @return The trained model and performance metrics.
    """
    if df[toPredict].dtypes == 'object':
        return "classificationLogistic"  # classificationLogisticRegression(df, toPredict=toPredict)
    else:
        return "regressionLinear"  # regressionLinear(df, toPredict=toPredict)

def fill_missing_values(df: DataFrame, toPredict: str, model):
    """
    @brief Fills missing values in a specified column using a regression model.

    This function separates the DataFrame into rows with and without missing values
    in the specified column, uses the provided regression model to predict the missing values,
    and fills them in.

    @param df The input DataFrame.
    @param toPredict The name of the column to fill.
    @param model The regression model used for prediction.
    @return DataFrame with missing values filled.
    """
    df_filled, df_missing = split_dataframe_on_column(df, toPredict)

    if df_missing.empty:
        return df
    relevant_features = select_highly_correlated_features(df, toPredict)
    X_train = df_filled[relevant_features]
    y_train = df_filled[toPredict]
    X_missing = df_missing[relevant_features]
    
    if not (X_train.columns == X_missing.columns).all():
        raise ValueError("The columns of training data and data with missing values do not match.")
    
    y_missing_pred = model.predict(X_missing)
    
    df.loc[df[toPredict].isna(), toPredict] = y_missing_pred
    
    return df

def select_highly_correlated_features(df: DataFrame, toPredict: str, top_n: int = 2):
    """
    @brief Selects the top N features highly correlated with the target column.
    
    This function calculates the correlation of all columns with the target column and selects
    the top N columns with the highest absolute correlation.
    
    @param df The input DataFrame.
    @param toPredict The name of the target column.
    @param top_n The number of top correlated features to select (default is 2).
    @return List of columns that are highly correlated with the target column.
    """
    df = transformDataAll(df)
    correlation_matrix = df.corr()
    target_correlation = correlation_matrix[toPredict]
    sorted_correlation = target_correlation.abs().sort_values(ascending=False)
    top_features = sorted_correlation.index[sorted_correlation.index != toPredict][:top_n].tolist()
    return top_features


def split_dataframe_on_column(df: DataFrame, column_name: str):
    """
    @brief Splits a DataFrame into two based on whether the specified column has missing values.
    
    This function checks for missing values in the specified column and separates the DataFrame 
    into two: one with rows where the column is filled and one with rows where the column is empty.
    
    @param df The input DataFrame.
    @param column_name The name of the column to check for missing values.
    @return Tuple containing two DataFrames: the first with filled values and the second with empty values.
    
    @exception ValueError Raised if the specified column does not exist in the DataFrame.
    """
    if column_name not in df.columns:
        raise ValueError(f"The column '{column_name}' does not exist in the DataFrame.")
    
    df_filled = df[df[column_name].notna()]
    df_empty = df[df[column_name].isna()]
    
    return df_filled, df_empty

def transformData(df: DataFrame, toPredict: str):
    """
    @brief Transforms categorical columns in the DataFrame to numerical values.
    
    This function replaces categorical values with numerical codes to prepare the DataFrame 
    for machine learning models.
    
    @param df The input DataFrame.
    @param toPredict The name of the target column.
    @return Transformed DataFrame with categorical values replaced by numerical codes.
    """
    for columnN in range(len(df.dtypes)):
        if df.columns[columnN] != toPredict:
            if df.dtypes[columnN] == 'object':
                type_mapping = {type_str: idx + 1 for idx, type_str in enumerate(df[df.columns[columnN]].unique())}
                df[df.columns[columnN]] = df[df.columns[columnN]].replace(type_mapping)
    return df

def transformDataAll(df: DataFrame):
    """
    @brief Transforms categorical columns in the DataFrame to numerical values.
    
    This function replaces categorical values with numerical codes to prepare the DataFrame 
    for machine learning models.
    
    @param df The input DataFrame.
    @return Transformed DataFrame with categorical values replaced by numerical codes.
    """
    for columnN in range(len(df.dtypes)):
        if df.dtypes[columnN] == 'object':
            type_mapping = {type_str: idx + 1 for idx, type_str in enumerate(df[df.columns[columnN]].unique())}
            df[df.columns[columnN]] = df[df.columns[columnN]].replace(type_mapping)
    return df

def separateValuesRegression(df: DataFrame, toPredict: str):
    """
    @brief Separates features and target variable for regression.
    
    This function separates the features (independent variables) and the target variable for regression analysis.
    
    @param df The dataset containing the features and target variable.
    @param toPredict The name of the target column.
    @return Tuple containing the features (X) and target variable (y) for regression analysis.
    """
    dfTmp = df.copy()
    y = dfTmp[dfTmp[toPredict] != 0][toPredict].values
    x = dfTmp[dfTmp[toPredict] != 0].drop([toPredict], axis=1)
    return x, y

def separateValuesClassification(df: DataFrame, toPredict: str):
    """
    @brief Separates features and target variable for classification.
    
    This function separates the features (independent variables) and the binary target variable 
    for classification analysis.
    
    @param df The dataset containing the features and target variable.
    @param toPredict The name of the target column.
    @return Tuple containing the features (X) and binary target variable (y) for classification analysis.
    """
    x = df.drop([toPredict], axis=1)
    y = df[toPredict].values
    return x, y

def initTraining(x, y):
    """
    @brief Initializes training and testing data.
    
    This function splits the dataset into training and testing sets for model evaluation.
    
    @param x The features for training and testing.
    @param y The target variable for training and testing.
    @return Tuple containing the training and testing features and target variable.
    """
    Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=0)
    
    Xtrain = Xtrain.values
    Xtest = Xtest.values

    if len(Xtrain.shape) < 2:
        Xtrain = Xtrain.reshape(-1, 1)
    if len(Xtest.shape) < 2:
        Xtest = Xtest.reshape(-1, 1)

    return Xtrain, Xtest, ytrain, ytest

def visualize_linear_regression(y_true, y_pred):
    """
    @brief Visualizes the results of a linear regression model using matplotlib's plot function.
    
    This function creates a scatter plot of the true values against the predicted values
    and adds a reference line (y=x) to show the ideal prediction.
    
    @param y_true The true values of the target variable.
    @param y_pred The predicted values from the regression model.
    """
    f = plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Linear Regression: True vs Predicted Values')
    return f

def regressionLinear(df: DataFrame, toPredict: str):
    """
    @brief Performs linear regression on the dataset.
    
    This function performs linear regression to predict the target variable and evaluates the model.
    
    @param df The input DataFrame.
    @param toPredict The name of the target column.
    @return The trained linear regression model and its performance metrics (R2 score and Mean Squared Error).
    """
    relevant_features = select_highly_correlated_features(df, toPredict)
    df = df[relevant_features + [toPredict]]
    df = transformData(df, toPredict=toPredict)
    X, Y = separateValuesRegression(df, toPredict)
        
    Xtrain, Xtest, ytrain, ytest = initTraining(X, Y)

    modelLinearRegression = LinearRegression()

    modelLinearRegression.fit(Xtrain, ytrain)

    ypreditLineatRegression = modelLinearRegression.predict(Xtest)

    mse_linear_regression = mean_squared_error(ytest, ypreditLineatRegression)
    r2_linear_regression = r2_score(ytest, ypreditLineatRegression)

    print("-- Linear Regression : ")
    print("Mean Squared Error - Linear Regression:", mse_linear_regression)
    print("R2 Score - Linear Regression:", r2_linear_regression)

    f = visualize_linear_regression(ytest, ypreditLineatRegression)
    
    return modelLinearRegression, r2_linear_regression, mse_linear_regression, f

def visualize_logistic_regression(y_true, y_pred):
    """
    @brief Visualizes the results of a logistic regression model using matplotlib's plot function.
    
    This function creates a scatter plot of the true values against the predicted values
    for the logistic regression model.
    
    @param y_true The true values of the target variable.
    @param y_pred The predicted values from the logistic regression model.
    """
    f = plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_true)), y_true, alpha=0.5, label='True Values')
    plt.scatter(range(len(y_true)), y_pred, alpha=0.5, label='Predicted Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Logistic Regression: True vs Predicted Values')
    plt.legend()
    return f

def classificationLogisticRegression(df: DataFrame, toPredict: str):
    """
    @brief Performs logistic regression on the dataset.
    
    This function performs logistic regression to predict the target variable and evaluates the model.
    
    @param df The input DataFrame.
    @param toPredict The name of the target column.
    @return The trained logistic regression model and its performance metric (accuracy score).
    """
    relevant_features = select_highly_correlated_features(df, toPredict)
    df = df[relevant_features + [toPredict]]
    df = transformData(df, toPredict=toPredict)
    X, Y = separateValuesClassification(df, toPredict=toPredict)

    Xtrain, Xtest, ytrain, ytest = initTraining(X, Y)

    modelLogisticRegression = LogisticRegression(C=5.0, solver='lbfgs', max_iter=10000)

    modelLogisticRegression.fit(Xtrain, ytrain)

    yPreditLogistic = modelLogisticRegression.predict(Xtest)


    print('Logistic Regression Accuracy Score:', accuracy_score(yPreditLogistic, ytest))

    yPreditLogistic = modelLogisticRegression.predict(Xtest)

    f = visualize_logistic_regression(ytest, yPreditLogistic)
    return modelLogisticRegression, accuracy_score(yPreditLogistic, ytest), f
