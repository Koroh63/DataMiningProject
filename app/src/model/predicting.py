import pandas as pd;
from sklearn.linear_model import LinearRegression

def separateValuesRegression(df : DataFrame, toPredict : str):
    """
    Function to separate features and target variable for regression.

    Separates the features (independent variables) and the target variable (Total Deaths) for regression analysis.

    Args:
    df (pandas.DataFrame): The dataset containing the features and target variable.
    toPredict : The Name of the column to predict as str

    Returns:
    tuple: A tuple containing the features (X) and target variable (y) for regression analysis.
    """
    # Make a copy of the dataset to avoid modification of the original dataset
    dfTmp = df.copy()
    # Extract target variable (Total Deaths) where Total Deaths is not equal to 0
    y = dfTmp[dfTmp[toPredict] != 0][toPredict].values
    # Extract features (independent variables) excluding 'TotaltoPredict : The Name of the column to predict as str Deaths', 'Total Affected', 'No Affected', and 'No Injured' columns
    x = dfTmp[dfTmp[toPredict] != 0].drop([toPredict], axis=1)
    return x,y


def separateValuesClassification(df : DataFrame, toPredict : str):
    """
    Function to separate features and target variable for classification.

    Separates the features (independent variables) and the binary target variable (Lethality) for classification analysis.

    Args:
    df (pandas.DataFrame): The dataset containing the features and target variable.
    toPredict : The Name of the column to predict as str

    Returns:
    tuple: A tuple containing the features (X) and binary target variable (y) for classification analysis.
    """
    # Extract features (independent variables) excluding 'Total Deaths', 'Total Affected', 'No Affected', 'No Injured', and 'Lethality' columns
    x = df.drop([toPredict],axis=1)
    # Extract binary target variable 'Lethality'
    y = df[toPredict].values
    return x,y

def initTraining(x,y):
    """
    Function to initialize training and testing data.

    Splits the dataset into training and testing sets for model evaluation.

    Args:
    x (numpy.ndarray): The features (independent variables) for training and testing.
    y (numpy.ndarray): The target variable for training and testing.

    Returns:
    tuple: A tuple containing the training and testing features and target variable.
    """
    # Split dataset into training and testing sets with 75% for training and 25% for testing
    Xtrain, Xtest, ytrain, ytest = train_test_split(x, y,test_size=0.25, random_state=0)
    
    # Convert training and testing features to numpy arrays
    Xtrain = Xtrain.values
    Xtest = Xtest.values

    # Ensure feature arrays have 2 dimensions
    if len(Xtrain.shape) < 2:
        Xtrain = Xtrain.reshape(-1, 1)
    if len(Xtest.shape) < 2:
        Xtest = Xtest.reshape(-1, 1)

    return Xtrain,Xtest,ytrain,ytest

def regressionLinear(df : DataFrame){
    # Separate features and target variable for regression
    X,Y = separateValuesRegression(df)
        
    # Initialize training and testing data
    Xtrain,Xtest,ytrain,ytest = initTraining(X,Y)

    # Initialize regression models
    modelLinearRegression = LinearRegression()

    # Fit regression models
    modelLinearRegression.fit(Xtrain,ytrain)

    # Predictions
    ypreditLineatRegression = modelLinearRegression.predict(Xtest)

    # Calculate Mean Squared Error and R2 Score
    mse_linear_regression = mean_squared_error(ytest, ypreditLineatRegression)

    r2_linear_regression = r2_score(ytest, ypreditLineatRegression)

    # Print results
        print( "-- Linear Regression : ")
        print("Mean Squared Error - Linear Regression:", mse_linear_regression)
        print("R2 Score - Linear Regression:", r2_linear_regression)

    return modelLinearRegression, r2_linear_regression, mse_linear_regression

}