import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

def main():

# Data Exploration
#-----------------

    # Importing Data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # Set plot parameters
    plt.style.use(style='ggplot')
    plt.rcParams['figure.figsize'] = (5, 3)

    # Investigating response distribution
    plt.hist(train.SalePrice, color='blue')
    plt.show()

        # NOTE: Response variable is skewed
    print(train.SalePrice.skew(),'\n')

        # Adjust for Skew
    response = np.log(train.SalePrice)

        # Check
    print(train.SalePrice.skew(),'\n')

# Feature Selection
#------------------

    # Quantitative Variables
    numeric_features = train.select_dtypes(include = (np.number))
    corr = numeric_features.corr()

        # Investigating the most positive and most negative correlated variables
    print(corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
    print(corr['SalePrice'].sort_values(ascending=False)[-5:], '\n')

main()