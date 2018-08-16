import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

def main():

#-----------------
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

#------------------
# Feature Selection
#------------------

    # Quantitative Variables
    #-----------------------

    numeric_features = train.select_dtypes(include = (np.number))
    corr = numeric_features.corr()


        # Investigating the most positive and most negative correlated variables

    print(corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
    print(corr['SalePrice'].sort_values(ascending=False)[-5:], '\n')


        # Visualizing the Positive Correlations

    print("Overall Quality: \n", train.OverallQual.unique(), "\n")
    print("Above Ground Living Area (ft-sq): \n", train.GrLivArea.unique(), "\n")
    print("No. of Cars in Garage: \n", train.GarageCars.unique(), "\n")
    print("Garage Area (sq-ft): \n", train.GarageArea.unique(), "\n")

    quality_pivot = train.pivot_table(index='OverallQual',
                                   values='SalePrice',aggfunc=np.median)
    quality_pivot.plot(kind='bar', color='blue')
    plt.xlabel('Overall Quality')
    plt.ylabel('Median Sale Price')
    plt.show()
    # NOTE: Outliers @ 4000+

    livArea = plt.scatter(x=train['GrLivArea'],y=response)
    plt.xlabel('Above Ground Living Area (ft^2)')
    plt.ylabel('Median Sale Price')
    plt.show()

    cars_pivot = train.pivot_table(index='GarageCars',
                                   values='SalePrice',aggfunc=np.median)
    cars_pivot.plot(kind='bar', color='blue')
    plt.xlabel('Overall Quality')
    plt.ylabel('Median Sale Price')
    plt.show()

    garageArea = plt.scatter(x=train['GarageArea'],y=response)
    plt.xlabel('Garage Area (ft^2)')
    plt.ylabel('Median Sale Price')
    plt.show()
    # NOTE: Outliers @ 1200+


        # Visualizing the Negative Correlations

    print('Year Sold: \n', train.YrSold.unique(), '\n')
    print('Overall Condition: \n', train.OverallCond.unique(), '\n')
    print('Building Class: \n', train.MSSubClass.unique(), '\n')
    print('Enclosed Porch: \n', train.EnclosedPorch.unique(), '\n')
    print('Above Ground Kitchen: \n', train.KitchenAbvGr.unique(), '\n')

    year_pivot = train.pivot_table(index='YrSold',
                                    values='SalePrice',aggfunc=np.median)
    year_pivot.plot(kind='bar', color='blue')
    plt.xlabel('Year Sold')
    plt.ylabel('Median Sale Price')
    plt.show()

    cond_pivot = train.pivot_table(index='OverallCond',
                                    values='SalePrice',aggfunc=np.median)
    cond_pivot.plot(kind='bar', color='blue')
    plt.xlabel('Overall Cond')
    plt.ylabel('Median Sale Price')
    plt.show()

    bldg_pivot = train.pivot_table(index='MSSubClass',
                                    values='SalePrice',aggfunc=np.median)
    bldg_pivot.plot(kind='bar', color='blue')
    plt.xlabel('Building Class')
    plt.ylabel('Median Sale Price')
    plt.show()

    porch_plot = plt.scatter(x=train['EnclosedPorch'],y=response)
    plt.xlabel('Enclosed Porch Area (ft^2)')
    plt.ylabel('Median Sale Price')
    plt.show()
    # NOTE: Outliers @ 400+

    ktch_pivot = train.pivot_table(index='KitchenAbvGr',
                                    values='SalePrice',aggfunc=np.median)
    ktch_pivot.plot(kind='bar', color='blue')
    plt.xlabel('Kitchen Above Ground(?)')
    plt.ylabel('Median Sale Price')
    plt.show()


        # Removing Outliers

    livArea = plt.scatter(x=train['GrLivArea'],y=response)
    plt.xlabel('Above Ground Living Area (ft^2)')
    plt.ylabel('Median Sale Price')
    plt.title('LIVING AREA')
    plt.show()

    train = train[train['GrLivArea'] < 4000]
    response = np.log(train.SalePrice)
    livArea = plt.scatter(x=train['GrLivArea'],y=response)
    plt.xlabel('Above Ground Living Area (ft^2)')
    plt.ylabel('Median Sale Price')
    plt.title('Outliers Removed')
    plt.show()

    garageArea = plt.scatter(x=train['GarageArea'],y=response)
    plt.xlabel('Garage Area (ft^2)')
    plt.ylabel('Median Sale Price')
    plt.title('GARAGE AREA')
    plt.show()

    train = train[train['GarageArea'] < 1200]
    response = np.log(train.SalePrice)
    garageArea = plt.scatter(x=train['GarageArea'],y=response)
    plt.xlabel('Garage Area (ft^2)')
    plt.ylabel('Median Sale Price')
    plt.title('Outliers Removed')
    plt.show()

    porch_plot = plt.scatter(x=train['EnclosedPorch'],y=response)
    plt.xlabel('Enclosed Porch Area (ft^2)')
    plt.ylabel('Median Sale Price')
    plt.title('ENCLOSED PORCH')
    plt.show()

    train = train[train['EnclosedPorch'] < 400]
    response = np.log(train.SalePrice)
    porch_plot = plt.scatter(x=train['EnclosedPorch'],y=response)
    plt.xlabel('Enclosed Porch Area (ft^2)')
    plt.ylabel('Median Sale Price')
    plt.title('Outliers Removed')
    plt.show()

main()