import commonFunctions
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH ="datasets/housing"
HOUSING_tgz = "housing.tgz"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/" + HOUSING_tgz

if not os.path.isdir(HOUSING_PATH):
    commonFunctions.fetchData(HOUSING_URL,HOUSING_PATH,HOUSING_tgz)


housing = commonFunctions.loadData(HOUSING_PATH, "housing.csv")
def displayData():
    housing.hist(bins=50, figsize=(20,15))
    plt.show()

# New uid
housing_with_id = housing.reset_index()
housing_with_id["id"] = housing["longitude"]*1000 + housing["latitude"]

# train : test = 8:2 using sklearn
train_set, test_set = train_test_split(housing_with_id,test_size=0.2,random_state= 40)

# Pearson's r
corr_matrix = housing.corr()

# using Pandas' scatter_matrix
def Pandas_scatter_matrix():
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12,8))
    plt.show()

Pandas_scatter_matrix()