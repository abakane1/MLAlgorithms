import os
import numpy as np

np.random.seed(42)

import matplotlib
import matplotlib.pyplot as plt

import pandas as pd


def load_data():
    cvs_path = "resultsL.csv"
    return pd.read_csv(cvs_path)


ResultsD = load_data()
ResultsD.head()
#print (ResultsD["e"].value_counts())

ResultsD.hist()
plt.show()