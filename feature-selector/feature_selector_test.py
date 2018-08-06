from feature_selector import FeatureSelector
import pandas as pd

train = pd.read_csv('credit_example.csv')
train_labels = train['TARGET']
train.head()
train = train.drop(columns=['TARGET'])

fs = FeatureSelector(data=train, labels=train_labels)

fs.identify_missing(missing_threshold=0.6)
fs.plot_missing()