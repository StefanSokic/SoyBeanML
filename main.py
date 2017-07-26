import pandas as pd
import numpy as np
import sklearn.model_selection as train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt

# import data intro dataframes
df1 = pd.read_csv('dataset/training_data.csv')
df2 = pd.read_csv('dataset/Geographic_information.csv')

# print the original dataframes
print('original dataframe with null rows', df1.head(5))
print('geographical dataframe', df2.head(5))

# joining the 2 dataframes on the 'LOCATION' columns
merged_df = pd.merge(df1, df2, how='outer', on='LOCATION')
print('merged df', merged_df.head(5))

# checking for nulls
print('merged_df nums', merged_df.isnull().values.any())

# removing NaN columns
clean_df = merged_df._get_numeric_data()
print('clean df', clean_df.head(5))

# 80/20 split of the data for training and testing
train = clean_df.sample(frac=0.8, random_state=200)
test = clean_df.drop(train.index)

print('train', train.head(5))
print('test', test.head(5))

# plot the data,
train.head(1000).plot(x='YEAR', y='YIELD', kind='scatter')
plt.show()

# linear regression with scikit learn
# univariate linear regression just on YEAR to start
# --------- how to split train and test + use multivariate ---------------
# x_train = train['YEAR'].values.reshape(-1,1)
# y_train = train['YIELD'].values.reshape(-1,1)
# x_test = test['YEAR'].values.reshape(-1,1)
# y_test = test['YIELD'].values.reshape(-1,1)

# linear regression
# regr = linear_model.LinearRegression()
# regr.fit(x_train, y_train)

# showing some coefficients
# print('Coefficients', regr.coef_)

