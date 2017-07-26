import pandas as pd
import numpy as np
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

# plot the data to explore the dataset
# train.head(1000).plot(x='YEAR', y='YIELD', kind='scatter')
# plt.show()

# setting the feature and target columns up for multivariate linear regression
target_col = ['YIELD']
feature_cols = list(train)
feature_cols.remove(target_col[0])
x_train = train[feature_cols]
y_train = train[target_col]
x_test = test[feature_cols]
y_test = test[target_col]

# linear regression
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

# showing some results
print('Coefficients', regr.coef_)
print("Mean squared error: %.2f" % np.mean((regr.predict(x_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x_test, y_test))
