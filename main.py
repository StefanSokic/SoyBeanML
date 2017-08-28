import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import xgboost as xgb

# import data intro dataframes
df1 = pd.read_csv('dataset/training_data.csv')
df2 = pd.read_csv('dataset/Geographic_information.csv')

# print the original dataframes
# print('original dataframe with null rows', df1.head(5))
# print('geographical dataframe', df2.head(5))

# joining the 2 dataframes on the 'LOCATION' columns
merged_df = pd.merge(df1, df2, how='outer', on='LOCATION')
# print('merged df', merged_df.head(5))

# checking for nulls
# print('merged_df nums', merged_df.isnull().values.any())

# removing NaN columns
clean_df = merged_df._get_numeric_data().drop(['CHECK'], axis=1)
print('clean df', clean_df.head(5).to_string())

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

# xgboost implementation
print('before', x_train.values)
xgb_x_train = xgb.DMatrix(x_train.values)
print('after', xgb_x_train)
xgb_y_train = xgb.DMatrix(y_train.values)
xgb_test = xgb.DMatrix(x_test)

model = xgb.XGBRegressor()
model.fit(xgb_x_train, xgb_y_train)

# y_pred = model.predict(x_test)
# predictions = [round(value) for value in y_pred]
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))

# param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'reg:linear'}
# watchlist = [(xgb_test, 'eval'), (xgb_train, 'train')]
#



# # train xgboost for 1 round
# bst = xgb.train(param, xgb_train, 1, watchlist)
# # Note: we need the margin value instead of transformed prediction in set_base_margin
# # do predict with output_margin=True, will always give you margin values before logistic transformation
# ptrain = bst.predict(xgb_train, output_margin=True)
# ptest = bst.predict(x_test, output_margin=True)
# dtrain.set_base_margin(ptrain)
# dtest.set_base_margin(ptest)
#
# print ('this is result of running from initial prediction')
# bst = xgb.train(param, dtrain, 1, watchlist)

# scikit code
# regr = linear_model.LinearRegression()
# regr.fit(xgb_train, y_train)
#
# # showing some results
# print('Coefficients', regr.coef_)
# print("Mean squared error: %.2f" % np.mean((regr.predict(x_test) - y_test) ** 2))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % regr.score(x_test, y_test))
