import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import confusion_matrix, mean_squared_error

# import data intro dataframes
df1 = pd.read_csv('dataset/training_data.csv')
df2 = pd.read_csv('dataset/Geographic_information.csv')

# joining the 2 dataframes on the 'LOCATION' columns
merged_df = pd.merge(df1, df2, how='outer', on='LOCATION')

# removing NaN columns
clean_df = merged_df._get_numeric_data().drop(['CHECK', 'LOCATION', 'CLASS_OF'], axis=1)
print('clean df', clean_df.head(5).to_string())

# 80/20 split of the data for training and testing
train = clean_df.sample(frac=0.8, random_state=200)
test = clean_df.drop(train.index)

# setting the feature and target columns up for multivariate linear regression
target_col = ['YIELD']
feature_cols = list(train)
feature_cols.remove(target_col[0])
x_train = np.array(train[feature_cols])
y_train = np.array(train[target_col])
x_test = np.array(test[feature_cols])
y_test = np.array(test[target_col])

# xgboost implementation
x_train = x_train
y_train = y_train

model = xgb.XGBRegressor()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# y_train_pred = model.predict(x_train)
print(mean_squared_error(y_test, y_pred))
# print(mean_squared_error(y_train, y_train_pred))

# 76.4736525104

