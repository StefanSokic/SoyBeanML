import pandas as pd
import numpy as np
import random
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

# # checking for nulls
# print('merged_df nums', merged_df.isnull().values.any())
#
# # removing NaN columns
clean_df = merged_df._get_numeric_data()
# print('clean df', clean_df.head(5))

# 80/20 split of the data for training and testing
train = clean_df.sample(frac=0.8, random_state=200)
test = clean_df.drop(train.index)

# plotting the data to get a feel for it
# train.head(1000).plot(x='YEAR', y='TEMP_01', kind='scatter')
# plt.show()

# linear regression with scikit learn
# splitting data into features and target
# just training on the first temperature columns since this is univariate
feature_cols = ['TEMP_01']
target_col = ['YIELD']
x_train = train[feature_cols]
y_train = train[target_col]
x_test = test[feature_cols]
y_test = test[target_col]

# showing the inputs and outputs
print('x_ train', x_train.head())
print('y_train', y_train.head())

# the cost function

# the number of training samples
m = x_train.shape[0]

# initialize the thetas
theta = np.zeros(shape=(m, 1))

predictions = x_train.dot(theta).flatten()
print('predictions', predictions)


# # keep track of the sum
# sum = 0
# # set theta 1 to be a random value
# random_theta_one = random.random()
# print(random_theta_one)
# # loop through all of the columns of the x_train
# for column in x_train:
#     print(x_train[column])

# # showing some results
# print('Coefficients', regr.coef_)
# print("Mean squared error: %.2f" % np.mean((regr.predict(x_test) - y_test) ** 2))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % regr.score(x_test, y_test))
#
# # Plot outputs
# plt.scatter(x_test, y_test,  color='black')
# plt.plot(x_test, regr.predict(x_test), color='blue', linewidth=3)
# plt.xticks(())
# plt.yticks(())
# plt.show()