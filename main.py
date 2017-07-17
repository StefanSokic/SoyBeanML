import pandas as pd
import numpy as np
import sklearn.model_selection as train_test_split
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

# try to visulalize - fails
clean_df.plot()

# 80/20 split of the data for training and testing
train = clean_df.sample(frac=0.8, random_state=200)
test = clean_df.drop(train.index)

print('train', train.head(5))
print('test', test.head(5))

