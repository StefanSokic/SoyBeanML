import pandas as pd

#import data intro dataframes
df1 = pd.read_csv('dataset/training_data.csv')
df2 = pd.read_csv('dataset/Geographic_information.csv')

# add
# for i in range(len(df2.columns.values)):
#     if df2.columns.values[i] == 'LOCATION':
#         continue
#     df1[df2.columns.values[i]] = 'null'

print('original dataframe with null rows', df1.head(5))

print('geographical dataframe', df2.head(5))

# match up the locations on each of the files and add the columns for df2 to df1 where the locations match
# for i in range(len(df1['LOCATION'])):
#     for j in range(len(df2['LOCATION'])):
#         if df1['LOCATION'][i] == df2['LOCATION'][j]:
#             pass

merged_df = pd.merge(df1, df2, how='outer', on='LOCATION')
print('merged df', merged_df.head(5))
