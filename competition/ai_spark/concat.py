import pandas as pd
path = './_data/ai_spark/TRAIN_AWS/'
train_csv = pd.read_csv(path+'공주.csv')
train_aws_csv = pd.read_csv('./_data/ai_spark/TRAIN_AWS/공주.csv')
print(train_csv.shape)      #(35064, 4)
print(train_aws_csv.shape)  #(35064, 8)

df_concat = pd.concat([train_csv, train_aws_csv.iloc[:,3:8]], axis=1)
print(df_concat.shape)#(35064, 9)
print(df_concat.columns)