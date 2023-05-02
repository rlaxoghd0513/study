import pandas as pd
import glob
path = './_data/ai_spark/'
train_csv = glob.glob(path+'TRAIN_AWS/*.csv')

li = []
for filename in train_csv:
    df = pd.read_csv(filename, index_col=None, header=0, encoding = 'utf-8-sig')
    li.append(df)
print(li)



# df_concat = pd.concat([train_csv, train_aws_csv.iloc[:,3:8]], axis=1)
# print(df_concat.shape)#(35064, 9)
# print(df_concat.columns)