import pandas as pd

df = pd.DataFrame({'A': [1,2,3,4,5],
                   'B': [10,20,30,40,50],
                   'C': [5,4,3,2,1,]})

print(df)
correlations = df.corr()

print(correlations)
#    A   B  C
# 0  1  10  5
# 1  2  20  4
# 2  3  30  3
# 3  4  40  2
# 4  5  50  1
#      A    B    C
# A  1.0  1.0 -1.0
# B  1.0  1.0 -1.0
# C -1.0 -1.0  1.0

#단순히 기울기만 보여주기 때문에 너무 신뢰하면 안된다 참고만 해야한다

