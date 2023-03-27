for i in range(len(df1.index)):
    df1.iloc[i,4] = int(df1.iloc[i,4].replace(',',""))
    
