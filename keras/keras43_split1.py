import numpy as np

dataset = np.array(range(1,11)) #1부터 10까지
timesteps = 5

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):  #len 개수  5번동안 반복한다
        subset = dataset[i : (i+timesteps)] #i부터 i+timesteps-1까지
        aaa.append(subset)        #append 넣는다
    return np.array(aaa)

bbb = split_x(dataset, timesteps)
print(bbb)
print(bbb.shape)

# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]

x = bbb[:, :4]   # 0번째 부터 시작 4-1번째 열까지 
# y = bbb[:, -1]
y = bbb[:, -1]
print(y)