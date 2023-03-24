import numpy as np
datasets = np.array(range(1,41)).reshape(10,4)
print(datasets)

# x_data = datasets[:,:3]
x_data = datasets[:,:-1] #모든행과 열의 마지막전까지
y_data = datasets[:,-1]
print(x_data.shape, y_data.shape)  #(10, 3) (10,)

timesteps = 2
############### x만들기 ##########################
def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps):  #스플릿한 x_daata중 마지막꺼 못쓰니까 +1없애준다  
        subset = dataset[i : (i+timesteps)] 
        aaa.append(subset)      
    return np.array(aaa)

x_data = split_x(x_data, timesteps)
print(x_data)
print(x_data.shape) #(6,5,3)

############### y만들기########################
y_data = y_data[timesteps:]   #y데이타는 예측을 해야하는거다  앞에 timestep 수만큼 빼준다

