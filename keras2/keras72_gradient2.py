import numpy as np

f = lambda x : x**2 - 4*x + 6
# def f(x):
    #    return x**2 -4*x + 6

gradient = lambda x: 2*x -4 #기울기를 찾아내는 f(x)를 미분한 함수 

x = -10.0
epochs = 20
learning_rate = 0.25

# gradientdescent 식/
# w = w - lr*(loss를 weight로 미분한거)

x_values = []  # x 값들을 저장하기 위한 리스트
f_values = []  # f(x) 값들을 저장하기 위한 리스트

print('epoch\t x\t f(x)')
print('{:02d}\t {:6.5f}\t {:6.5f}\t'.format(0,x,f(x)))
for i in range(epochs):

    x = x - learning_rate * gradient(x)

    print('{:02d}\t {:6.5f}\t {:6.5f}\t'.format(i+1,x,f(x)))


# 그래프 그리기
# import matplotlib.pyplot as plt

# x_values = np.array(x_values)
# f_values = np.array(f_values)

# plt.plot(x_values, f_values, 'k-')
# plt.grid()
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.show()