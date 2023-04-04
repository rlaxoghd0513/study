import numpy as np

a = [[1,2,3],[4,5,6]] #(2,3)

b = np.array(a)
print(b) #[[1 2 3]
         # [4 5 6]]
         
c = [[1,2,3], [4,5]]
print(c)
d = np.array(c)
print(d) #[list([1, 2, 3]) list([4, 5])] 리스트 두개가 넘파이안에 들어가있는 형태

# 리스트는 파이썬,  리스트에 들어가는건 크기가 달라도 상관없다
# 하지만 넘파이로 바꿀라하면 오류는 안뜨지만 우리가 원하던 형태로는 뜨지가 안는다
# 넘파이는 크기가 같아야한다 리스트는 크기가 달라도 상관없다

##############################################################
e = [[1,2,3],["바보", "맹구", 5,6]]
print(e) #[[1, 2, 3], ['바보', '맹구', 5, 6]]

#리스트 안에는 수치형 문자형 각각 다른 걸 넣어도 상관없다

f = np.array(e)
print(f) #[list([1, 2, 3]) list(['바보', '맹구', 5, 6])] 
# 이대로는 연산을 할 수 없다
# numpy 크기만 맞으면 되긴 한데 이 상태로는 연산이 안됨
# print(e.shape) #리스트는 쉐이프가 제공되지 않는다
print(len(e)) #2

print(f.shape)

#넘파이나 판다스는 한가지 자료형만 써야한다