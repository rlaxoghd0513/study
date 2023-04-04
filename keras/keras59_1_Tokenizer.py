from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

text = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'

#Tokenizer 문장을 어절별로 (어절)(띄어쓰기단위) 토큰화해서 잘라낸다
#이미지를 컴퓨터가 인식할수 있게 수치화한것 처럼 문자도 수치화한다

token = Tokenizer() #토크나이저로 나오는 형태는 리스트
token.fit_on_texts([text]) #한문장뿐 아니라 여러 문장을 토큰화 할수있다 두개이상은 리스트

print(token.word_index) #토큰의 형태를 보기위함
#{'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8} 
#제일 많은 어구가 맨 앞 1로 인덱싱 됐다 같은 개수들은 앞에서 부터 순서대로 인덱싱 됐다

print(token.word_counts) 
#OrderedDict([('나는', 1), ('진짜', 1), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1)]) 키밸류형태로나옴
#매우는 두번나왔고 마구는 세번 나왔다

x = token.texts_to_sequences([text]) #수치화된걸 text 순서대로 찍어라
print(x) #[[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]] ##(1,11) 1행 11열

print(type(x))  #<class 'list'>
##############원핫인코딩################## (11,8)나와야함


####################################################    
 # 케라스  to_categorical  #리스트도 받아들이는것 같다    ####고쳐보기#### ###################################################
 # 문제점 : 투카테고리컬로 하면 0부터 생기는데  토크나이저는 1부터 생긴다
 # 0을 지우고 리쉐잎 하면 되긴한다

from tensorflow.keras.utils import to_categorical 
x = to_categorical(x)

print(x)
print(x.shape) #(1, 11, 9)
 

#####################################################
# 판다스 get_dummies      #겟더미는 1차원만 받아들인다
# 문제점 : 리스트는 겟더미에서 받아들일 수 없다 
# 해결 : 리스트를 넘파이로 바꾸거나
# 왜 못받아들이는지 : 

# import pandas as pd
# import numpy as np

# x = np.array(x) 
# print(type(x))#<class 'numpy.ndarray'>
# print(x)#[[3 4 2 2 5 6 7 1 1 1 8]]
# print(x.shape) #(1, 11) #2차원

# #1 이것도 되고,
#  # x = np.squeeze(x)
#  # print(type(x))#<class 'numpy.ndarray'>
#  # print(x) #[3 4 2 2 5 6 7 1 1 1 8]
#  # print(x.shape) #(11,)
 
# #2 리쉐잎으로 차원 바꾸는것도 된다
# # x = pd.get_dummies(x.reshape(11,))

# #3 ravel() = 넘파이 쭉피는 놈 1차원으로 만들라고
# x = pd.get_dummies(x.ravel()) 

# print(x)
#     1  2  3  4  5  6  7  8
# 0   0  0  1  0  0  0  0  0
# 1   0  0  0  1  0  0  0  0
# 2   0  1  0  0  0  0  0  0
# 3   0  1  0  0  0  0  0  0
# 4   0  0  0  0  1  0  0  0
# 5   0  0  0  0  0  1  0  0
# 6   0  0  0  0  0  0  1  0
# 7   1  0  0  0  0  0  0  0
# 8   1  0  0  0  0  0  0  0
# 9   1  0  0  0  0  0  0  0
# 10  0  0  0  0  0  0  0  1
# print(type(x)) #<class 'pandas.core.frame.DataFrame'>

####################################################
# 사이킷런 onehotencoder #2차원으로 받아들인다   #리스트 안받는다

# from sklearn.preprocessing import OneHotEncoder

# ohe = OneHotEncoder() #원핫인코딩은 열별로 원핫을 하는데 각 열에 하나의 숫자씩 밖에 없으니까 가로세로 바꿔줘야한다 아니면 다 1로 원핫 된다

# #1 한줄로 요약한거 
# # x  = ohe.fit_transform(np.array(x).reshape(11,1)).toarray()

# #2 여러줄 
# x = np.array(x)
# print(type(x)) #<class 'numpy.ndarray'>
# print(x.shape) #(1, 11)

# x = x.reshape(-1,1)
# print(x)
# # [[3]
# #  [4]
# #  [2]
# #  [2]
# #  [5]
# #  [6]
# #  [7]
# #  [1]
# #  [1]
# #  [1]
# #  [8]]

# x = ohe.fit_transform(x).toarray() 
# print(type(x)) #<class 'numpy.ndarray'>
# print(x.shape) #(11, 8)
# print(x)
# # [[0. 0. 1. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 1. 0. 0. 0. 0.]
# #  [0. 1. 0. 0. 0. 0. 0. 0.]
# #  [0. 1. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 1. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 1. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 1. 0.]
# #  [1. 0. 0. 0. 0. 0. 0. 0.]
# #  [1. 0. 0. 0. 0. 0. 0. 0.]
# #  [1. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 0. 1.]]



