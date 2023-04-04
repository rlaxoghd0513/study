from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

text1 = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'
text2 = '나는 지구용사 배환희다. 멋있다. 또 또 얘기해부아'


token = Tokenizer() #토크나이저로 나오는 형태는 리스트
token.fit_on_texts([text1, text2]) 
print(token.word_index) #토큰의 형태를 보기위함
#{'마구': 1, '나는': 2, '매우': 3, '또': 4, '진짜': 5, '맛있는': 6, '밥을': 7, '엄청': 8, '먹었다': 9, '지구용사': 10, '배환희다': 11, '멋있다': 12, '얘기해부아': 13}

print(token.word_counts) 
#OrderedDict([('나는', 2), ('진짜', 1), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 3), 
#             ('먹었다', 1), ('지구용사', 1), ('배환희다', 1), ('멋있다', 1), ('또', 2), ('얘기해부아', 1)])  

x = token.texts_to_sequences([text1, text2]) #수치화된걸 text 순서대로 찍어라
print(x)  #[[2, 5, 3, 3, 6, 7, 8, 1, 1, 1, 9], [2, 10, 11, 12, 4, 4, 13]]  #둘이 길이가 다르다


print(type(x))  #<class 'list'>
##############원핫인코딩################## 


####################################################    
#  # 케라스  to_categorical
 
#  #쫙펴라
# x = x[0]+ x[1]   # 넘파이는 컨캣으로 리스트는 걍 더하기
# print(x) #[2, 5, 3, 3, 6, 7, 8, 1, 1, 1, 9, 2, 10, 11, 12, 4, 4, 13]

# from tensorflow.keras.utils import to_categorical 
# x = to_categorical(x)

# print(x)
# print(x.shape) #(18, 14)
 

#####################################################
# 판다스 get_dummies      #겟더미는 1차원만 받아들인다
# 문제점 : 리스트는 겟더미에서 받아들일 수 없다 
# 해결 : 리스트를 넘파이로 바꾸거나
# 왜 못받아들이는지 : 

# import pandas as pd
# import numpy as np

# x = x[0]+ x[1]

# x = np.array(x) 
# print(type(x))#<class 'numpy.ndarray'>
# print(x)#
# print(x.shape) #

# x = pd.get_dummies(x.ravel()) 

# print(x)

# print(type(x)) #<class 'pandas.core.frame.DataFrame'>

####################################################
# 사이킷런 onehotencoder #2차원으로 받아들인다   #리스트 안받는다

from sklearn.preprocessing import OneHotEncoder

x = x[0]+ x[1]

ohe = OneHotEncoder() #원핫인코딩은 열별로 원핫을 하는데 각 열에 하나의 숫자씩 밖에 없으니까 가로세로 바꿔줘야한다 아니면 다 1로 원핫 된다

#1 한줄로 요약한거 
# x  = ohe.fit_transform(np.array(x).reshape(11,1)).toarray()

#2 여러줄 
x = np.array(x)
print(type(x)) #<class 'numpy.ndarray'>
print(x.shape) #

x = x.reshape(-1,1)
print(x)


x = ohe.fit_transform(x).toarray() 
print(type(x)) #<class 'numpy.ndarray'>
print(x.shape) #(18, 13)
print(x)

