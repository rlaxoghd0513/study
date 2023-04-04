from keras.preprocessing.text import Tokenizer
import numpy as np

#1 데이터
docs = ['너무 재밋어요', '참 최고에요','참 잘 만든 영화에요', '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글세요', 
        '별로에요', '생각보다 지루해요', '연기가 어색해요', '재미없어요', '너무 재미없다', '참 재밌네요', '환희가 잘 생기긴 했어요', '환희가 안해요']

# 긍정 1 , 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0])
x_predict = ['나는 성호가 정말 재미없다 너무 정말']


#수치화
data = docs + x_predict
token = Tokenizer()
token.fit_on_texts(data)
print(token.word_index)
# {'너무': 1, '참': 2, '잘': 3, '재미없다': 4, '환희가': 5, '정말': 6, '재밋어요': 7, '최고에요': 8, '만든': 9, '영화에요': 10, '추천하고': 11, '싶은': 12, '영화입니다': 13, '한': 14, '번': 15, '더': 16, '보
# 고': 17, '싶네요': 18, '글세요': 19, '별로에요': 20, '생각보다': 21, '지루해요': 22, '연기가': 23, '어색해요': 24, '재미없어요': 25, '재밌네요': 26, '생기긴': 27, '했어요': 28, '안해요': 29, '나는': 30, '성
# 호가': 31}

print(token.word_counts)
# OrderedDict([('너무', 3), ('재밋어요', 1), ('참', 3), ('최고에요', 1), ('잘', 2), ('만든', 1), ('영화에요', 1), ('추천하고', 1), ('싶은', 1), ('영화입니다', 1), ('한', 1), ('번', 1), ('더', 1), ('보고', 1), ('싶네요', 1), ('글세요', 1), ('별로에요', 1), ('생각보다', 1), ('지루해요', 1), ('연기가', 1),
#              ('어색해요', 1), ('재미없어요', 1), ('재미없다', 2), ('재밌네요', 1), ('환희가', 2), ('생기긴', 1), ('했어요', 1), ('안해요', 1), ('나는', 1), ('성호가', 1), ('정말', 2)])


x = token.texts_to_sequences(data)
print(x) 
#[[1, 7], [2, 8], [2, 3, 9, 10], [11, 12, 13], [14, 15, 16, 17, 18], [19], [20], [21, 22], [23, 24], [25], [1, 4], [2, 26], [5, 3, 27, 28], [5, 29], [30, 31, 6, 4, 1, 6]]
print(type(x)) #<class 'list'> 


#수치화된 리스트 안의 문자들이 길이가 다 다르다

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen = 6) 
print(pad_x)
print(pad_x.shape) #(15, 6)

word_size = len(token.word_index) #총 몇개로 인덱싱 됐나
print('단어사전의 개수 : ', word_size) #단어사전의 개수 : 31

x_train = pad_x[:14,:]
x_test = pad_x[-1,:]
print(x_train.shape, x_test.shape) #(14, 6) (6,)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], 1)
x_test = x_test.reshape(1,6,1)

#모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Flatten, Dropout, Dense, Embedding
#어순이 있기 때문에 시계열? rnn


model = Sequential()
# model.add(Embedding(27,32, input_length = 5)) 
# model.add(Embedding(27,32,5)) 에러뜸  
model.add(Embedding(input_dim = 31, output_dim = 32, input_length = 6))  #Embedding(단어사전의 갯수, 아웃풋 갯수, 단어최대길이)  
#                                      단어사전의 갯수 부분 왠만하면 단어사전 갯수만큼 쓰는게 좋지만 더 큰 값을 넣든 작은값을 넣든 상관x                                
#                                      단어길이는 안써도 디폴트(최대값), 아웃풋은 통상 dense옆에 적던 숫자(아웃풋 필터 등)  
model.add(LSTM(64))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss= 'binary_crossentropy', optimizer= 'adam', metrics = ['acc'])
model.fit(x_train, labels, epochs = 100, batch_size = 32)

acc = model.evaluate(x_train, labels)[1]
print('acc:', acc)

###############################실습################################

y_predict = model.predict(x_test)
print(np.round(y_predict))


