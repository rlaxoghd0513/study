from keras.preprocessing.text import Tokenizer
import numpy as np

#1 데이터
docs = ['너무 재밋어요', '참 최고에요','참 잘 만든 영화에요', '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글세요', 
        '별로에요', '생각보다 지루해요', '연기가 어색해요', '재미없어요', '너무 재미없다', '참 재밌네요', '환희가 잘 생기긴 했어요', '환희가 안해요']

# 긍정 1 , 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0])
x_predict = '나는 성호가 정말 재미없다 너무 정말'
#수치화
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
#{'참': 1, '너무': 2, '잘': 3, '환희가': 4, '재밋어요': 5, '최고에요': 6, '만든': 7, '영화에요': 8, '추천하고': 9, '싶은': 10, '영화입니다': 11, '한': 12, '번': 13, '더': 14, '보고': 15, '싶네요': 16, '글세
# 요': 17, '별로에요': 18, '생각보다': 19, '지루해요': 20, '연기가': 21, '어색해요': 22, '재미없어요': 23, '재미없다': 24, '재밌네요': 25, '생기긴': 26, '했어요': 27, '안해요': 28}

print(token.word_counts)


x = token.texts_to_sequences(docs)
print(x) #[[2, 5], [1, 6], [1, 3, 7, 8], [9, 10, 11], [12, 13, 14, 15, 16], [17], [18], [19, 20], [21, 22], [23], [2, 24], [1, 25], [4, 3, 26, 27], [4, 28]]
print(type(x)) #<class 'list'> 


#수치화된 리스트 안의 문자들이 길이가 다 다르다
#제일 긴게 4개 들어있는거니까 맞춰주기 위해 빵구난 부분 0으로 채운다(패딩) 패딩은 앞에 한다 -> (14,4)
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen = 6) #패딩=프리: 앞에서부터 0을 채울거다  #맥스렌: 제일긴거에 맞출거다   #맥스렌보다 크면 패딩이 프리이기 때문에 앞에서부터 잘려나간다
print(pad_x)
print(pad_x.shape)

word_size = len(token.word_index) #총 몇개로 인덱싱 됐나
print('단어사전의 개수 : ', word_size) #단어사전의 개수 :  28


#모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Flatten, Dropout, Dense, Embedding
#어순이 있기 때문에 시계열? rnn


pad_x = pad_x.reshape(pad_x.shape[0],pad_x.shape[1],1)

#임베딩 3차원 받고 3차원 내보냄
#임베딩 후 통상 LSTM, Conv1D

model = Sequential()
# model.add(Embedding(27,32, input_length = 5)) 
# model.add(Embedding(27,32,5)) 에러뜸  
model.add(Embedding(input_dim = 31, output_dim = 32, input_length = 6))  #Embedding(단어사전의 갯수, 아웃풋 갯수, 단어최대길이)  
#                                      단어사전의 갯수 부분 왠만하면 단어사전 갯수만큼 쓰는게 좋지만 더 큰 값을 넣든 작은값을 넣든 상관x                                
#                                      단어길이는 안써도 디폴트(최대값), 아웃풋은 통상 dense옆에 적던 숫자(아웃풋 필터 등)  
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss= 'binary_crossentropy', optimizer= 'adam', metrics = ['acc'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', mode = 'min', restore_best_weights=True, patience = 50)
model.fit(pad_x, labels, epochs = 10000, batch_size = 32 , callbacks=[es])

acc = model.evaluate(pad_x, labels)[1]
print('acc:', acc)

###############################실습################################


#긍정인지 부정인지

token.fit_on_texts([x_predict])
print(token.word_index)
#{'너무': 1, '참': 2, '잘': 3, '재미없다': 4, '환희가': 5, '정말': 6, '재밋어요': 7, '최고에요': 8, '만든': 9, '영화에요': 10, '추천하고': 11, '싶은': 12, '영화입니다': 13, '한': 14, '번': 15, '더': 16, '보 
# 고': 17, '싶네요': 18, '글세요': 19, '별로에요': 20, '생각보다': 21, '지루해요': 22, '연기가': 23, '어색해요': 24, '재미없어요': 25, '재밌네요': 26, '생기긴': 27, '했어요': 28, '안해요': 29, '나는': 30, '성
# 호가': 31}

x_predict = token.texts_to_sequences([x_predict])
print(x_predict) #[[30, 31, 6, 4, 1, 6]]

y_predict = model.predict(x_predict)
print(np.round(y_predict)) 


