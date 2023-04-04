#임베딩 효율적인 원핫
# 원핫은 데이터가 많아지면 무수한 0이 생겨난다 사전을 원핫하면 컴퓨터 터진다 
# 임베딩은 좌표로 찍어준다
# 유사한 데이터들끼리 모인다
# 다 사용가능하지만 텍스트 데이터에서 특히 잘먹힌다 자연어처리


from keras.preprocessing.text import Tokenizer
import numpy as np

#1 데이터
docs = ['너무 재밋어요', '참 최고에요','참 잘 만든 영화에요', '추천하고 싶은 영화입니다', '한번 더 보고 싶네요', '글세요', 
        '별로에요', '생각보다 지루해요', '연기가 어색해요', '재미없어요', '너무 재미없다', '참 재밌네요', '환희가 잘 생기긴 했어요', '환희가 안해요']

# 긍정 1 , 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0])

#수치화
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
#{'참': 1, '너무': 2, '잘': 3, '환희가': 4, '재밋어요': 5, '최고에요': 6, '만든': 7, '영화에요': 8, '추천하고': 9, '싶은': 10, '영화입니다': 11, '한번': 12, '더': 13, '보고': 14, '싶네요': 15, '글세요': 16, 
# '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24, '생기긴': 25, '했어요': 26, '안해요': 27}

print(token.word_counts)


x = token.texts_to_sequences(docs)
print(x) #[[2, 5], [1, 6], [1, 3, 7, 8], [9, 10, 11], [12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [4, 3, 25, 26], [4, 27]]
print(type(x)) #<class 'list'> 


#수치화된 리스트 안의 문자들이 길이가 다 다르다
#제일 긴게 4개 들어있는거니까 맞춰주기 위해 빵구난 부분 0으로 채운다(패딩) 패딩은 앞에 한다 -> (14,4)

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen = 4) #패딩=프리: 앞에서부터 0을 채울거다  #맥스렌: 제일긴거에 맞출거다   #맥스렌보다 크면 패딩이 프리이기 때문에 앞에서부터 잘려나간다
print(pad_x)
print(pad_x.shape)

word_size = len(token.word_index) #총 몇개로 인덱싱 됐나
print('단어사전의 개수 : ', word_size) #단어사전의 개수 :  27


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
model.add(Embedding(input_dim = 27, output_dim = 10, input_length = 4))  #Embedding(단어사전의 갯수, 아웃풋 갯수, 단어최대길이)  
#                                      단어사전의 갯수 부분 왠만하면 단어사전 갯수만큼 쓰는게 좋지만 더 큰 값을 넣든 작은값을 넣든 상관x                                
#                                      단어길이는 안써도 디폴트(최대값), 아웃풋은 통상 dense옆에 적던 숫자(아웃풋 필터 등)  
model.add(LSTM(32))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss= 'binary_crossentropy', optimizer= 'adam')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', mode = 'min', restore_best_weights=True, patience = 50)
model.fit(pad_x, labels, epochs = 10000, batch_size = 32 , callbacks=[es])

loss = model.evaluate(pad_x, labels)
print('loss:', loss)

y_predict = model.predict(pad_x)

from sklearn.metrics import accuracy_score
acc = accuracy_score(labels ,np.round(y_predict))
print('acc:', acc)