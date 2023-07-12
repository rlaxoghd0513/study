import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):
    
    def __init__(self,max_len,d_model,device):
        super(PositionalEncoding,self).__init__()
        
        self.encoding = torch.zeros(max_len,d_model,device = device)
        self.encoding.requires_grad = False # we don't need to compute gradient
        
        pos = torch.arange(0,max_len,device=device)
        pos = pos.float().unsqueeze(dim = 1)
        
        _2i = torch.arange(0,d_model,step = 2,device = device).float()
        
        self.encoding[:,0::2] = torch.sin(pos/(10000**(_2i/d_model)))
        self.encoding[:,1::2] = torch.cos(pos/(10000**(_2i/d_model)))
        
    def forward(self,x):
        batch_size,seq_len = x.size()
        
        return self.encoding[:seq_len,:]
    
class ScaleDotProductAttention(nn.Module):
    '''
    Compute scale dot product attention
    실질적인 attention score을 계산하는 클래스
    
    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Query(encoder)
    Value : every sentence same with Key(Encoder)
    '''
    def __init__(self):
        super(ScaleDotProductAttention,self).__init__()
        self.softmax = nn.Softmax()
        
    def forward(self,q,k,v,mask = None, e = 1e-12):
        # input is 4 dimension tensor
        # [batch_size,head,length,d_tensor]
        batch_size,head,length,d_tensor = k.size()
        
        # 1. dot product Query with Key^T to compute similarity
        k_t = k.view(batch_size,head,d_tensor,length)
        score = (q @ k_t) / math.sqrt(d_tensor) # @연산은 np.matmul과 같은 역할
        
        '''
        Note) '@' operator
        If either argument is N-D, N > 2, 
        it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
        '''
        
        # 2. applying masking(optional)
        if mask is not None:
            score = score.masked_fill(mask == 0 ,-e)
        
        # 3. pass tem softmax to make [0,1] range
        score = self.softmax(score)
        
        # 4. Multiply with Value
        v = score @ v
        
        return v, score
    
class MultiHeadAttention(nn.Module):
    
    def __init__(self,d_model,n_head):
        super(MultiHeadAttention,self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_concat = nn.Linear(d_model,d_model)
    
    def split(self,tensor):
        '''
        splits tensor by number of head
        
        param tensor = [batch_size,length,d_model]
        out = [batch_size,head,length,d_tensor]
        
        d_model을 head와 d_tensor로 쪼개는걸로 이해하면 될듯. 
        d_tensor는 head의 값에 따라 변함.(head값은 정해주는 값이기 때문..)
        '''
        batch_size,length,d_model = tensor.size()
        
        d_tensor = d_model//self.n_head
        
        tensor = tensor.view(batch_size,self.n_head,length,d_tensor)
        
        return tensor
    
    def concat(self,tensor):
        '''
        inverse function of self.split(tensor = torch.Tensor)
        
        param tensor = [batch_size,head,length,d_tensor]
        out = [batch_size,length,d_model]
        '''
        batch_size,head,length,d_tensor = tensor.size()
        d_model = head*d_tensor
        
        tensor = tensor.view(batch_size,length,d_model)
        return tensor
    
    def forward(self,q,k,v,mask = None):
        
        #1. dot product with weight metrics
        q,k,v = self.w_q(q),self.w_k(k),self.w_v(v)
        
        # 2. split tensor by number of heads
        q,k,v = self.split(q),self.split(k),self.split(v)
        
        # 3. do scale dot product to compute similarity (attention 계산)
        out,attention = self.attention(q,k,v, mask = mask)
        
        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)
        
        return out
    
class LayerNorm(nn.Module):
    def __init__(self,d_model,eps = 1e-12):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self,x):
        mean = x.mean(-1,keepdim = True)
        std = x.std(-1,keepdim = True)
        # '-1' means last dimension
        
        out = (x-mean)/(std + self.eps)
        out = self.gamma * out + self.beta
        
        return out
    
class PositionwiseFeedForward(nn.Module):
    
    def __init__(self,d_model,hidden,drop_prob = 0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.linear1 = nn.Linear(d_model,hidden)
        self.linear2 = nn.Linear(hidden,d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = drop_prob)
    
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    

class EncoderLayer(nn.Module):
    
    def __init__(self,d_model,ffn_hidden,n_head,drop_prob):
        super(EncoderLayer,self).__init__()
        
        #Multi-Head Attention
        self.attention = MultiHeadAttention(d_model,n_head)
        
        #Layer Normalization(Multi-Head Attention ->)
        self.norm1 = LayerNorm(d_model = d_model)
        self.dropout1 = nn.Dropout(p = drop_prob)
        
        #Feed-Forward
        self.ffn = PositionwiseFeedForward(d_model = d_model,hidden = ffn_hidden,drop_prob = drop_prob)
        
        #Layer Normalization(FFN ->)
        self.norm2= LayerNorm(d_model = d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
    
    def forward(self,x,src_mask):
        _x = x
        
        #1. Compute Multi-Head Attention
        x = self.attention(q= x,k= x,v= x,mask = src_mask)
        
        #2. Compute add & norm
        x = self.norm1(x + _x)
        x = self.dropout1(x)
        
        # 3. Compute Feed-Forward Network
        _x = x
        x = self.ffn(x)
        
        # 4. Compute add & norm
        x = self.norm2(x + _x)
        x = self.dropout2(x)
        
        return x
    
class Encoder(nn.Module):
    
    def __init__(self,enc_voc_size,max_len,d_model,ffn_hidden,n_head,n_layers,
                drop_prob,device):
        super().__init__()
        
        #Embedding
        self.embed = nn.Embedding(num_embeddings = len(kor_text.vocab),embedding_dim = d_model,padding_idx = 1)
        
        #Positional Encoding
        self.pe = PositionalEncoding(max_len = max_len,d_model = d_model,device = device)
        
        #Add Multi layers
        self.layers = nn.ModuleList([EncoderLayer(d_model = d_model,
                                                 ffn_hidden = ffn_hidden,
                                                 n_head = n_head,
                                                 drop_prob = drop_prob)
                                    for _ in range(n_layers)])
        
    def forward(self,x,src_mask):
    	#Compute Embedding
        x = self.emb(x) #sentence -> vector
        
        #Get Positional Encoding
        x_pe = self.pe(x)
        
        #Embedding + Positional Encoding
        x = x + x_pe
        
        #Compute Encoder layers
        for layer in self.layers:
            x = layer(x,src_mask)
        
        #Return encoder output
        return x
    
class DecoderLayer(nn.Module):
    
    def __init__(self,d_model,ffn_hidden,n_head,drop_prob):
        super(DecoderLayer,self).__init__()
        
        #self attention(only Decoder input)
        self.self_attention = MultiHeadAttention(d_model = d_model,n_head = n_head)
        
        #layer normalization(first)
        self.norm1 = LayerNorm(d_model = d_model)
        #dropout(first)
        self.dropout1 = nn.Dropout(p=drop_prob)
        
        #attention(encoder + decoder)
        self.enc_dec_attention = MultiHeadAttention(d_model = d_model,n_head = n_head)
        
        #layer normalization(second)
        self.norm2 = LayerNorm(d_model = d_model)
        #dropout(second)
        self.dropout2 = nn.Dropout(p=drop_prob)
        
        #Feed-Forward
        self.ffn = PositionwiseFeedForward(d_model = d_model,hidden = ffn_hidden,
                                           drop_prob = drop_prob)
        #Layer normalization(third)
        self.norm3 = LayerNorm(d_model = d_model)
        #dropout(third)
        self.dropout3 = nn.Dropout(p= drop_prob)
        
    def forward(self,dec,enc,trg_mask,src_mask):
        
        _x = dec
        #Compute self-attention
        x = self.self_attention(q = dec,k = dec,v = dec,mask = trg_mask)
        
        #Compute add & norm
        x = self.norm1(x + _x)
        x=  self.dropout1(x)
        
        if enc is not None:  #encoder의 출력값이 있다면 (없으면 FFN으로 넘어감)
            _x = x
            
            #Compute encoder - decoder attention
            #Query(q) : decoder attention output
            #Key(k) : Encoder output
            #Value(v) : Encoder output
            x = self.enc_dec_attention(q = x,k = enc,v = enc,mask = src_mask)
            
            #Compute add & norm
            x = self.norm2(x + _x)
            x = self.dropout2(x)
            
        _x = x
        
        #Compute FFN
        x = self.ffn(x)
        
        #Compute add & norm
        x = self.norm3(x + _x)
        x = self.dropout3(x)
        
        return x
    
class Decoder(nn.Module):
    def __init__(self,dec_voc_size,max_len,d_model,ffn_hidden,n_head,n_layers,
                drop_prob,device):
        super().__init__()
        
        #Embedding
        self.embed = nn.Embedding(num_embeddings = len(eng_text.vocab),embedding_dim = d_model,padding_idx = 1)
        
        #Positional Encoding
        self.pe = PositionalEncoding(max_len = 50,d_model = d_model,device = 'cuda')
        
        #Add decoder layers
        self.layers = nn.ModuleList([DecoderLayer(d_model = d_model,
                                                 ffn_hidden = ffn_hidden,
                                                 n_head = n_head,
                                                 drop_prob = drop_prob)
                                    for _ in range(n_layers)])
        
        #Linear
        self.linear = nn.Linear(d_model,dec_voc_size)
    
    def forward(self,trg,src,trg_mask,src_mask):
        
        #Compute Embedding
        trg = self.embed(trg)
        
        #Get Positional Encoding
        trg_pe = self.pe(trg)
        
        #Embedding + Positional Encoding
        trg = trg + trg_pe
        
        #Compute Decoder layers
        for layer in self.layers:
            trg = layer(trg,src,trg_mask,src_mask)
        
        #pass to LM head
        output = self.linear(trg)
        
        return output

class Transformer(nn.Module):
    
    def __init__(self,src_pad_idx,trg_pad_idx,trg_sos_idx,enc_voc_size,dec_voc_size,d_model,n_head,max_len,
                ffn_hidden,n_layers,drop_prob,device):
        super().__init__()
        #Get <PAD> idx
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        
        #Encoder
        self.encoder = Encoder(enc_voc_size = enc_voc_size,
                              max_len = max_len,
                              d_model = d_model,
                              ffn_hidden = ffn_hidden,
                              n_head = n_head,
                              n_layers = n_layers,
                              drop_prob = drop_prob,
                              device = device)
        
        #Decoder
        self.decoder = Decoder(dec_voc_size = dec_voc_size,
                              max_len = max_len,
                              d_model = d_model,
                              ffn_hidden = ffn_hidden,
                              n_head = n_head,
                              n_layers = n_layers,
                              drop_prob = drop_prob,
                              device = device)
        self.device = device
    
    def make_pad_mask(self,q,k):
    
    	#Padding부분은 attention연산에서 제외해야하므로 mask를 씌워줘서 계산이 되지 않도록 한다.
        
        len_q,len_k = q.size(1),k.size(1)
        print(len_k)
        #batch_size x 1 x 1 x len_k
        k = k.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        print(k.shape)
        # batch_size x1 x len_1 x len_k
        k = k.repeat(1,1,len_q,1)
        
        #batch_size x 1 x len_q x 1
        q = q.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(3)
        #batch_size x 1 x len_q x len_k
        q = q.repeat(1,1,1,len_k)
        
        mask = k & q
        
        return mask
    
    def make_no_peak_mask(self,q,k):
    	
        #Decoder 부분에서 t번째 단어를 예측하기 위해 입력으로 t-1번째 단어까지 넣어야 하므로 나머지 부분을 masking처리 한다.
        #만약 t번째 단어를 예측하는데 이미 decoder에 t번째 단어가 들어간다면?? => 답을 이미 알고 있는 상황..
        #따라서 Seq2Seq 모델에서 처럼 t번째 단어를 예측하기 위해서 t-1번째 단어까지만 입력될 필요가 있음
        #(나머지 t,t+1,...,max_len)까지 단어는 t번째 단어를 예측하는데 전혀 필요하지 않음 => Masking!!
        len_q,len_k = q.size(1),k.size(1)
        
        #len_q x len_k (torch.tril = 하삼각행렬)
        mask = torch.tril(torch.ones(len_q,len_k)).type(torch.BoolTensor).to(self.device)
        
        return mask
    
    def forward(self,src,trg):
    	
        #Get Mask
        src_mask = self.make_pad_mask(src,src)
        src_trg_mask = self.make_pad_mask(trg,src)
        trg_mask = self.make_pad_mask(trg,trg) * self.make_no_peak_mask(trg,trg)
        
        #Compute Encoder
        enc_src = self.encoder(src,src_mask)
        
        #Compute Decoder
        output = self.decoder(trg,enc_src,trg_mask,src_trg_mask)
        
        return output
    
import pandas as pd
import os
from sklearn.model_selection import train_test_split

file_path = "D:/한일중 말뭉치/엑셀변환/"
file_list= ['한일.xlsx']
#or use os.listdir
#1
file_1 = pd.read_excel(file_path + file_list[0])
kor_total = file_1['ko']
eng_total = file_1["jp"]

# #2. 
# file_2 = pd.read_excel(file_path + file_list[1])
# kor_2 = file_2['한국어']
# eng_2 = file_2['영어검수']

# #3.
# file_3 = pd.read_excel(file_path + file_list[2])
# kor_3 = file_3['원문']
# eng_3 = file_3['REVIEW']

# #4~6
# kor_46 = pd.Series()
# eng_46 = pd.Series()
# for name in file_list[3:]:
#     files = pd.read_excel(file_path + name)
    
#     kor_46 = pd.concat([kor_46,files['원문']])
#     eng_46 = pd.concat([eng_46,files['Review']])

# kor_total = pd.concat([kor_1,kor_2,kor_3,kor_46])
# eng_total = pd.concat([eng_1,eng_2,eng_3,eng_46])

total = pd.DataFrame({"kor" : kor_total,"eng" : eng_total})

print(total.head())

#split data
train,test= train_test_split(total,test_size = 0.3)
valid,test = train_test_split(test,test_size = 0.6)

print("train_data size : ",len(train))
print("valid_data size : ",len(valid))
print("test_data size : ",len(test))

train.to_csv("D:/한일중 말뭉치/한일씨에스브이/train.csv",encoding = 'utf-8',index = False)
valid.to_csv("D:/한일중 말뭉치/한일씨에스브이/valid.csv",encoding = 'utf-8',index = False)
test.to_csv("D:/한일중 말뭉치/한일씨에스브이/test.csv",encoding = 'utf-8',index = False)

# from torchtext import data
# from konlpy.tag import Mecab

# tokenizer = Mecab()

# kor_text = data.Field(fix_length=50,sequential= True,batch_first= True,tokenize=tokenizer.morphs)
# eng_text = data.Field(fix_length=50,sequential= True,batch_first= True,tokenize=str.split,lower= True,init_token="<sos>",eos_token="<eos>")

# #데이터 불러오기
# train_data,valid_data,test_data = data.TabularDataset.splits(
#     path = './data/',
#     train = 'train.csv',validation='valid.csv',test = 'test.csv',
#     format = 'csv',
#     fields = field,
#     skip_header = True
# )

# #Field 적용
# kor_text.build_vocab(train_data)
# eng_text.build_vocab(train_data)

# #print special tokens
# print("kor <pad>token id : ",kor_text.vocab.stoi['<pad>'])
# print("eng <pad>token id : ",eng_text.vocab.stoi['<pad>'])
# print("eng <eos>token id : ",eng_text.vocab.stoi['<eos>'])
# print("eng <sos>token id : ",eng_text.vocab.stoi['<sos>'])

# train_loader = Iterator(dataset = train_data, batch_size = batch_size)
# valid_loader = Iterator(dataset = valid_data, batch_size = batch_size)
# test_loader = Iterator(dataset = test_data,batch_size = batch_size)

# #one sample
# batch = next(iter(train_loader))
# print(batch.kor)
# print(batch.eng)

# def train(model,train_data,optimizer,device,epoch):

#     #set model to train
#     model.train()
    
#     total_loss = 0
#     for idx,batch in enumerate(train_data):
#         src = batch.kor
#         trg = batch.eng

#         src = src.to(device)
#         trg = trg.to(device)

#         out = model(src,trg[:,:-1]) #output shape : [batch_size,trg_len -1,output_dim]
        
#         trg = trg[:,1:].contiguous().view(-1)
#         out = out.contiguous().view(-1,out.shape[-1])

#         loss = loss_fn(out,trg) #만약 작동이 안되면 out = out.long(),trg = trg.float()를 시도해볼것.
#         optimizer.zero_grad()
#         loss.backward()
        
#         #gradient clipping
#         torch.nn.utils.clip_grad_norm_(model.parameters(),1)
#         optimizer.step()

#         total_loss += loss.item()
        
#         if idx % 450 == 0 :
#             print(f"Epoch : {epoch} | Step :{idx}/{len(train_data)} | loss : {loss.item()}")
        
#     return total_loss / len(train_data)

