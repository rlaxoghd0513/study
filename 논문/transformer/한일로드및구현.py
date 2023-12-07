#한-일 변환
from torchtext.legacy import data
from nltk.tokenize import WhitespaceTokenizer
import torch

tokenizer = WhitespaceTokenizer()

kor_text = data.Field(fix_length=50, sequential=True, batch_first=True, tokenize=tokenizer.tokenize)
eng_text = data.Field(fix_length=50, sequential=True, batch_first=True, tokenize=str.split, lower=True, init_token="<sos>", eos_token="<eos>")

# 데이터 불러오기
fields = [('kor', kor_text), ('eng', eng_text)]
train_data, valid_data, test_data = data.TabularDataset.splits(
    path='D:/한일중 말뭉치/한일씨에스브이',
    train='train.csv', validation='valid.csv', test='test.csv',
    format='csv',
    fields=fields,
    skip_header=True
)

# Field 적용
kor_text.build_vocab(train_data)
eng_text.build_vocab(train_data)

# print special tokens
print("kor <pad> token id:", kor_text.vocab.stoi['<pad>'])
print("eng <pad> token id:", eng_text.vocab.stoi['<pad>'])
print("eng <eos> token id:", eng_text.vocab.stoi['<eos>'])
print("eng <sos> token id:", eng_text.vocab.stoi['<sos>'])

batch_size = 32

train_loader = data.Iterator(dataset=train_data, batch_size=batch_size)
valid_loader = data.Iterator(dataset=valid_data, batch_size=batch_size)
test_loader = data.Iterator(dataset=test_data, batch_size=batch_size)

# one sample
batch = next(iter(train_loader))
print(batch.kor)
print(batch.eng)

def train(model, train_data, optimizer, device, epoch):
    # set model to train
    model.train()

    total_loss = 0
    for idx, batch in enumerate(train_data):
        src = batch.kor
        trg = batch.eng

        src = src.to(device)
        trg = trg.to(device)

        out = model(src, trg[:, :-1])  # output shape: [batch_size, trg_len - 1, output_dim]

        trg = trg[:, 1:].contiguous().view(-1)
        out = out.contiguous().view(-1, out.shape[-1])

        loss = loss_fn(out, trg)  # 만약 작동이 안되면 out = out.long(),trg = trg.float()를 시도해볼것.
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        total_loss += loss.item()

        if idx % 450 == 0:
            print(f"Epoch: {epoch} | Step: {idx}/{len(train_data)} | loss: {loss.item()}")

    return total_loss / len(train_data)
