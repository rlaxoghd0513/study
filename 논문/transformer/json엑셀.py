import json
import pandas as pd

# JSON 데이터를 읽어옵니다
with open('D:/한일중 말뭉치/157.방송 콘텐츠 한-중, 한-일 번역 병렬 말뭉치 데이터/01.데이터/1.Training/원천데이터/TS1/2-5_방송콘텐츠_일한_train_set_360000.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# 데이터에서 필요한 필드를 추출하여 새로운 리스트를 만듭니다.
features = ['jp_original','jp','mt','ko']
data = []
for item in json_data['data']:
    feature_values = [item[feature] for feature in features]
    data.append(feature_values)

# DataFrame을 생성합니다.
df = pd.DataFrame(data, columns=features)

# DataFrame을 Excel 파일로 저장합니다.
df.to_excel('D:/한일중 말뭉치/엑셀변환/한일.xlsx', index=False)