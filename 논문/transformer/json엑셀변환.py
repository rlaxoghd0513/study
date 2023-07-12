import pandas as pd

# JSON 데이터를 pandas DataFrame으로 읽기
data = pd.read_json('D:/한일중 말뭉치/157.방송 콘텐츠 한-중, 한-일 번역 병렬 말뭉치 데이터/01.데이터/1.Training/원천데이터/TS1/2-5_방송콘텐츠_일한_train_set_360000.json')

# DataFrame을 Excel 형식으로 변환
data.to_excel('D:/한일중 말뭉치/엑셀변환/output.xlsx', index=False)