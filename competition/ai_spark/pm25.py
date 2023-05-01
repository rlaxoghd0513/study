import datetime
path = './'
# 데이터 일시 문자열
data_time_str = 

# 문자열을 datetime 객체로 변환
data_time_obj = datetime.datetime.strptime(data_time_str, '%Y-%m-%d %H:%M:%S')

# 년도 추출
year = data_time_obj.year

# 시간 추출
hour = data_time_obj.hour
minute = data_time_obj.minute
second = data_time_obj.second

# 결과 출력
print("년도:", year)
print("시간:", hour, ":", minute, ":", second)