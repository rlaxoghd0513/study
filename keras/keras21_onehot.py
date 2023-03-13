#[과제]
# 3가지 원핫인코딩 방식을 비교할것
 
#1 pandas 의 get_dummies
#2 keras 의 to_categorical
#3 sklearn의 OneHotEncoder

#미세한 차이를 정리하시오

#keras 의 to_categorical은 무조건 0부터 시작한다. 그래서 카테고리컬을 쓸땐 y라벨이 0부터 시작하지 않을 경우 0을 빼주는 코드를 입력해야한다
#sklearn의 OneHotEncoder는 train 데이터의 특성을 학습할 수 있다
#pandas의 get_dummies는 train 데이터의 특성을 학습하지 않기 때문에 train 데이터에만 있고 test 데이터에는 없는 카테고리를 test 데이터에서 원핫인코딩 된 칼럼으로 바꿔주지 않는다.