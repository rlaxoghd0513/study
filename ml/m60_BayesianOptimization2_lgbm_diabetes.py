import numpy as np
from sklearn.datasets import load_diabetes
from bayes_opt import BayesianOptimization
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization

bayesian_params = {
    'max_depth':(3,16),
    'num_leaves':(24,64),
    'min_child_samples':(10,200),
    'min_child_weight':(1,50),
    'subsample':(0.5,1),
    'colsample_bytree':(0.5,1),
    'max_bin':(10,500),
    'reg_lambda':(0.001,10),
    'reg_alpha':(0.01,50)
}

# 데이터 로드
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# 목적함수 정의
def lgbm_cv(max_depth, num_leaves, min_child_samples, min_child_weight,
            subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    estimator = LGBMRegressor(
        objective='regression',
        n_estimators=1000,
        max_depth=int(max_depth),
        learning_rate=0.05,
        num_leaves=int(num_leaves),
        min_child_samples=int(min_child_samples),
        subsample=subsample,
        min_child_weight=int(min_child_weight),
        colsample_bytree=colsample_bytree,
        max_bin=int(max_bin),
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha
    )
    cval = -1 * cross_val_score(estimator, X, y, scoring='r2', cv=5).mean()
    return cval

# 베이지안 옵티마이저 초기화
lgbm_bo = BayesianOptimization(
    lgbm_cv,
    pbounds=bayesian_params,
    random_state=42
)

# 하이퍼파라미터 최적화 실행
lgbm_bo.maximize(
    init_points=10,
    n_iter=30
)

# 최적화된 하이퍼파라미터 출력
print(lgbm_bo.max)

# 최적화된 하이퍼파라미터를 사용하여 최종 모델 학습
best_params = lgbm_bo.max['params']
model = LGBMRegressor(
    objective='regression',
    n_estimators=1000,
    max_depth=int(best_params['max_depth']),
    learning_rate=0.05,
    num_leaves=int(best_params['num_leaves']),
    min_child_samples=int(best_params['min_child_samples']),
    subsample=best_params['subsample'],
    min_child_weight=int(best_params['min_child_weight']),
    colsample_bytree=best_params['colsample_bytree'],
    max_bin=int(best_params['max_bin']),
    reg_lambda=best_params['reg_lambda'],
    reg_alpha=best_params['reg_alpha']
)
model.fit(X, y)

# 최종 모델 평가
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(f"Best model R2 Score: {r2:.4f}")





