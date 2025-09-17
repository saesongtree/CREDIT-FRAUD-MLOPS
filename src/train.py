# --- 전체 전처리 파이프라인 코드 ---

import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# 1. 데이터 로드 및 스케일링
df = pd.read_csv('data/creditcard.csv')
scaler = RobustScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# 2. 데이터 분리
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. 오버샘플링 (학습 데이터에만 적용)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 이제 이 데이터를 사용하여 모델을 학습하고 평가합니다.
# model.fit(X_train_resampled, y_train_resampled)
# evaluation = model.evaluate(X_test, y_test)