# 필요한 라이브러리들을 모두 불러옵니다.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, recall_score, precision_score
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000")

def main():
    """
    메인 함수: 데이터 로드, 전처리, 모델 학습, 평가 및 MLflow 로깅을 수행합니다.
    """
    print("스크립트 실행 시작...")

    # --- 1. 데이터 로드 및 전처리 ---
    try:
        df = pd.read_csv('data/creditcard.csv')
    except FileNotFoundError:
        print("에러: '../data/creditcard.csv' 파일을 찾을 수 없습니다.")
        return

    scaler = RobustScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("데이터 전처리 완료.")

    # --- 2. MLflow 실험 시작 ---
    # with 블록 안의 모든 내용이 하나의 실험(Run)으로 기록됩니다.
    with mlflow.start_run():
        print("MLflow Run 시작...")

        # 사용할 모델과 파라미터를 정의합니다.
        model_params = {
            "random_state": 42,
            "max_iter": 1000,
            "solver": "liblinear" # 수렴을 돕는 solver
        }
        model = LogisticRegression(**model_params)

        # 3. 모델 학습
        print("모델 학습 시작...")
        model.fit(X_train_resampled, y_train_resampled)
        print("모델 학습 완료.")

        # 4. 모델 평가
        # predict()는 0 또는 1을, predict_proba()는 확률을 반환합니다.
        preds = model.predict(X_test)
        preds_proba = model.predict_proba(X_test)[:, 1]
        
        # 평가지표 계산
        auprc = average_precision_score(y_test, preds_proba) # AUPRC는 확률값을 사용
        recall = recall_score(y_test, preds)
        precision = precision_score(y_test, preds)

        print("\n--- 모델 평가 결과 ---")
        print(f"AUPRC: {auprc:.4f}")
        print(f"Recall (재현율): {recall:.4f}")
        print(f"Precision (정밀도): {precision:.4f}")
        print("--------------------")

        # --- 5. MLflow에 모든 결과 기록 ---
        print("MLflow에 결과 기록 시작...")
        
        # 파라미터 기록
        mlflow.log_params(model_params)
        mlflow.log_param("sampling_method", "SMOTE")
        
        # 평가지표 기록
        mlflow.log_metric("auprc", auprc)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)

        # 학습된 모델 기록
        mlflow.sklearn.log_model(model, "model")
        
        print("MLflow 로깅 완료.")


if __name__ == "__main__":
    main()