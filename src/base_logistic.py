# 필요한 라이브러리들을 모두 불러옵니다.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, recall_score, precision_score
from sklearn.metrics import f1_score
import mlflow
import mlflow.sklearn

# 실험 그룹 이름을 "logistic"으로 설정합니다.
mlflow.set_experiment("logistic")
# MLflow 서버 주소를 설정합니다.
mlflow.set_tracking_uri("http://localhost:5000")

def main():
    """
    메인 함수: 데이터 로드, 전처리, 모델 학습, 평가 및 MLflow 로깅을 수행합니다.
    """
    print("스크립트 실행 시작 (베이스라인 모델)...")

    # --- 1. 데이터 로드 및 전처리 ---
    try:
        df = pd.read_csv('data/creditcard.csv')
    except FileNotFoundError:
        print("에러: 'data/creditcard.csv' 파일을 찾을 수 없습니다.")
        return

    # Time, Amount 컬럼이 그대로 유지됩니다.

    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("데이터 전처리 완료 (전처리 없음).")

    # --- 2. MLflow 실험 시작 ---
    # with 블록 안의 모든 내용이 하나의 실험(Run)으로 기록됩니다.
    # run_name을 변경하여 UI에서 구분하기 쉽게 합니다.
    with mlflow.start_run(run_name="LOGISTIC_BASELINE"):
        print("MLflow Run 시작...")

        # 사용할 모델과 파라미터를 정의합니다.
        model_params = {
            "random_state": 42,
            "max_iter": 1000,
            "solver": "liblinear"
        }
        model = LogisticRegression(**model_params)

        # 3. 모델 학습
        print("모델 학습 시작...")
        # 원본 학습 데이터(X_train, y_train)로 모델을 학습시킵니다.
        model.fit(X_train, y_train)
        print("모델 학습 완료.")

        # 4. 모델 평가
        preds = model.predict(X_test)
        preds_proba = model.predict_proba(X_test)[:, 1]
        
        auprc = average_precision_score(y_test, preds_proba)
        recall = recall_score(y_test, preds)
        precision = precision_score(y_test, preds)
        f1 = f1_score(y_test, preds) # preds는 model.predict(X_test)의 결과

        print("\n--- 모델 평가 결과 ---")
        print(f"AUPRC: {auprc:.4f}")
        print(f"Recall (재현율): {recall:.4f}")
        print(f"Precision (정밀도): {precision:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("--------------------")
        
        # --- 5. MLflow에 모든 결과 기록 ---
        print("MLflow에 결과 기록 시작...")
        
        mlflow.log_params(model_params)
        # sampling_method 파라미터를 "None"으로 기록하여 전처리 안 했음을 명시합니다.
        mlflow.log_param("sampling_method", "None")
        mlflow.log_param("scaling_method", "None")
        
        mlflow.log_metric("auprc", auprc)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")
        
        print("MLflow 로깅 완료.")


if __name__ == "__main__":
    main()