# 필요한 라이브러리들을 모두 불러옵니다.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier # LightGBM 모델 추가
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import argparse # argparse 라이브러리 추가

mlflow.set_experiment("logistic")
# mlflow.set_tracking_uri("http://localhost:5000") # 로컬 서버 사용 시 주석 해제

def main(args):
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
    run_name = f"{args.model_name}_n_est_{args.n_estimators}_lr_{args.learning_rate}"
    with mlflow.start_run(run_name=run_name):
        print(f"MLflow Run 시작: {run_name}")

        # 파라미터 로깅
        mlflow.log_params(vars(args))
        mlflow.log_param("sampling_method", "SMOTE")
        mlflow.log_param("scaling_method", "RobustScaler")
        
        # 모델 선택 및 정의
        if args.model_name == 'logistic_regression':
            model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
        elif args.model_name == 'lgbm':
            model = LGBMClassifier(
                random_state=42,
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate
            )
        else:
            raise ValueError(f"지원하지 않는 모델 이름입니다: {args.model_name}")

        # 3. 모델 학습
        model.fit(X_train_resampled, y_train_resampled)

        # 4. 모델 평가
        preds = model.predict(X_test)
        preds_proba = model.predict_proba(X_test)[:, 1]
        
        auprc = average_precision_score(y_test, preds_proba)
        recall = recall_score(y_test, preds)
        precision = precision_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        print("\n--- 모델 평가 결과 ---")
        print(f"AUPRC: {auprc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1-Score: {f1:.4f}")
        
        # 5. 평가지표 로깅
        mlflow.log_metric("auprc", auprc)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score", f1)
        
        # 모델 로깅
        mlflow.sklearn.log_model(model, "model")
        print("MLflow 로깅 완료.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection MLOps Pipeline")
    parser.add_argument("--model_name", type=str, default="logistic_regression", help="사용할 모델 이름: logistic_regression 또는 lgbm")
    parser.add_argument("--n_estimators", type=int, default=100, help="LightGBM의 n_estimators 하이퍼파라미터")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="LightGBM의 learning_rate 하이퍼파라미터")
    args = parser.parse_args()
    main(args)

# 사용법
# python src/train.py --model_name lgbm --n_estimators 150 --learning_rate 0.05