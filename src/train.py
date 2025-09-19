# 필요한 라이브러리들을 모두 불러옵니다.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier # LightGBM 모델 추가
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from catboost import CatBoostClassifier
import mlflow
import mlflow.sklearn
import argparse

mlflow.set_experiment("credit_fraud_experiment")
mlflow.set_tracking_uri("http://localhost:5000") # 로컬 서버 사용 시 주석 해제

def main(args):
    """
    메인 함수: 데이터 로드, 전처리, 모델 학습, 평가 및 MLflow 로깅을 수행합니다.
    """
    print("스크립트 실행 시작...")

    # --- 1. 데이터 로드 및 전처리 ---
    try:
        df = pd.read_csv('data/creditcard.csv')
    except FileNotFoundError:
        print("에러: 'data/creditcard.csv' 파일을 찾을 수 없습니다.")
        return

    scaler = RobustScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    if args.model_name != 'isolation_forest':
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print("데이터 전처리 완료.")
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
        print("Isolation Forest는 비지도 학습이므로 SMOTE를 적용하지 않습니다.")

    # --- 2. MLflow 실험 시작 ---
    run_name = f"{args.model_name}_run"
    with mlflow.start_run(run_name=run_name):
        print(f"MLflow Run 시작: {run_name}")

        # 파라미터 로깅 (모델별로 사용된 파라미터만 선별하여 기록)
        params = {"model_name": args.model_name}

        if args.model_name == 'lgbm':
            params["n_estimators"] = args.n_estimators
            params["learning_rate"] = args.learning_rate
            params["num_leaves"] = args.num_leaves
            params["max_depth"] = args.max_depth
            params["min_child_samples"] = args.min_child_samples
            params["subsample"] = args.subsample
            params["colsample_bytree"] = args.colsample_bytree
        elif args.model_name == 'xgboost':
            params["max_depth"] = args.max_depth
        elif args.model_name == 'random_forest':
            params["n_estimators"] = args.n_estimators
        elif args.model_name == 'catboost':
            params["n_estimators"] = args.n_estimators
            params["learning_rate"] = args.learning_rate
        elif args.model_name == 'isolation_forest':
            params["n_estimators"] = args.n_estimators
            params["contamination"] = args.contamination
        
        if args.model_name != 'isolation_forest':
            params["sampling_method"] = "SMOTE"
        params["scaling_method"] = "RobustScaler"
        
        mlflow.log_params(params)
        
        # 모델 선택 및 정의
        if args.model_name == 'logistic_regression':
            model = LogisticRegression(
                random_state=42, 
                max_iter=1000,
                solver='liblinear')
        elif args.model_name == 'lgbm':
            model = LGBMClassifier(
                random_state=42, 
                n_estimators=args.n_estimators, 
                learning_rate=args.learning_rate,
                num_leaves=args.num_leaves,
                max_depth=args.max_depth,
                min_child_samples=args.min_child_samples,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree
                )
        elif args.model_name == 'xgboost':
            model = XGBClassifier(
                random_state=42, 
                max_depth=args.max_depth,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                gamma=args.gamma, 
                use_label_encoder=False, 
                eval_metric='logloss'
                )
        elif args.model_name == 'random_forest': 
            model = RandomForestClassifier(
                random_state=42, 
                n_estimators=args.n_estimators,
                max_depth=args.max_depth if args.max_depth else None, # None 처리
                min_samples_split=args.min_samples_split,
                min_samples_leaf=args.min_samples_leaf,
                n_jobs=-1 # CPU를 모두 사용하여 학습 속도 향상
                )
        elif args.model_name == 'catboost': 
            model = CatBoostClassifier(
                random_state=42, 
                n_estimators=args.n_estimators, 
                learning_rate=args.learning_rate, 
                depth=args.depth,
                l2_leaf_reg=args.l2_leaf_reg,
                verbose=0
                )
        elif args.model_name == 'isolation_forest': 
            model = IsolationForest(
                random_state=42, 
                n_estimators=args.n_estimators, 
                contamination=args.contamination,
                max_samples=args.max_samples,
                max_features=args.max_features,
                n_jobs=-1
                )
        else:
            raise ValueError(f"지원하지 않는 모델 이름입니다: {args.model_name}")

        # 3. 모델 학습
        model.fit(X_train_resampled, y_train_resampled)

        # 4. 모델 평가
        if args.model_name == 'isolation_forest':
            preds_raw = model.predict(X_test)
            preds = [0 if p == 1 else 1 for p in preds_raw]
            scores = model.decision_function(X_test)
            preds_proba = max(scores) - scores 
        else:
            preds = model.predict(X_test)
            preds_proba = model.predict_proba(X_test)[:, 1]
        
        auprc = average_precision_score(y_test, preds_proba)
        recall = recall_score(y_test, preds)
        precision = precision_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        print("\n--- 모델 평가 결과 ---")
        print(f"[테스트 데이터] AUPRC: {auprc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1-Score: {f1:.4f}")
        
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

    # --- 공통 파라미터 ---
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="logistic_regression", 
        help="사용할 모델 이름: logistic_regression, lgbm, xgboost, random_forest, catboost, isolation_forest"
    )

    # --- 앙상블 모델 공유 파라미터 ---
    parser.add_argument(
        "--n_estimators", 
        type=int, 
        default=100, 
        help="LGBM, XGBoost, RandomForest, CatBoost, IsolationForest에서 사용할 나무의 개수"
    )

    # --- 부스팅 모델 공유 파라미터 (LGBM, XGBoost, CatBoost) ---
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=0.1, 
        help="LGBM, XGBoost, CatBoost의 학습률"
    )

    # --- LightGBM 파라미터 ---
    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--min_child_samples", type=int, default=20)
    
    # --- XGBoost & LightGBM 공유 파라미터 ---
    parser.add_argument("--subsample", type=float, default=1.0, help="데이터 샘플링 비율")
    parser.add_argument("--colsample_bytree", type=float, default=1.0, help="피처 샘플링 비율")

    # --- XGBoost 파라미터 ---
    parser.add_argument("--max_depth", type=int, default=3, help="XGBoost 트리의 최대 깊이")
    parser.add_argument("--gamma", type=float, default=0, help="XGBoost 분기 제어")

    # --- CatBoost 파라미터 ---
    parser.add_argument("--depth", type=int, default=6, help="CatBoost 트리의 깊이")
    parser.add_argument("--l2_leaf_reg", type=int, default=3, help="CatBoost L2 규제")

    # --- Random Forest 파라미터 ---
    parser.add_argument("--min_samples_split", type=int, default=2, help="RandomForest 분기를 위한 최소 샘플 수")
    parser.add_argument("--min_samples_leaf", type=int, default=1, help="RandomForest 리프 노드의 최소 샘플 수")

    # --- Isolation Forest 파라미터 ---
    parser.add_argument("--contamination", type=float, default=0.0017, help="예상되는 이상치(사기)의 비율")
    parser.add_argument("--max_samples", type=float, default=1.0, help="각 트리를 학습시킬 때 사용할 샘플의 비율")
    parser.add_argument("--max_features", type=float, default=1.0, help="각 트리를 학습시킬 때 사용할 피처의 비율")
    
    args = parser.parse_args()
    main(args)