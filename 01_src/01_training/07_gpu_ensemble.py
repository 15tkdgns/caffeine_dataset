"""
GPU 앙상블 모델 학습
- XGBoost (이미 완료)
- cuML RandomForest (GPU)
- 앙상블 (Voting)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import json
import os
from datetime import datetime

try:
    from cuml.ensemble import RandomForestClassifier as CumlRF
    HAS_CUML = True
except ImportError:
    HAS_CUML = False
    print("⚠️ cuML 없음, RandomForest 스킵")


def load_selected_features():
    """피처 로드"""
    feature_file = '02_data/01_processed/selected_features_enhanced.json'
    with open(feature_file, 'r', encoding='utf-8') as f:
        feature_info = json.load(f)
    
    selected_features = [f"{f}_scaled" for f in feature_info['selected_features']]
    return selected_features


def load_data(file_path, selected_features):
    """데이터 로드"""
    print(f"\n데이터 로드: {file_path}")
    df = pd.read_csv(file_path)
    
    X = df[selected_features].values.astype('float32')
    y = df['Next_Category_encoded'].values.astype('int32')
    
    print(f"  샘플: {len(X):,}개")
    print(f"  피처: {len(selected_features)}개")
    
    return X, y


def train_cuml_rf(X_train, y_train, X_test, y_test):
    """cuML RandomForest (GPU)"""
    if not HAS_CUML:
        return None, None
    
    print("\n" + "="*70)
    print("cuML RandomForest (GPU)")
    print("="*70)
    
    import cupy as cp
    
    # GPU로 데이터 전송
    print("데이터를 GPU로 전송 중...")
    X_train_gpu = cp.array(X_train)
    y_train_gpu = cp.array(y_train)
    X_test_gpu = cp.array(X_test)
    
    # 모델 생성
    model = CumlRF(
        n_estimators=200,
        max_depth=15,
        max_features=0.8,
        n_streams=4,  # GPU 병렬 스트림
        random_state=42
    )
    
    print("\n학습 시작...")
    start_time = datetime.now()
    
    model.fit(X_train_gpu, y_train_gpu)
    
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"학습 완료: {training_time:.1f}초")
    
    # 예측
    print("예측 중...")
    y_pred_gpu = model.predict(X_test_gpu)
    y_pred = cp.asnumpy(y_pred_gpu).astype(int)
    
    # CPU로 복사
    del X_train_gpu, y_train_gpu, X_test_gpu, y_pred_gpu
    
    # 평가
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n성능:")
    print(f"  Accuracy:      {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Macro F1:      {f1_macro:.4f}")
    print(f"  Weighted F1:   {f1_weighted:.4f}")
    
    return model, {
        'accuracy': acc,
        'macro_f1': f1_macro,
        'weighted_f1': f1_weighted,
        'training_time': training_time
    }


def train_voting_ensemble(X_train, y_train, X_test, y_test):
    """앙상블 (XGBoost + cuML RF)"""
    print("\n" + "="*70)
    print("Voting Ensemble (XGBoost + cuML RF)")
    print("="*70)
    
    # XGBoost
    print("\n[1/2] XGBoost 학습...")
    xgb_model = xgb.XGBClassifier(
        device='cuda',
        tree_method='hist',
        max_depth=10,
        learning_rate=0.1,
        n_estimators=200,
        random_state=42
    )
    xgb_model.fit(X_train, y_train, verbose=False)
    xgb_pred_proba = xgb_model.predict_proba(X_test)
    
    # cuML RF
    if not HAS_CUML:
        print("⚠️ cuML 없음, XGBoost만 사용")
        return xgb_model, None
    
    print("[2/2] cuML RandomForest 학습...")
    import cupy as cp
    
    X_train_gpu = cp.array(X_train)
    y_train_gpu = cp.array(y_train)
    X_test_gpu = cp.array(X_test)
    
    rf_model = CumlRF(
        n_estimators=150,
        max_depth=12,
        random_state=42
    )
    rf_model.fit(X_train_gpu, y_train_gpu)
    rf_pred_proba_gpu = rf_model.predict_proba(X_test_gpu)
    rf_pred_proba = cp.asnumpy(rf_pred_proba_gpu)
    
    del X_train_gpu, y_train_gpu, X_test_gpu, rf_pred_proba_gpu
    
    # Soft Voting
    print("\n앙상블 예측 (Soft Voting)...")
    ensemble_proba = (xgb_pred_proba + rf_pred_proba) / 2
    ensemble_pred = np.argmax(ensemble_proba, axis=1)
    
    # 평가
    acc = accuracy_score(y_test, ensemble_pred)
    f1_macro = f1_score(y_test, ensemble_pred, average='macro')
    f1_weighted = f1_score(y_test, ensemble_pred, average='weighted')
    
    print(f"\n앙상블 성능:")
    print(f"  Accuracy:      {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Macro F1:      {f1_macro:.4f}")
    print(f"  Weighted F1:   {f1_weighted:.4f}")
    
    return {
        'xgb': xgb_model,
        'rf': rf_model
    }, {
        'accuracy': acc,
        'macro_f1': f1_macro,
        'weighted_f1': f1_weighted
    }


def main():
    """메인"""
    print("="*70)
    print("GPU 앙상블 모델 학습")
    print("="*70)
    
    # 데이터 로드
    selected_features = load_selected_features()
    data_file = '02_data/01_processed/preprocessed_enhanced.csv'
    X, y = load_data(data_file, selected_features)
    
    # 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n학습: {len(X_train):,}개, 테스트: {len(X_test):,}개")
    
    results = {}
    
    # 1. cuML RandomForest
    rf_model, rf_perf = train_cuml_rf(X_train, y_train, X_test, y_test)
    if rf_perf:
        results['RandomForest'] = rf_perf
    
    # 2. Voting Ensemble
    ensemble_models, ensemble_perf = train_voting_ensemble(X_train, y_train, X_test, y_test)
    if ensemble_perf:
        results['Ensemble'] = ensemble_perf
    
    # 3. 이전 XGBoost 결과 로드
    xgb_result_file = '03_models/08_final/metadata_20251203_120028.json'
    if os.path.exists(xgb_result_file):
        with open(xgb_result_file, 'r') as f:
            xgb_meta = json.load(f)
            results['XGBoost'] = xgb_meta['performance']
    
    # 최종 비교
    print("\n" + "="*70)
    print("모델 성능 비교")
    print("="*70)
    print(f"\n{'모델':<20} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}")
    print("-"*70)
    
    for model_name, perf in results.items():
        print(f"{model_name:<20} {perf['accuracy']:>9.4f} {perf['macro_f1']:>10.4f} {perf.get('weighted_f1', 0):>12.4f}")
    
    # Refer 모델
    print("-"*70)
    print(f"{'Refer (목표)':<20} {0.6309:>9.4f} {0.5486:>10.4f} {'N/A':>12}")
    print("="*70)
    
    # 최고 모델
    best_model = max(results.items(), key=lambda x: x[1]['macro_f1'])
    print(f"\n🏆 최고 모델: {best_model[0]}")
    print(f"   Macro F1: {best_model[1]['macro_f1']:.4f}")
    
    # 모델 저장
    if rf_model:
        output_dir = '03_models/09_ensemble'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # RandomForest 저장 (pickle로)
        rf_file = os.path.join(output_dir, f'cuml_rf_{timestamp}.pkl')
        import pickle
        with open(rf_file, 'wb') as f:
            pickle.dump(rf_model, f)
        print(f"\n✅ RandomForest 저장: {rf_file}")
    
    print("\n🎯 결론:")
    print(f"  - GPU 가속으로 빠른 학습 완료")
    print(f"  - 앙상블로 성능 개선 시도")
    print(f"  - Refer 모델 갭: {(best_model[1]['macro_f1'] - 0.5486)*100:.2f}%p")


if __name__ == '__main__':
    main()
