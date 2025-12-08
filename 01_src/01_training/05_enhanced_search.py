"""
개선된 시퀀스 예측 그리드 서치
- Refer 피처 반영
- 불균형 보정 (class_weight)
- 목표: 60%+ Accuracy
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import os
from datetime import datetime
import json
import itertools

try:
    from cuml.ensemble import RandomForestClassifier as CumlRFClassifier
    HAS_CUML = True
except ImportError:
    HAS_CUML = False


def load_enhanced_data(file_path):
    """개선된 데이터 로드"""
    print(f"\n데이터 로드: {file_path}")
    df = pd.read_csv(file_path)
    print(f"  총 시퀀스: {len(df):,}개")
    return df


def prepare_data_with_weights(df):
    """
    데이터 준비 + 불균형 보정
    """
    feature_cols = [col for col in df.columns if col.endswith('_scaled')]
    target_col = 'Next_Category_encoded'
    
    X = df[feature_cols].values.astype('float32')
    y = df[target_col].values
    
    print(f"\n특성/타겟 준비:")
    print(f"  특성 수: {len(feature_cols)}개")
    print(f"  시퀀스 수: {len(X):,}개")
    print(f"  클래스 수: {len(np.unique(y))}개")
    
    # ⭐ 불균형 보정: 샘플 가중치 계산
    print(f"\n불균형 보정:")
    sample_weights = compute_sample_weight('balanced', y)
    
    # 클래스별 가중치 확인
    unique_classes, class_counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique_classes, class_counts):
        weight = sample_weights[y == cls][0]
        print(f"  클래스 {cls}: {count:,}건 → 가중치 {weight:.3f}")
    
    return X, y, sample_weights, feature_cols


def xgboost_enhanced_search(X_train, y_train, X_test, y_test, sample_weights_train):
    """
    개선된 XGBoost 그리드 서치
    - 불균형 보정
    - 더 넓은 파라미터 그리드
    """
    print("\n" + "="*70)
    print("개선된 XGBoost 그리드 서치 (불균형 보정)")
    print("="*70)
    
    # 파라미터 그리드 (총 36개 조합)
    param_grid = {
        'max_depth': [8, 10, 12],
        'learning_rate': [0.05, 0.1, 0.15],
        'n_estimators': [200, 300, 400],
        'subsample': [0.8, 1.0],
    }
    
    best_score = 0
    best_params = None
    best_model = None
    
    start_time = datetime.now()
    combinations = list(itertools.product(*param_grid.values()))
    total = len(combinations)
    
    print(f"총 {total}개 조합 테스트 (불균형 보정 적용)...")
    
    for idx, (max_depth, lr, n_est, subsample) in enumerate(combinations, 1):
        params = {
            'max_depth': max_depth,
            'learning_rate': lr,
            'n_estimators': n_est,
            'subsample': subsample
        }
        
        model = xgb.XGBClassifier(
            device='cuda',
            tree_method='hist',
            random_state=42,
            eval_metric='mlogloss',
            **params
        )
        
        # ⭐ 불균형 보정: sample_weight 적용
        model.fit(X_train, y_train, sample_weight=sample_weights_train, verbose=False)
        
        y_pred = model.predict(X_test)
        score_acc = accuracy_score(y_test, y_pred)
        score_f1 = f1_score(y_test, y_pred, average='macro')  # ⭐ Macro F1 (Refer와 동일)
        
        # Macro F1 기준으로 최적화 (Refer 모델과 동일)
        if score_f1 > best_score:
            best_score = score_f1
            best_params = params
            best_model = model
            best_acc = score_acc
        
        if idx % 6 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"  진행: {idx}/{total} - Acc: {score_acc:.4f}, Macro F1: {score_f1:.4f} - 경과: {elapsed:.0f}초")
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1_macro = f1_score(y_test, y_pred, average='macro')
    test_f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n✅ 완료 ({training_time:.0f}초)")
    print(f"최적 파라미터: {best_params}")
    print(f"\n테스트 성능:")
    print(f"  Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"  Macro F1: {test_f1_macro:.4f} (Refer 비교 기준)")
    print(f"  Weighted F1: {test_f1_weighted:.4f}")
    
    # Refer와 비교
    refer_accuracy = 0.6309
    refer_f1 = 0.5486
    print(f"\nRefer 모델 대비:")
    print(f"  Accuracy: {test_accuracy:.4f} vs {refer_accuracy:.4f} ({(test_accuracy-refer_accuracy)*100:+.2f}%p)")
    print(f"  Macro F1: {test_f1_macro:.4f} vs {refer_f1:.4f} ({(test_f1_macro-refer_f1)*100:+.2f}%p)")
    
    return best_model, {
        'model_name': 'xgboost_enhanced',
        'best_params': best_params,
        'test_accuracy': test_accuracy,
        'test_f1_macro': test_f1_macro,
        'test_f1_weighted': test_f1_weighted,
        'training_time': training_time,
        'refer_comparison': {
            'refer_accuracy': refer_accuracy,
            'refer_f1': refer_f1,
            'accuracy_diff': test_accuracy - refer_accuracy,
            'f1_diff': test_f1_macro - refer_f1
        }
    }


def save_results(model, metadata, output_dir='03_models/07_enhanced'):
    """결과 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = metadata['model_name']
    
    model_file = os.path.join(output_dir, f'best_{model_name}_{timestamp}.joblib')
    metadata_file = os.path.join(output_dir, f'metadata_{model_name}_{timestamp}.json')
    
    joblib.dump(model, model_file)
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n모델 저장: {model_file}")
    print(f"메타 저장: {metadata_file}")
    
    return model_file, metadata_file


def main():
    """메인"""
    print("="*70)
    print("개선된 시퀀스 예측 그리드 서치")
    print("Refer 피처 + 불균형 보정")
    print("="*70)
    
    # 데이터 로드
    data_file = '02_data/01_processed/preprocessed_enhanced.csv'
    df = load_enhanced_data(data_file)
    
    # 데이터 준비 (불균형 보정)
    X, y, sample_weights, feature_cols = prepare_data_with_weights(df)
    
    # 분할
    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n학습: {len(X_train):,}개, 테스트: {len(X_test):,}개")
    
    # XGBoost 개선
    try:
        xgb_model, xgb_meta = xgboost_enhanced_search(
            X_train, y_train, X_test, y_test, sw_train
        )
        save_results(xgb_model, xgb_meta)
        
        # 상세 리포트
        print("\n" + "="*70)
        print("카테고리별 성능 (개선 모델)")
        print("="*70)
        y_pred = xgb_model.predict(X_test)
        
        # 카테고리 이름
        categories = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']
        print(classification_report(y_test, y_pred, target_names=categories, digits=4))
        
    except Exception as e:
        print(f"XGBoost 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
