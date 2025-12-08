"""
시퀀스 예측 GPU 그리드 서치
X: 현재 거래 카테고리 + 특성
Y: 다음 거래 카테고리
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
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


def load_sequence_data(file_path, sample_frac=1.0):
    """시퀀스 데이터 로드"""
    print(f"\n데이터 로드: {file_path}")
    df = pd.read_csv(file_path)
    
    if sample_frac < 1.0:
        print(f"  샘플링: {sample_frac*100}%")
        df = df.sample(frac=sample_frac, random_state=42)
    
    print(f"  총 시퀀스: {len(df):,}개")
    return df


def prepare_sequence_data(df):
    """시퀀스 데이터 준비"""
    feature_cols = [col for col in df.columns if col.endswith('_scaled')]
    target_col = 'Next_Category_encoded'
    
    X = df[feature_cols].values.astype('float32')
    y = df[target_col].values
    
    print(f"\n특성/타겟 준비:")
    print(f"  특성 수: {len(feature_cols)}개")
    print(f"  시퀀스 수: {len(X):,}개")
    print(f"  클래스 수: {len(np.unique(y))}개")
    
    # 현재 카테고리가 특성에 포함됐는지 확인
    current_cat_features = [f for f in feature_cols if 'Current_Category' in f]
    print(f"  ✓ 현재 카테고리 피처: {current_cat_features}")
    
    return X, y, feature_cols


def xgboost_sequence_search(X_train, y_train, X_test, y_test):
    """XGBoost 시퀀스 예측"""
    print("\n" + "="*70)
    print("XGBoost 시퀀스 예측 그리드 서치")
    print("="*70)
    
    param_grid = {
        'max_depth': [6, 8, 10],
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
    }
    
    best_score = 0
    best_params = None
    best_model = None
    
    start_time = datetime.now()
    combinations = list(itertools.product(*param_grid.values()))
    total = len(combinations)
    
    print(f"총 {total}개 조합 테스트...")
    
    for idx, (max_depth, lr, n_est) in enumerate(combinations, 1):
        params = {
            'max_depth': max_depth,
            'learning_rate': lr,
            'n_estimators': n_est
        }
        
        model = xgb.XGBClassifier(
            device='cuda',
            tree_method='hist',
            random_state=42,
            eval_metric='mlogloss',
            **params
        )
        
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred, average='weighted')
        
        if score > best_score:
            best_score = score
            best_params = params
            best_model = model
        
        if idx % 5 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"  진행: {idx}/{total} - F1: {score:.4f} - 경과: {elapsed:.0f}초")
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n✅ 완료 ({training_time:.0f}초)")
    print(f"최적 파라미터: {best_params}")
    print(f"테스트 Accuracy: {test_accuracy:.4f}")
    print(f"테스트 F1: {test_f1:.4f}")
    
    return best_model, {
        'model_name': 'xgboost_sequence',
        'best_params': best_params,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'training_time': training_time
    }


def randomforest_sequence_search(X_train, y_train, X_test, y_test):
    """RandomForest 시퀀스 예측"""
    if not HAS_CUML:
        return None, None
    
    print("\n" + "="*70)
    print("RandomForest 시퀀스 예측 그리드 서치")
    print("="*70)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 16, 20],
        'max_features': [0.8, 1.0],
    }
    
    best_score = 0
    best_params = None
    best_model = None
    
    start_time = datetime.now()
    combinations = list(itertools.product(*param_grid.values()))
    total = len(combinations)
    
    print(f"총 {total}개 조합 테스트...")
    
    for idx, (n_est, max_depth, max_feat) in enumerate(combinations, 1):
        params = {
            'n_estimators': n_est,
            'max_depth': max_depth,
            'max_features': max_feat
        }
        
        try:
            model = CumlRFClassifier(random_state=42, n_streams=1, **params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = f1_score(y_test, y_pred, average='weighted')
            
            if score > best_score:
                best_score = score
                best_params = params
                best_model = model
            
            if idx % 3 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"  진행: {idx}/{total} - F1: {score:.4f} - 경과: {elapsed:.0f}초")
        except Exception as e:
            print(f"  조합 {idx} 실패: {e}")
    
    if best_model is None:
        return None, None
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n✅ 완료 ({training_time:.0f}초)")
    print(f"최적 파라미터: {best_params}")
    print(f"테스트 Accuracy: {test_accuracy:.4f}")
    print(f"테스트 F1: {test_f1:.4f}")
    
    return best_model, {
        'model_name': 'randomforest_sequence',
        'best_params': best_params,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'training_time': training_time
    }


def save_results(model, metadata, output_dir='03_models/06_sequence'):
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
    print("시퀀스 예측 GPU 그리드 서치")
    print("현재 카테고리 → 다음 카테고리 예측")
    print("="*70)
    
    # 데이터 로드
    data_file = '02_data/01_processed/preprocessed_sequence.csv'
    df = load_sequence_data(data_file, sample_frac=1.0)
    
    # 데이터 준비
    X, y, feature_cols = prepare_sequence_data(df)
    
    # 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n학습: {len(X_train):,}개, 테스트: {len(X_test):,}개")
    
    all_results = []
    
    # XGBoost
    try:
        xgb_model, xgb_meta = xgboost_sequence_search(X_train, y_train, X_test, y_test)
        if xgb_model:
            save_results(xgb_model, xgb_meta)
            all_results.append(xgb_meta)
    except Exception as e:
        print(f"XGBoost 실패: {e}")
    
    # RandomForest
    try:
        rf_model, rf_meta = randomforest_sequence_search(X_train, y_train, X_test, y_test)
        if rf_model:
            save_results(rf_model, rf_meta)
            all_results.append(rf_meta)
    except Exception as e:
        print(f"RandomForest 실패: {e}")
    
    # 요약
    print("\n" + "="*70)
    print("최종 결과")
    print("="*70)
    
    if all_results:
        df_results = pd.DataFrame(all_results)
        print("\n성능 비교:")
        print(df_results[['model_name', 'test_accuracy', 'test_f1', 'training_time']])
        
        best_idx = df_results['test_f1'].idxmax()
        best = df_results.loc[best_idx]
        print(f"\n🏆 최고 성능: {best['model_name']}")
        print(f"  Accuracy: {best['test_accuracy']:.4f}")
        print(f"  F1: {best['test_f1']:.4f}")
        print(f"  시간: {best['training_time']:.0f}초")


if __name__ == '__main__':
    main()
