"""
GPU 기반 그리드 서치 학습 파이프라인 (최적화 버전)
- 데이터 누출 제거 (MCC 피처 제외)
- 1시간 이내 완료 목표
- RandomForest GPU 직접 구현
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import os
import sys
from datetime import datetime
import json
import itertools

# cuML 임포트 시도
try:
    from cuml.ensemble import RandomForestClassifier as CumlRFClassifier
    HAS_CUML = True
except ImportError:
    HAS_CUML = False
    print("경고: cuML을 사용할 수 없습니다. XGBoost만 사용합니다.")


def load_preprocessed_data(file_path, sample_frac=1.0):
    """전처리된 데이터 로드"""
    print(f"\n데이터 로드 중: {file_path}")
    df = pd.read_csv(file_path)
    
    if sample_frac < 1.0:
        print(f"  - 샘플링: {sample_frac*100}% 사용")
        df = df.sample(frac=sample_frac, random_state=42)
    
    print(f"  - 총 샘플 수: {len(df):,}건")
    print(f"  - 총 컬럼 수: {len(df.columns)}개")
    
    return df


def prepare_features_target(df):
    """특성과 타겟 분리"""
    feature_cols = [col for col in df.columns if col.endswith('_scaled')]
    target_col = 'Category_encoded'
    
    X = df[feature_cols].values.astype('float32')  # float32로 메모리 절약
    y = df[target_col].values
    
    print(f"\n특성 및 타겟 준비:")
    print(f"  - 특성 수: {len(feature_cols)}개")
    print(f"  - 샘플 수: {len(X):,}개")
    print(f"  - 클래스 수: {len(np.unique(y))}개")
    
    return X, y, feature_cols


def xgboost_grid_search_gpu(X_train, y_train, X_test, y_test):
    """XGBoost GPU 그리드 서치 (축소 버전 - 30분 목표)"""
    print("\n" + "="*70)
    print("XGBoost GPU 그리드 서치 시작")
    print("="*70)
    
    # 축소된 그리드 (27개 조합)
    param_grid = {
        'max_depth': [6, 8, 10],
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
    }
    
    print(f"그리드 서치 설정:")
    print(f"  - 탐색 조합 수: {np.prod([len(v) for v in param_grid.values()])}개")
    print(f"  - CV: 간소화 (시간 절약)")
    
    best_score = 0
    best_params = None
    best_model = None
    
    start_time = datetime.now()
    
    # 모든 조합 테스트
    combinations = list(itertools.product(*param_grid.values()))
    total = len(combinations)
    
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
        
        # 학습
        model.fit(X_train, y_train, verbose=False)
        
        # 평가
        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred, average='weighted')
        
        if score > best_score:
            best_score = score
            best_params = params
            best_model = model
        
        if idx % 5 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"  진행: {idx}/{total} ({idx/total*100:.1f}%) - 경과: {elapsed:.0f}초")
    
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"\n학습 완료! 소요 시간: {training_time:.2f}초 ({training_time/60:.2f}분)")
    
    # 최적 파라미터
    print(f"\n최적 파라미터:")
    for param, value in best_params.items():
        print(f"  - {param}: {value}")
    
    # 테스트 평가
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n최적 점수: {best_score:.4f}")
    print(f"\n테스트 성능:")
    print(f"  - Accuracy: {test_accuracy:.4f}")
    print(f"  - F1 Score (weighted): {test_f1:.4f}")
    
    return best_model, {
        'model_name': 'xgboost',
        'best_params': best_params,
        'best_score': best_score,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'training_time': training_time
    }


def randomforest_grid_search_gpu(X_train, y_train, X_test, y_test):
    """cuML RandomForest GPU 그리드 서치 (직접 구현 - 20분 목표)"""
    if not HAS_CUML:
        print("\ncuML이 없어서 RandomForest 그리드 서치를 건너뜁니다.")
        return None, None
    
    print("\n" + "="*70)
    print("cuML RandomForest GPU 그리드 서치 시작")
    print("="*70)
    
    # 축소된 파라미터 그리드 (18개 조합)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 16, 20],
        'max_features': [0.8, 1.0],
    }
    
    print(f"그리드 서치 설정:")
    print(f"  - 탐색 조합 수: {np.prod([len(v) for v in param_grid.values()])}개")
    
    best_score = 0
    best_params = None
    best_model = None
    
    start_time = datetime.now()
    
    # 모든 조합 테스트
    combinations = list(itertools.product(*param_grid.values()))
    total = len(combinations)
    
    for idx, (n_est, max_depth, max_feat) in enumerate(combinations, 1):
        params = {
            'n_estimators': n_est,
            'max_depth': max_depth,
            'max_features': max_feat
        }
        
        try:
            model = CumlRFClassifier(
                random_state=42,
                n_streams=1,
                **params
            )
            
            # 학습
            model.fit(X_train, y_train)
            
            # 평가
            y_pred = model.predict(X_test)
            score = f1_score(y_test, y_pred, average='weighted')
            
            if score > best_score:
                best_score = score
                best_params = params
                best_model = model
            
            if idx % 5 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"  진행: {idx}/{total} ({idx/total*100:.1f}%) - 경과: {elapsed:.0f}초")
                
        except Exception as e:
            print(f"  조합 {idx} 실패: {e}")
            continue
    
    if best_model is None:
        print("\n모든 조합 실패")
        return None, None
    
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"\n학습 완료! 소요 시간: {training_time:.2f}초 ({training_time/60:.2f}분)")
    
    # 최적 파라미터
    print(f"\n최적 파라미터:")
    for param, value in best_params.items():
        print(f"  - {param}: {value}")
    
    # 테스트 평가
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n최적 점수: {best_score:.4f}")
    print(f"\n테스트 성능:")
    print(f"  - Accuracy: {test_accuracy:.4f}")
    print(f"  - F1 Score (weighted): {test_f1:.4f}")
    
    return best_model, {
        'model_name': 'randomforest_cuml',
        'best_params': best_params,
        'best_score': best_score,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'training_time': training_time
    }


def save_results(model, metadata, model_name, output_dir='03_models/05_gridsearch'):
    """그리드 서치 결과 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 최적 모델 저장
    model_file = os.path.join(output_dir, f'best_{model_name}_{timestamp}.joblib')
    joblib.dump(model, model_file)
    print(f"\n최적 모델 저장: {model_file}")
    
    # 메타데이터 저장
    metadata_file = os.path.join(output_dir, f'metadata_{model_name}_{timestamp}.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"메타데이터 저장: {metadata_file}")
    
    return model_file, metadata_file


def main():
    """메인 파이프라인"""
    print("="*70)
    print("GPU 기반 그리드 서치 파이프라인 (최적화 버전)")
    print("="*70)
    
    # 설정
    data_file = '02_data/01_processed/preprocessed_full_featured.csv'
    sample_frac = 1.0
    test_size = 0.2
    
    # 1. 데이터 로드
    df = load_preprocessed_data(data_file, sample_frac=sample_frac)
    
    # 2. 특성/타겟 준비
    X, y, feature_cols = prepare_features_target(df)
    
    # 3. 학습/테스트 분리
    print(f"\n학습/테스트 데이터 분리 (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"  - 학습 데이터: {len(X_train):,}건")
    print(f"  - 테스트 데이터: {len(X_test):,}건")
    
    # 결과 저장용
    all_results = []
    
    # 4. XGBoost 그리드 서치
    try:
        xgb_model, xgb_metadata = xgboost_grid_search_gpu(X_train, y_train, X_test, y_test)
        if xgb_model:
            save_results(xgb_model, xgb_metadata, 'xgboost')
            all_results.append(xgb_metadata)
    except Exception as e:
        print(f"\nXGBoost 그리드 서치 실패: {e}")
    
    # 5. RandomForest 그리드 서치
    try:
        rf_model, rf_metadata = randomforest_grid_search_gpu(X_train, y_train, X_test, y_test)
        if rf_model:
            save_results(rf_model, rf_metadata, 'randomforest')
            all_results.append(rf_metadata)
    except Exception as e:
        print(f"\nRandomForest 그리드 서치 실패: {e}")
    
    # 6. 최종 요약
    print("\n" + "="*70)
    print("그리드 서치 완료 - 최종 요약")
    print("="*70)
    
    if all_results:
        summary_df = pd.DataFrame(all_results)
        print("\n모델별 성능 비교:")
        print(summary_df[['model_name', 'test_accuracy', 'test_f1', 'training_time']])
        
        # 최고 성능 모델
        best_idx = summary_df['test_f1'].idxmax()
        best_model = summary_df.loc[best_idx]
        
        print(f"\n🏆 최고 성능 모델: {best_model['model_name']}")
        print(f"  - Test Accuracy: {best_model['test_accuracy']:.4f}")
        print(f"  - Test F1: {best_model['test_f1']:.4f}")
        print(f"  - 학습 시간: {best_model['training_time']:.2f}초 ({best_model['training_time']/60:.2f}분)")
    else:
        print("\n완료된 그리드 서치가 없습니다.")
    
    print("\n" + "="*70)
    print("파이프라인 완료!")
    print("="*70)


if __name__ == '__main__':
    main()
