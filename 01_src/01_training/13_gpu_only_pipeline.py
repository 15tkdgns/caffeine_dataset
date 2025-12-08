"""
GPU 전용 모델 비교 파이프라인
모든 모델을 GPU로 실행 (CPU 모델 제외)

실행 순서:
1. 기본 환경: XGBoost, TensorFlow, CatBoost
2. cuML 환경: RandomForest GPU
3. LightGBM 환경: LightGBM CUDA
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time


def load_data():
    """데이터 로드"""
    data_file = '02_data/01_processed/preprocessed_enhanced.csv'
    
    print("="*70)
    print("데이터 로드")
    print("="*70)
    
    df = pd.read_csv(data_file)
    
    feature_cols = [col for col in df.columns if col.endswith('_scaled')]
    X = df[feature_cols].values.astype('float32')
    y = df['Next_Category_encoded'].values.astype('int32')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"총 샘플: {len(X):,}개")
    print(f"피처 수: {X.shape[1]}개")
    print(f"클래스 수: {len(np.unique(y))}개")
    print(f"학습: {len(X_train):,}, 테스트: {len(X_test):,}")
    
    return X_train, X_test, y_train, y_test, feature_cols


def train_basic_gpu_models(X_train, X_test, y_train, y_test):
    """기본 GPU 모델 (XGBoost, TensorFlow, CatBoost)"""
    results = {}
    
    # 1. XGBoost GPU
    print("\n[1/3] XGBoost (GPU)")
    try:
        import xgboost as xgb
        
        start = time.time()
        model = xgb.XGBClassifier(
            device='cuda', tree_method='hist',
            n_estimators=300, max_depth=10, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        y_pred = model.predict(X_test)
        
        results['XGBoost (GPU)'] = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'macro_f1': float(f1_score(y_test, y_pred, average='macro')),
            'weighted_f1': float(f1_score(y_test, y_pred, average='weighted')),
            'train_time': train_time,
            'device': 'GPU'
        }
        print(f"  ✅ Accuracy: {results['XGBoost (GPU)']['accuracy']:.4f} ({train_time:.1f}초)")
    except Exception as e:
        print(f"  ❌ 실패: {e}")
    
    # 2. TensorFlow NN GPU
    print("\n[2/3] TensorFlow Neural Network (GPU)")
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        
        model = keras.Sequential([
            keras.layers.Dense(256, activation='relu', input_shape=(n_features,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        start = time.time()
        model.fit(X_train, y_train, epochs=10, batch_size=1024,
                  validation_split=0.1, verbose=0)
        train_time = time.time() - start
        
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        
        results['TensorFlow NN (GPU)'] = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'macro_f1': float(f1_score(y_test, y_pred, average='macro')),
            'weighted_f1': float(f1_score(y_test, y_pred, average='weighted')),
            'train_time': train_time,
            'device': 'GPU'
        }
        print(f"  ✅ Accuracy: {results['TensorFlow NN (GPU)']['accuracy']:.4f} ({train_time:.1f}초)")
    except Exception as e:
        print(f"  ❌ 실패: {e}")
    
    # 3. CatBoost GPU
    print("\n[3/3] CatBoost (GPU)")
    try:
        from catboost import CatBoostClassifier
        
        start = time.time()
        model = CatBoostClassifier(
            task_type='GPU', devices='0',
            iterations=300, depth=10, learning_rate=0.1,
            random_state=42, verbose=False
        )
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        y_pred = model.predict(X_test)
        
        results['CatBoost (GPU)'] = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'macro_f1': float(f1_score(y_test, y_pred, average='macro')),
            'weighted_f1': float(f1_score(y_test, y_pred, average='weighted')),
            'train_time': train_time,
            'device': 'GPU'
        }
        print(f"  ✅ Accuracy: {results['CatBoost (GPU)']['accuracy']:.4f} ({train_time:.1f}초)")
    except Exception as e:
        print(f"  ❌ 실패: {e}")
    
    return results


def train_cuml_model(X_train, X_test, y_train, y_test):
    """cuML RandomForest GPU"""
    print("\n[4/5] cuML RandomForest (GPU)")
    
    try:
        import cupy as cp
        from cuml.ensemble import RandomForestClassifier
        
        # GPU 배열로 변환
        X_train_gpu = cp.array(X_train, dtype=cp.float32)
        y_train_gpu = cp.array(y_train, dtype=cp.int32)
        X_test_gpu = cp.array(X_test, dtype=cp.float32)
        
        start = time.time()
        model = RandomForestClassifier(
            n_estimators=200, max_depth=15, max_features=0.8,
            random_state=42, n_streams=4
        )
        model.fit(X_train_gpu, y_train_gpu)
        train_time = time.time() - start
        
        y_pred = cp.asnumpy(model.predict(X_test_gpu))
        
        result = {
            'cuML RandomForest (GPU)': {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'macro_f1': float(f1_score(y_test, y_pred, average='macro')),
                'weighted_f1': float(f1_score(y_test, y_pred, average='weighted')),
                'train_time': train_time,
                'device': 'GPU'
            }
        }
        print(f"  ✅ Accuracy: {result['cuML RandomForest (GPU)']['accuracy']:.4f} ({train_time:.1f}초)")
        return result
        
    except Exception as e:
        print(f"  ❌ 실패: {e}")
        return {}


def train_lightgbm_model(X_train, X_test, y_train, y_test):
    """LightGBM CUDA"""
    print("\n[5/5] LightGBM (CUDA)")
    
    try:
        import lightgbm as lgb
        
        start = time.time()
        model = lgb.LGBMClassifier(
            device='cuda',
            gpu_platform_id=0,
            gpu_device_id=0,
            n_estimators=300,
            max_depth=10,
            learning_rate=0.1,
            num_leaves=128,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        y_pred = model.predict(X_test)
        
        result = {
            'LightGBM (CUDA)': {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'macro_f1': float(f1_score(y_test, y_pred, average='macro')),
                'weighted_f1': float(f1_score(y_test, y_pred, average='weighted')),
                'train_time': train_time,
                'device': 'GPU'
            }
        }
        print(f"  ✅ Accuracy: {result['LightGBM (CUDA)']['accuracy']:.4f} ({train_time:.1f}초)")
        return result
        
    except Exception as e:
        print(f"  ❌ 실패: {e}")
        return {}


def save_results(results, output_dir='03_models/gpu_comparison'):
    """결과 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON 저장
    with open(f'{output_dir}/gpu_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # CSV 저장
    df = pd.DataFrame(results).T
    df.to_csv(f'{output_dir}/gpu_results.csv')
    
    print(f"\n결과 저장: {output_dir}/")


def print_summary(results):
    """결과 요약 출력"""
    print("\n" + "="*70)
    print("GPU 모델 비교 결과")
    print("="*70)
    
    # 정렬
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print(f"\n{'모델':<30} {'Accuracy':>10} {'Macro F1':>10} {'시간(초)':>10}")
    print("-"*62)
    
    for name, metrics in sorted_results:
        print(f"{name:<30} {metrics['accuracy']:>10.4f} {metrics['macro_f1']:>10.4f} {metrics['train_time']:>10.1f}")
    
    # Top 3
    print("\n" + "="*70)
    print("🏆 Top 3 모델")
    print("="*70)
    
    for i, (name, metrics) in enumerate(sorted_results[:3], 1):
        print(f"\n{i}. {name}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Macro F1: {metrics['macro_f1']:.4f}")
        print(f"   학습 시간: {metrics['train_time']:.1f}초")


if __name__ == '__main__':
    print("="*70)
    print("GPU 전용 모델 비교 파이프라인")
    print("="*70)
    
    # 현재 환경 확인
    env = os.environ.get('CONDA_DEFAULT_ENV', '')
    venv = os.environ.get('VIRTUAL_ENV', '')
    
    if 'cuml' in venv:
        print("환경: cuML (venv_cuml)")
    elif 'lightgbm' in venv:
        print("환경: LightGBM (venv_lightgbm)")
    else:
        print("환경: 기본 시스템")
    
    # 데이터 로드
    X_train, X_test, y_train, y_test, feature_cols = load_data()
    
    all_results = {}
    
    # 기본 GPU 모델
    print("\n" + "="*70)
    print("기본 GPU 모델 학습")
    print("="*70)
    basic_results = train_basic_gpu_models(X_train, X_test, y_train, y_test)
    all_results.update(basic_results)
    
    # cuML (별도 환경 필요)
    if 'cuml' in venv:
        cuml_results = train_cuml_model(X_train, X_test, y_train, y_test)
        all_results.update(cuml_results)
    
    # LightGBM (별도 환경 필요)
    if 'lightgbm' in venv:
        lgb_results = train_lightgbm_model(X_train, X_test, y_train, y_test)
        all_results.update(lgb_results)
    
    # 결과 저장
    save_results(all_results)
    
    # 요약 출력
    print_summary(all_results)
