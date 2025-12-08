"""
환경별 모델 학습 실행기
각 Conda 환경에서 해당 모델만 학습하고 결과 저장
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time


def train_in_env1(data_file, output_dir):
    """
    Environment 1: TensorFlow, XGBoost, CatBoost
    """
    print("="*70)
    print("Environment 1: Basic GPU Models")
    print("="*70)
    
    # 데이터 로드
    df = pd.read_csv(data_file)
    feature_cols = [col for col in df.columns if col.endswith('_scaled')]
    X = df[feature_cols].values.astype('float32')
    y = df['Next_Category_encoded'].values.astype('int32')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    results = {}
    
    # XGBoost GPU
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
            'framework': 'xgboost',
            'device': 'GPU',
            'environment': 'env1_basic'
        }
        print(f"  Accuracy: {results['XGBoost (GPU)']['accuracy']:.4f}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # TensorFlow NN
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
        
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        results['Neural Network (TF GPU)'] = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'macro_f1': float(f1_score(y_test, y_pred, average='macro')),
            'weighted_f1': float(f1_score(y_test, y_pred, average='weighted')),
            'train_time': train_time,
            'framework': 'tensorflow',
            'device': 'GPU',
            'environment': 'env1_basic'
        }
        print(f"  Accuracy: {results['Neural Network (TF GPU)']['accuracy']:.4f}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # CatBoost GPU
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
            'framework': 'catboost',
            'device': 'GPU',
            'environment': 'env1_basic'
        }
        print(f"  Accuracy: {results['CatBoost (GPU)']['accuracy']:.4f}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'env1_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Environment 1 results saved")
    return results


def train_in_env2(data_file, output_dir):
    """
    Environment 2: cuML RandomForest
    """
    print("="*70)
    print("Environment 2: RAPIDS cuML")
    print("="*70)
    
    try:
        import cupy as cp
        from cuml.ensemble import RandomForestClassifier
        
        # 데이터 로드
        df = pd.read_csv(data_file)
        feature_cols = [col for col in df.columns if col.endswith('_scaled')]
        X = df[feature_cols].values.astype('float32')
        y = df['Next_Category_encoded'].values.astype('int32')
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
        
        # GPU 배열로 변환
        X_train_gpu = cp.array(X_train, dtype=cp.float32)
        y_train_gpu = cp.array(y_train, dtype=cp.int32)
        X_test_gpu = cp.array(X_test, dtype=cp.float32)
        
        # cuML RandomForest
        print("\n[1/1] cuML RandomForest (GPU)")
        start = time.time()
        model = RandomForestClassifier(
            n_estimators=200, max_depth=15, max_features=0.8,
            random_state=42, n_streams=4
        )
        model.fit(X_train_gpu, y_train_gpu)
        train_time = time.time() - start
        
        y_pred_gpu = model.predict(X_test_gpu)
        y_pred = cp.asnumpy(y_pred_gpu)
        
        results = {
            'RandomForest (cuML GPU)': {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'macro_f1': float(f1_score(y_test, y_pred, average='macro')),
                'weighted_f1': float(f1_score(y_test, y_pred, average='weighted')),
                'train_time': train_time,
                'framework': 'cuml',
                'device': 'GPU',
                'environment': 'env2_rapids'
            }
        }
        print(f"  Accuracy: {results['RandomForest (cuML GPU)']['accuracy']:.4f}")
        
        # 결과 저장
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'env2_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Environment 2 results saved")
        return results
        
    except Exception as e:
        print(f"❌ Environment 2 failed: {e}")
        return {}


def train_in_env3(data_file, output_dir):
    """
    Environment 3: LightGBM CUDA
    """
    print("="*70)
    print("Environment 3: LightGBM CUDA")
    print("="*70)
    
    try:
        import lightgbm as lgb
        
        # 데이터 로드
        df = pd.read_csv(data_file)
        feature_cols = [col for col in df.columns if col.endswith('_scaled')]
        X = df[feature_cols].values.astype('float32')
        y = df['Next_Category_encoded'].values.astype('int32')
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
        
        # LightGBM CUDA
        print("\n[1/1] LightGBM (CUDA)")
        start = time.time()
        model = lgb.LGBMClassifier(
            device='cuda',
            gpu_platform_id=0,
            gpu_device_id=0,
            n_estimators=300,
            max_depth=10,
            learning_rate=0.1,
            num_leaves=128,
            random_state=42
        )
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        y_pred = model.predict(X_test)
        
        results = {
            'LightGBM (CUDA)': {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'macro_f1': float(f1_score(y_test, y_pred, average='macro')),
                'weighted_f1': float(f1_score(y_test, y_pred, average='weighted')),
                'train_time': train_time,
                'framework': 'lightgbm',
                'device': 'GPU',
                'environment': 'env3_lightgbm'
            }
        }
        print(f"  Accuracy: {results['LightGBM (CUDA)']['accuracy']:.4f}")
        
        # 결과 저장
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'env3_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Environment 3 results saved")
        return results
        
    except Exception as e:
        print(f"❌ Environment 3 failed: {e}")
        return {}


if __name__ == '__main__':
    # 환경 감지
    env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    data_file = '02_data/01_processed/preprocessed_enhanced.csv'
    output_dir = '03_models/multi_env_comparison'
    
    print(f"Current environment: {env}")
    
    if 'gpu_basic' in env:
        results = train_in_env1(data_file, output_dir)
    elif 'rapids' in env:
        results = train_in_env2(data_file, output_dir)
    elif 'lightgbm' in env:
        results = train_in_env3(data_file, output_dir)
    else:
        print("❌ Unknown environment. Please activate one of:")
        print("  - gpu_basic")
        print("  - rapids_cuml")
        print("  - lightgbm_cuda")
        sys.exit(1)
