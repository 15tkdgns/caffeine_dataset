#!/bin/bash
# SMOTE 증강 데이터로 GPU 모델 재학습

echo "======================================================================"
echo "SMOTE 증강 데이터로 GPU 모델 재학습"
echo "======================================================================"

cd /root/ibm_data2
OUTPUT_DIR="03_models/gpu_smote_comparison"
mkdir -p $OUTPUT_DIR

START_TIME=$(date +%s)

# 1. 기본 GPU 모델 (XGBoost, TensorFlow, CatBoost)
echo -e "\n######################################################################"
echo "# [1/3] 기본 GPU 모델 (SMOTE 데이터)"
echo "######################################################################"

python3 << 'EOF'
import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score
import time

print("="*70)
print("SMOTE 증강 데이터로 GPU 모델 학습")
print("="*70)

# 데이터 로드 (SMOTE 증강)
print("\n데이터 로드...")
X_train = np.load('02_data/02_augmented/X_train_smote.npy')
y_train = np.load('02_data/02_augmented/y_train_smote.npy')
X_test = np.load('02_data/02_augmented/X_test.npy')
y_test = np.load('02_data/02_augmented/y_test.npy')

print(f"학습: {len(X_train):,}, 테스트: {len(X_test):,}")

results = {}

# XGBoost
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
        'train_time': train_time
    }
    print(f"  ✅ Accuracy: {results['XGBoost (GPU)']['accuracy']:.4f} ({train_time:.1f}초)")
except Exception as e:
    print(f"  ❌ 실패: {e}")

# TensorFlow
print("\n[2/3] TensorFlow Neural Network (GPU)")
try:
    import tensorflow as tf
    from tensorflow import keras
    
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(len(np.unique(y_train)), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    start = time.time()
    model.fit(X_train, y_train, epochs=10, batch_size=2048, validation_split=0.1, verbose=0)
    train_time = time.time() - start
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    results['TensorFlow NN (GPU)'] = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'macro_f1': float(f1_score(y_test, y_pred, average='macro')),
        'train_time': train_time
    }
    print(f"  ✅ Accuracy: {results['TensorFlow NN (GPU)']['accuracy']:.4f} ({train_time:.1f}초)")
except Exception as e:
    print(f"  ❌ 실패: {e}")

# CatBoost
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
        'train_time': train_time
    }
    print(f"  ✅ Accuracy: {results['CatBoost (GPU)']['accuracy']:.4f} ({train_time:.1f}초)")
except Exception as e:
    print(f"  ❌ 실패: {e}")

# 결과 저장
with open('03_models/gpu_smote_comparison/basic_gpu_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ 기본 GPU 모델 결과 저장 완료")
EOF

# 2. cuML RandomForest (GPU)
echo -e "\n######################################################################"
echo "# [2/3] cuML RandomForest (GPU)"
echo "######################################################################"

source /root/ibm_data2/venv_cuml/bin/activate

python3 << 'EOF'
import numpy as np
import json
import cupy as cp
from cuml.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import time

print("="*70)
print("cuML RandomForest (GPU) - SMOTE 데이터")
print("="*70)

X_train = np.load('02_data/02_augmented/X_train_smote.npy')
y_train = np.load('02_data/02_augmented/y_train_smote.npy')
X_test = np.load('02_data/02_augmented/X_test.npy')
y_test = np.load('02_data/02_augmented/y_test.npy')

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
        'train_time': train_time
    }
}

print(f"✅ Accuracy: {result['cuML RandomForest (GPU)']['accuracy']:.4f} ({train_time:.1f}초)")

with open('03_models/gpu_smote_comparison/cuml_results.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"✅ cuML 결과 저장 완료")
EOF

deactivate

# 3. LightGBM CUDA
echo -e "\n######################################################################"
echo "# [3/3] LightGBM (CUDA)"
echo "######################################################################"

source /root/ibm_data2/venv_lightgbm/bin/activate

python3 << 'EOF'
import numpy as np
import json
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
import time

print("="*70)
print("LightGBM (CUDA) - SMOTE 데이터")
print("="*70)

X_train = np.load('02_data/02_augmented/X_train_smote.npy')
y_train = np.load('02_data/02_augmented/y_train_smote.npy')
X_test = np.load('02_data/02_augmented/X_test.npy')
y_test = np.load('02_data/02_augmented/y_test.npy')

start = time.time()
model = lgb.LGBMClassifier(
    device='cuda',
    n_estimators=300, max_depth=10, learning_rate=0.1,
    num_leaves=128, random_state=42, verbose=-1
)
model.fit(X_train, y_train)
train_time = time.time() - start

y_pred = model.predict(X_test)

result = {
    'LightGBM (CUDA)': {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'macro_f1': float(f1_score(y_test, y_pred, average='macro')),
        'train_time': train_time
    }
}

print(f"✅ Accuracy: {result['LightGBM (CUDA)']['accuracy']:.4f} ({train_time:.1f}초)")

with open('03_models/gpu_smote_comparison/lightgbm_results.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"✅ LightGBM 결과 저장 완료")
EOF

deactivate

# 결과 통합 및 비교
echo -e "\n######################################################################"
echo "# 결과 통합 및 원본 데이터와 비교"
echo "######################################################################"

python3 << 'EOF'
import json
import pandas as pd
import os

output_dir = '03_models/gpu_smote_comparison'
all_results = {}

# SMOTE 결과 로드
for fname in ['basic_gpu_results.json', 'cuml_results.json', 'lightgbm_results.json']:
    fpath = os.path.join(output_dir, fname)
    if os.path.exists(fpath):
        with open(fpath, 'r') as f:
            results = json.load(f)
            all_results.update(results)

# 원본 결과 로드
original_dir = '03_models/gpu_comparison'
original_results = {}
for fname in ['basic_gpu_results.json', 'cuml_results.json', 'lightgbm_results.json']:
    fpath = os.path.join(original_dir, fname)
    if os.path.exists(fpath):
        with open(fpath, 'r') as f:
            results = json.load(f)
            original_results.update(results)

# 통합 저장
with open(os.path.join(output_dir, 'all_gpu_smote_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

# SMOTE 결과 정렬
sorted_smote = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

print("\n" + "="*70)
print("🏆 SMOTE 데이터로 학습한 GPU 모델 순위")
print("="*70)

print(f"\n{'순위':<4} {'모델':<30} {'Accuracy':>10} {'Macro F1':>10} {'시간(초)':>10}")
print("-"*70)

for i, (name, metrics) in enumerate(sorted_smote, 1):
    print(f"{i:<4} {name:<30} {metrics['accuracy']:>10.4f} {metrics['macro_f1']:>10.4f} {metrics['train_time']:>10.1f}")

# 비교
print("\n" + "="*70)
print("📊 원본 vs SMOTE 비교")
print("="*70)

print(f"\n{'모델':<30} {'원본 Acc':>12} {'SMOTE Acc':>12} {'변화':>10}")
print("-"*70)

for name in all_results:
    if name in original_results:
        orig_acc = original_results[name]['accuracy']
        smote_acc = all_results[name]['accuracy']
        diff = smote_acc - orig_acc
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
        print(f"{name:<30} {orig_acc:>12.4f} {smote_acc:>12.4f} {arrow} {abs(diff):>+.4f}")
EOF

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo -e "\n======================================================================"
echo "완료!"
echo "======================================================================"
echo "총 소요 시간: $(($DURATION / 60))분 $(($DURATION % 60))초"
echo "결과: $OUTPUT_DIR/"
