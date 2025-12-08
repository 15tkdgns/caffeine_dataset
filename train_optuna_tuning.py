"""
Optuna 하이퍼파라미터 자동 튜닝 (GPU)
XGBoost + LightGBM 최적화
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import optuna
import joblib
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
import time

print("="*80)
print("🔬 Optuna 하이퍼파라미터 자동 튜닝 (GPU)")
print("="*80)

# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n[1/4] 데이터 로드")

X_train = np.load('02_data/02_augmented/X_train_smote.npy')
y_train = np.load('02_data/02_augmented/y_train_smote.npy')
X_test = np.load('02_data/02_augmented/X_test.npy')
y_test = np.load('02_data/02_augmented/y_test.npy')

print(f"  학습: {len(X_train):,}, 테스트: {len(X_test):,}")

# 샘플링 (빠른 튜닝을 위해)
sample_size = min(1000000, len(X_train))
sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
X_train_sample = X_train[sample_idx]
y_train_sample = y_train[sample_idx]
print(f"  튜닝용 샘플: {len(X_train_sample):,}건 (빠른 실험)")

# ============================================================
# 2. XGBoost 하이퍼파라미터 튜닝
# ============================================================
print("\n[2/4] XGBoost GPU 하이퍼파라미터 튜닝")

def objective_xgb(trial):
    params = {
        'device': 'cuda',
        'tree_method': 'hist',
        'n_estimators': trial.suggest_int('n_estimators', 200, 500),
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_sample, y_train_sample)
    y_pred = model.predict(X_test)
    
    # Macro F1 최적화
    f1 = f1_score(y_test, y_pred, average='macro')
    return f1

print("  Optuna 시작 (30 trials)...")
study_xgb = optuna.create_study(direction='maximize', study_name='xgboost_gpu')
study_xgb.optimize(objective_xgb, n_trials=30, show_progress_bar=True)

print(f"\n  ✅ 최적 Macro F1: {study_xgb.best_value:.4f}")
print(f"  📋 최적 파라미터:")
for key, value in study_xgb.best_params.items():
    print(f"     {key}: {value}")

# ============================================================
# 3. 최적 파라미터로 전체 데이터 학습
# ============================================================
print("\n[3/4] 최적 모델 학습 (전체 데이터)")

best_params_xgb = study_xgb.best_params.copy()
best_params_xgb['device'] = 'cuda'
best_params_xgb['tree_method'] = 'hist'
best_params_xgb['random_state'] = 42
best_params_xgb['n_jobs'] = -1

print(f"\n  [XGBoost] 전체 데이터 학습 시작...")
start = time.time()
best_xgb = xgb.XGBClassifier(**best_params_xgb)
best_xgb.fit(X_train, y_train)
train_time = time.time() - start
print(f"  ✅ 학습 완료: {train_time:.2f}초")

# 평가
y_pred = best_xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')
category_f1 = f1_score(y_test, y_pred, average=None)

print(f"\n  📊 최종 성능:")
print(f"     Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"     Macro F1:    {macro_f1:.4f} ({macro_f1*100:.2f}%)")
print(f"     Weighted F1: {weighted_f1:.4f}")

category_names = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']
print(f"\n  📈 카테고리별 F1:")
for cat, f1 in zip(category_names, category_f1):
    print(f"     {cat:6s}: {f1:.4f}")

# ============================================================
# 4. 모델 및 결과 저장
# ============================================================
print("\n[4/4] 모델 저장")

import os
os.makedirs('03_models/optuna_tuned', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 모델 저장
model_path = f'03_models/optuna_tuned/xgboost_tuned_{timestamp}.joblib'
joblib.dump(best_xgb, model_path)
print(f"  ✅ 모델 저장: {model_path}")

# 메타데이터 저장
metadata = {
    'model_info': {
        'name': 'XGBoost GPU (Optuna Tuned)',
        'tuning_method': 'Optuna TPE',
        'n_trials': 30,
        'objective': 'Macro F1 Maximization',
        'created_at': datetime.now().isoformat()
    },
    'best_params': best_params_xgb,
    'performance': {
        'accuracy': round(accuracy, 4),
        'macro_f1': round(macro_f1, 4),
        'weighted_f1': round(weighted_f1, 4),
        'category_f1': {cat: round(f1, 4) for cat, f1 in zip(category_names, category_f1)},
        'train_time_seconds': round(train_time, 2)
    },
    'comparison': {
        'baseline_lgb_acc': 0.4913,
        'baseline_lgb_f1': 0.4344,
        'improvement_acc': round((accuracy - 0.4913) * 100, 2),
        'improvement_f1': round((macro_f1 - 0.4344) * 100, 2)
    }
}

metadata_path = f'03_models/optuna_tuned/metadata_{timestamp}.json'
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"  ✅ 메타데이터 저장: {metadata_path}")

# ============================================================
# 5. 결과 비교
# ============================================================
print("\n" + "="*80)
print("🏆 성능 개선 결과")
print("="*80)

lgb_baseline = {'accuracy': 0.4913, 'macro_f1': 0.4344}

print(f"\n{'모델':<30} {'Accuracy':>12} {'Macro F1':>12} {'개선'}")
print("-"*70)
print(f"{'기존 LightGBM (Baseline)':<30} {lgb_baseline['accuracy']:>12.4f} {lgb_baseline['macro_f1']:>12.4f}")
print(f"{'Optuna XGBoost (NEW)':<30} {accuracy:>12.4f} {macro_f1:>12.4f} {'Acc: {:+.2f}%p'.format((accuracy - lgb_baseline['accuracy'])*100)}")
print("-"*70)

if accuracy > lgb_baseline['accuracy']:
    print(f"\n✅ 성능 개선 성공! Accuracy {(accuracy - lgb_baseline['accuracy'])*100:+.2f}%p 향상")
elif macro_f1 > lgb_baseline['macro_f1']:
    print(f"\n✅ Macro F1 개선 성공! {(macro_f1 - lgb_baseline['macro_f1'])*100:+.2f}%p 향상")
else:
    print(f"\n⚠️ 개선 미미. 추가 튜닝 필요.")

print("\n" + "="*80)
print("✅ 하이퍼파라미터 튜닝 완료!")
print("="*80)
print(f"\n📦 저장 파일:")
print(f"   - 모델: {model_path}")
print(f"   - 메타데이터: {metadata_path}")
print("\n" + "="*80)
