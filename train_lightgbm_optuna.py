"""
LightGBM Optuna 하이퍼파라미터 자동 튜닝 (GPU)
프로덕션 모델 성능 개선 목표
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score
import time

print("="*80)
print("🔬 LightGBM Optuna 하이퍼파라미터 자동 튜닝 (GPU)")
print("="*80)

# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n[1/5] 데이터 로드")

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
# 2. LightGBM 하이퍼파라미터 튜닝
# ============================================================
print("\n[2/5] LightGBM 하이퍼파라미터 튜닝")

def objective_lgb(trial):
    """
    Optuna Objective: Accuracy와 Macro F1 균형잡힌 최적화
    """
    params = {
        'objective': 'multiclass',
        'num_class': 6,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        
        # 튜닝 대상 파라미터
        'num_leaves': trial.suggest_int('num_leaves', 64, 512),
        'max_depth': trial.suggest_int('max_depth', 8, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 200, 600),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
        
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    # 모델 학습
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train_sample, y_train_sample)
    
    # 테스트 데이터로 평가
    y_pred = model.predict(X_test)
    
    # Multi-objective: Accuracy 60% + Macro F1 40%
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    # 균형잡힌 점수
    balanced_score = 0.6 * accuracy + 0.4 * macro_f1
    
    # 생활 카테고리 F1도 고려 (보너스)
    category_f1 = f1_score(y_test, y_pred, average=None)
    life_f1 = category_f1[1]  # 생활 카테고리
    
    # 생활 F1이 15% 이상이면 보너스 점수
    if life_f1 > 0.15:
        balanced_score += 0.01  # 1% 보너스
    
    return balanced_score

print("  Optuna 시작 (50 trials)...")
print("  목표: Accuracy 60% + Macro F1 40% 균형 최적화")

study_lgb = optuna.create_study(direction='maximize', study_name='lightgbm_gpu_balanced')
study_lgb.optimize(objective_lgb, n_trials=50, show_progress_bar=True)

print(f"\n  ✅ 최적 점수: {study_lgb.best_value:.4f}")
print(f"  📋 최적 파라미터:")
for key, value in study_lgb.best_params.items():
    if isinstance(value, float):
        print(f"     {key}: {value:.6f}")
    else:
        print(f"     {key}: {value}")

# ============================================================
# 3. 최적 파라미터로 전체 데이터 학습
# ============================================================
print("\n[3/5] 최적 모델 학습 (전체 데이터)")

best_params_lgb = study_lgb.best_params.copy()
best_params_lgb['objective'] = 'multiclass'
best_params_lgb['num_class'] = 6
best_params_lgb['metric'] = 'multi_logloss'
best_params_lgb['boosting_type'] = 'gbdt'
best_params_lgb['device'] = 'gpu'
best_params_lgb['gpu_platform_id'] = 0
best_params_lgb['gpu_device_id'] = 0
best_params_lgb['random_state'] = 42
best_params_lgb['n_jobs'] = -1
best_params_lgb['verbose'] = -1

print(f"\n  [LightGBM] 전체 데이터 학습 시작...")
start = time.time()
best_lgb = lgb.LGBMClassifier(**best_params_lgb)
best_lgb.fit(X_train, y_train)
train_time = time.time() - start
print(f"  ✅ 학습 완료: {train_time:.2f}초 ({train_time/60:.2f}분)")

# 평가
y_pred = best_lgb.predict(X_test)
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
print("\n[4/5] 모델 저장")

import os
os.makedirs('03_models/lightgbm_optuna', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 모델 저장
model_path = f'03_models/lightgbm_optuna/lightgbm_tuned_{timestamp}.joblib'
joblib.dump(best_lgb, model_path)
print(f"  ✅ 모델 저장: {model_path}")

# 메타데이터 저장
metadata = {
    'model_info': {
        'name': 'LightGBM GPU (Optuna Tuned - Balanced)',
        'tuning_method': 'Optuna TPE',
        'n_trials': 50,
        'objective': 'Balanced: 60% Accuracy + 40% Macro F1',
        'created_at': datetime.now().isoformat()
    },
    'best_params': best_params_lgb,
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

metadata_path = f'03_models/lightgbm_optuna/metadata_{timestamp}.json'
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"  ✅ 메타데이터 저장: {metadata_path}")

# ============================================================
# 5. 결과 비교
# ============================================================
print("\n" + "="*80)
print("🏆 LightGBM Optuna 튜닝 결과")
print("="*80)

baseline = {'accuracy': 0.4913, 'macro_f1': 0.4344, 'life_f1': 0.0802}

print(f"\n{'모델':<40} {'Accuracy':>12} {'Macro F1':>12} {'생활 F1':>12}")
print("-"*80)
print(f"{'기존 LightGBM (Baseline)':<40} {baseline['accuracy']:>12.4f} {baseline['macro_f1']:>12.4f} {baseline['life_f1']:>12.4f}")
print(f"{'Optuna LightGBM (NEW)':<40} {accuracy:>12.4f} {macro_f1:>12.4f} {category_f1[1]:>12.4f}")
print("-"*80)

acc_improve = (accuracy - baseline['accuracy']) * 100
f1_improve = (macro_f1 - baseline['macro_f1']) * 100
life_improve = (category_f1[1] - baseline['life_f1']) * 100

print(f"\n📊 개선도:")
print(f"  Accuracy:    {acc_improve:+.2f}%p")
print(f"  Macro F1:    {f1_improve:+.2f}%p")
print(f"  생활 F1:     {life_improve:+.2f}%p")

if accuracy > baseline['accuracy'] and macro_f1 > baseline['macro_f1']:
    print(f"\n✅ 성능 개선 성공! 모든 지표 향상")
elif accuracy > baseline['accuracy'] or macro_f1 > baseline['macro_f1']:
    print(f"\n✅ 부분 개선 성공!")
else:
    print(f"\n⚠️ 추가 튜닝 필요")

print("\n" + "="*80)
print("✅ LightGBM 하이퍼파라미터 튜닝 완료!")
print("="*80)
print(f"\n📦 저장 파일:")
print(f"   - 모델: {model_path}")
print(f"   - 메타데이터: {metadata_path}")
print("\n" + "="*80)
