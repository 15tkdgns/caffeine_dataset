"""
XGBoost Optuna 하이퍼파라미터 Multi-Objective 튜닝 (GPU)
Accuracy와 Macro F1 동시 최적화
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import joblib
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score
import time

print("="*80)
print("🔬 XGBoost Multi-Objective Optuna 튜닝 (GPU)")
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
sample_size = min(1500000, len(X_train))
sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
X_train_sample = X_train[sample_idx]
y_train_sample = y_train[sample_idx]
print(f"  튜닝용 샘플: {len(X_train_sample):,}건 (빠른 실험)")

# ============================================================
# 2. XGBoost Multi-Objective 하이퍼파라미터 튜닝
# ============================================================
print("\n[2/5] XGBoost Multi-Objective 하이퍼파라미터 튜닝")

def objective_xgb_multi(trial):
    """
    Multi-Objective: Accuracy + Macro F1 + 생활 카테고리 F1 개선
    """
    params = {
        'device': 'cuda',
        'tree_method': 'hist',
        
        # 튜닝 대상 파라미터
        'n_estimators': trial.suggest_int('n_estimators', 200, 600),
        'max_depth': trial.suggest_int('max_depth', 6, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
        
        'random_state': 42,
        'n_jobs': -1
    }
    
    # 모델 학습
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_sample, y_train_sample)
    
    # 테스트 데이터로 평가
    y_pred = model.predict(X_test)
    
    # Accuracy와 Macro F1
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    # Multi-Objective Score
    # 1. Accuracy 50% 가중치
    # 2. Macro F1 40% 가중치
    # 3. 생활 카테고리 F1 10% 가중치 (중요하지만 과도하지 않게)
    category_f1 = f1_score(y_test, y_pred, average=None)
    life_f1 = category_f1[1]  # 생활 카테고리
    
    balanced_score = 0.5 * accuracy + 0.4 * macro_f1 + 0.1 * life_f1
    
    return balanced_score

print("  Optuna 시작 (60 trials - 더 많이 탐색)...")
print("  목표: Accuracy 50% + Macro F1 40% + 생활 F1 10%")

study_xgb = optuna.create_study(direction='maximize', study_name='xgboost_gpu_multi_objective')
study_xgb.optimize(objective_xgb_multi, n_trials=60, show_progress_bar=True)

print(f"\n  ✅ 최적 점수: {study_xgb.best_value:.4f}")
print(f"  📋 최적 파라미터:")
for key, value in study_xgb.best_params.items():
    if isinstance(value, float):
        print(f"     {key}: {value:.6f}")
    else:
        print(f"     {key}: {value}")

# ============================================================
# 3. 최적 파라미터로 전체 데이터 학습
# ============================================================
print("\n[3/5] 최적 모델 학습 (전체 데이터)")

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
print(f"  ✅ 학습 완료: {train_time:.2f}초 ({train_time/60:.2f}분)")

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
    emoji = "⭐" if f1 > 0.5 else "✅" if f1 > 0.3 else "⚠️"
    print(f"     {emoji} {cat:6s}: {f1:.4f}")

# ============================================================
# 4. 모델 및 결과 저장
# ============================================================
print("\n[4/5] 모델 저장")

import os
os.makedirs('03_models/xgboost_optuna_multi', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 모델 저장
model_path = f'03_models/xgboost_optuna_multi/xgboost_multi_tuned_{timestamp}.joblib'
joblib.dump(best_xgb, model_path)
print(f"  ✅ 모델 저장: {model_path}")

# 메타데이터 저장
metadata = {
    'model_info': {
        'name': 'XGBoost GPU (Multi-Objective Optuna)',
        'tuning_method': 'Optuna TPE (Multi-Objective)',
        'n_trials': 60,
        'objective': 'Balanced: 50% Acc + 40% F1 + 10% Life F1',
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
        'baseline_life_f1': 0.0802,
        'improvement_acc': round((accuracy - 0.4913) * 100, 2),
        'improvement_f1': round((macro_f1 - 0.4344) * 100, 2),
        'improvement_life_f1': round((category_f1[1] - 0.0802) * 100, 2)
    }
}

metadata_path = f'03_models/xgboost_optuna_multi/metadata_{timestamp}.json'
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"  ✅ 메타데이터 저장: {metadata_path}")

# ============================================================
# 5. 결과 비교
# ============================================================
print("\n" + "="*80)
print("🏆 XGBoost Multi-Objective 튜닝 결과")
print("="*80)

baseline = {'accuracy': 0.4913, 'macro_f1': 0.4344, 'life_f1': 0.0802}
previous_tuning = {'accuracy': 0.4807, 'macro_f1': 0.4547, 'life_f1': 0.2051}

print(f"\n{'모델':<45} {'Accuracy':>12} {'Macro F1':>12} {'생활 F1':>12}")
print("-"*85)
print(f"{'기존 LightGBM (Baseline)':<45} {baseline['accuracy']:>12.4f} {baseline['macro_f1']:>12.4f} {baseline['life_f1']:>12.4f}")
print(f"{'이전 XGBoost Optuna (F1 최적화)':<45} {previous_tuning['accuracy']:>12.4f} {previous_tuning['macro_f1']:>12.4f} {previous_tuning['life_f1']:>12.4f}")
print(f"{'NEW: Multi-Objective XGBoost':<45} {accuracy:>12.4f} {macro_f1:>12.4f} {category_f1[1]:>12.4f}")
print("-"*85)

acc_vs_baseline = (accuracy - baseline['accuracy']) * 100
f1_vs_baseline = (macro_f1 - baseline['macro_f1']) * 100
life_vs_baseline = (category_f1[1] - baseline['life_f1']) * 100

print(f"\n📊 Baseline 대비 개선도:")
print(f"  Accuracy:    {acc_vs_baseline:+.2f}%p")
print(f"  Macro F1:    {f1_vs_baseline:+.2f}%p")
print(f"  생활 F1:     {life_vs_baseline:+.2f}%p")

# 종합 평가
improvements = 0
if accuracy >= baseline['accuracy']: improvements += 1
if macro_f1 >= baseline['macro_f1']: improvements += 1
if category_f1[1] >= 0.15: improvements += 1  # 생활 F1 15% 이상이면 성공

print(f"\n🎯 종합 평가:")
if improvements == 3:
    print(f"  ✅✅✅ 완벽한 성공! 모든 지표 개선")
elif improvements == 2:
    print(f"  ✅✅ 우수! 주요 지표 개선")
elif improvements == 1:
    print(f"  ✅ 부분 성공")
else:
    print(f"  ⚠️ 추가 튜닝 필요")

print("\n" + "="*80)
print("✅ XGBoost Multi-Objective 튜닝 완료!")
print("="*80)
print(f"\n📦 저장 파일:")
print(f"   - 모델: {model_path}")
print(f"   - 메타데이터: {metadata_path}")
print("\n" + "="*80)
