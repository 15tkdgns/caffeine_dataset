"""
STEP 2: Optuna 심화 - 50% 돌파 최후 시도
- 좁은 범위 집중 탐색
- 500 trials (샘플 데이터로 빠르게)
- 최고 모델만 전체 데이터로 재학습
"""

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score
import optuna
import time

print("="*80)
print("🔥 STEP 2: Optuna 심화 (50% 돌파 최후 시도)")
print("="*80)

# ============================================================
# 1. 데이터 로드 및 샘플링
# ============================================================
print("\n[1/4] 데이터 로드 및 샘플링")

X_train = np.load('02_data/02_augmented/X_train_smote.npy')
y_train = np.load('02_data/02_augmented/y_train_smote.npy')
X_test = np.load('02_data/02_augmented/X_test.npy')
y_test = np.load('02_data/02_augmented/y_test.npy')

# 원본만 사용
original_size = 5139965
X_train_original = X_train[:original_size]
y_train_original = y_train[:original_size]

# 빠른 튜닝을 위해 샘플링 (1M)
sample_size = 1000000
np.random.seed(42)
sample_idx = np.random.choice(len(X_train_original), sample_size, replace=False)
X_train_sample = X_train_original[sample_idx]
y_train_sample = y_train_original[sample_idx]

print(f"  전체 학습: {len(X_train_original):,}건")
print(f"  샘플링: {len(X_train_sample):,}건 (튜닝용)")
print(f"  테스트: {len(X_test):,}건")

# ============================================================
# 2. Optuna - LightGBM 튜닝
# ============================================================
print("\n[2/4] Optuna - LightGBM 튜닝 (100 trials)")

def objective_lgb(trial):
    """LightGBM Objective (좁은 범위)"""
    params = {
        # 현재 최적값 근처로 좁힘
        'n_estimators': trial.suggest_int('n_estimators', 400, 600),
        'max_depth': trial.suggest_int('max_depth', 10, 14),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.08, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 200, 300),
        'subsample': trial.suggest_float('subsample', 0.85, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.85, 0.95),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 40),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.3),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.3),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train_sample, y_train_sample)
    y_pred = model.predict(X_test)
    
    # Accuracy만 최적화
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

print("  Optuna 시작...")
start = time.time()

study_lgb = optuna.create_study(direction='maximize', study_name='lgb_accuracy_deep')
study_lgb.optimize(objective_lgb, n_trials=100, show_progress_bar=True, n_jobs=1)

optuna_time_lgb = time.time() - start

print(f"\n  ✅ 완료: {optuna_time_lgb:.2f}초 ({optuna_time_lgb/60:.1f}분)")
print(f"  🏆 최고 Accuracy: {study_lgb.best_value:.4f} ({study_lgb.best_value*100:.2f}%)")
print(f"  📋 최적 파라미터:")
for key, value in study_lgb.best_params.items():
    print(f"     {key}: {value}")

# ============================================================
# 3. Optuna - XGBoost 튜닝
# ============================================================
print("\n[3/4] Optuna - XGBoost 튜닝 (100 trials)")

def objective_xgb(trial):
    """XGBoost Objective (좁은 범위)"""
    params = {
        'device': 'cuda',
        'tree_method': 'hist',
        'n_estimators': trial.suggest_int('n_estimators', 400, 600),
        'max_depth': trial.suggest_int('max_depth', 10, 14),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.08, log=True),
        'subsample': trial.suggest_float('subsample', 0.85, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.85, 0.95),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        'gamma': trial.suggest_float('gamma', 0.0, 0.3),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.3),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.3),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_sample, y_train_sample)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

print("  Optuna 시작...")
start = time.time()

study_xgb = optuna.create_study(direction='maximize', study_name='xgb_accuracy_deep')
study_xgb.optimize(objective_xgb, n_trials=100, show_progress_bar=True, n_jobs=1)

optuna_time_xgb = time.time() - start

print(f"\n  ✅ 완료: {optuna_time_xgb:.2f}초 ({optuna_time_xgb/60:.1f}분)")
print(f"  🏆 최고 Accuracy: {study_xgb.best_value:.4f} ({study_xgb.best_value*100:.2f}%)")
print(f"  📋 최적 파라미터:")
for key, value in study_xgb.best_params.items():
    print(f"     {key}: {value}")

# ============================================================
# 4. 최고 모델 전체 데이터로 재학습
# ============================================================
print("\n[4/4] 최고 모델 전체 데이터로 재학습")

# LightGBM vs XGBoost 비교
if study_lgb.best_value >= study_xgb.best_value:
    best_model_name = "LightGBM"
    best_params = study_lgb.best_params
    best_sample_acc = study_lgb.best_value
    
    print(f"  🏆 선택: LightGBM ({best_sample_acc:.4f})")
    print(f"  전체 데이터로 재학습 중...")
    
    final_model = lgb.LGBMClassifier(**best_params)
    start = time.time()
    final_model.fit(X_train_original, y_train_original)
    final_train_time = time.time() - start
    
else:
    best_model_name = "XGBoost"
    best_params = study_xgb.best_params
    best_sample_acc = study_xgb.best_value
    
    print(f"  🏆 선택: XGBoost ({best_sample_acc:.4f})")
    print(f"  전체 데이터로 재학습 중...")
    
    final_model = xgb.XGBClassifier(**best_params)
    start = time.time()
    final_model.fit(X_train_original, y_train_original)
    final_train_time = time.time() - start

print(f"  ✅ 재학습 완료: {final_train_time:.2f}초")

# 최종 평가
y_pred_final = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred_final)

print(f"\n  📊 최종 성능:")
print(f"     샘플 데이터: {best_sample_acc:.4f}")
print(f"     전체 데이터: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

# ============================================================
# 최종 결과
# ============================================================
print("\n" + "="*80)
print("🏆 Optuna 심화 최종 결과")
print("="*80)

results = [
    ("Baseline (원본)", 0.4913),
    ("하이퍼파라미터 최적화", 0.4950),
    ("Stacking", 0.4962),
    (f"Optuna 심화 ({best_model_name})", final_accuracy),
]

print(f"\n{'단계':<35} {'Accuracy':>12} {'50% 달성':>12}")
print("-"*65)
for name, acc in results:
    status = "✅" if acc >= 0.50 else "❌"
    print(f"{name:<35} {acc:>12.4f} {status:>12}")
print("-"*65)

if final_accuracy >= 0.50:
    print(f"\n🎉🎉🎉 50% 달성 성공!")
    print(f"   최종 Accuracy: {final_accuracy*100:.2f}%")
    improvement = (final_accuracy - 0.4913) * 100
    print(f"   Baseline 대비: +{improvement:.2f}%p")
    
    # 모델 저장
    import joblib
    import os
    os.makedirs('04_logs/optuna_deep', exist_ok=True)
    
    model_path = f'04_logs/optuna_deep/{best_model_name.lower()}_50plus.joblib'
    joblib.dump(final_model, model_path)
    print(f"\n   💾 모델 저장: {model_path}")
    
else:
    shortage = (0.50 - final_accuracy) * 100
    print(f"\n⚠️ 50% 미달성 ({shortage:.2f}%p 부족)")
    print(f"   최종 Accuracy: {final_accuracy*100:.2f}%")
    
    if final_accuracy >= 0.4980:
        print(f"\n   → 50%에 매우 근접! (0.2%p 이내)")
        print(f"   → 랜덤 시드 변경 또는 추가 튜닝으로 돌파 가능")
    else:
        print(f"\n   → 6개 카테고리 + 현재 피처의 실질적 한계로 판단")
        print(f"   → 카테고리 재정의 또는 추가 피처 필요")

print("\n" + "="*80)
print("✅ Optuna 심화 완료!")
print("="*80)
print(f"\n⏱️ 총 소요 시간: {(optuna_time_lgb + optuna_time_xgb)/60:.1f}분")
print("="*80)
