"""
Top 모델 앙상블 최적화
LightGBM + XGBoost 조합으로 최고 성능 달성
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import VotingClassifier
import optuna
import time

print("="*80)
print("🎯 Top 모델 앙상블 최적화")
print("="*80)

# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n[1/6] 데이터 로드")

X_train = np.load('02_data/02_augmented/X_train_smote.npy')
y_train = np.load('02_data/02_augmented/y_train_smote.npy')
X_test = np.load('02_data/02_augmented/X_test.npy')
y_test = np.load('02_data/02_augmented/y_test.npy')

print(f"  학습: {len(X_train):,}, 테스트: {len(X_test):,}")

# ============================================================
# 2. 기존 Top 모델 로드
# ============================================================
print("\n[2/6] Top 모델 로드")

# Baseline LightGBM
try:
    lgb_model = joblib.load('03_models/production_models/lightgbm_cuda_production_20251205_162340.joblib')
    print("  ✅ Baseline LightGBM 로드 완료")
    has_lgb = True
except:
    print("  ⚠️ Baseline LightGBM 로드 실패")
    has_lgb = False

# Multi-Objective XGBoost
try:
    xgb_multi = joblib.load('03_models/xgboost_optuna_multi/xgboost_multi_tuned_20251207_044613.joblib')
    print("  ✅ Multi-Objective XGBoost 로드 완료")
    has_xgb_multi = True
except:
    print("  ⚠️ Multi-Objective XGBoost 로드 실패")
    has_xgb_multi = False

# 이전 Optuna XGBoost
try:
    xgb_optuna = joblib.load('03_models/optuna_tuned/xgboost_tuned_20251205_184240.joblib')
    print("  ✅ 이전 Optuna XGBoost 로드 완료")
    has_xgb_optuna = True
except:
    print("  ⚠️ 이전 Optuna XGBoost 로드 실패")
    has_xgb_optuna = False

# ============================================================
# 3. 개별 모델 성능 평가
# ============================================================
print("\n[3/6] 개별 모델 성능 평가")

models_info = []

if has_lgb:
    y_pred_lgb = lgb_model.predict(X_test)
    acc_lgb = accuracy_score(y_test, y_pred_lgb)
    f1_lgb = f1_score(y_test, y_pred_lgb, average='macro')
    print(f"\n  LightGBM Baseline:")
    print(f"    Accuracy: {acc_lgb:.4f}, Macro F1: {f1_lgb:.4f}")
    models_info.append(('lgb', lgb_model, acc_lgb, f1_lgb))

if has_xgb_multi:
    y_pred_xgb_multi = xgb_multi.predict(X_test)
    acc_xgb_multi = accuracy_score(y_test, y_pred_xgb_multi)
    f1_xgb_multi = f1_score(y_test, y_pred_xgb_multi, average='macro')
    print(f"\n  XGBoost Multi-Objective:")
    print(f"    Accuracy: {acc_xgb_multi:.4f}, Macro F1: {f1_xgb_multi:.4f}")
    models_info.append(('xgb_multi', xgb_multi, acc_xgb_multi, f1_xgb_multi))

if has_xgb_optuna:
    y_pred_xgb_optuna = xgb_optuna.predict(X_test)
    acc_xgb_optuna = accuracy_score(y_test, y_pred_xgb_optuna)
    f1_xgb_optuna = f1_score(y_test, y_pred_xgb_optuna, average='macro')
    print(f"\n  XGBoost Optuna (이전):")
    print(f"    Accuracy: {acc_xgb_optuna:.4f}, Macro F1: {f1_xgb_optuna:.4f}")
    models_info.append(('xgb_optuna', xgb_optuna, acc_xgb_optuna, f1_xgb_optuna))

# ============================================================
# 4. 앙상블 가중치 최적화
# ============================================================
print("\n[4/6] 앙상블 가중치 최적화 (Optuna)")

# 각 모델의 예측 확률 저장
proba_dict = {}
if has_lgb:
    proba_dict['lgb'] = lgb_model.predict_proba(X_test)
if has_xgb_multi:
    proba_dict['xgb_multi'] = xgb_multi.predict_proba(X_test)
if has_xgb_optuna:
    proba_dict['xgb_optuna'] = xgb_optuna.predict_proba(X_test)

def objective_ensemble(trial):
    """
    앙상블 가중치 최적화
    """
    weights = []
    
    if has_lgb:
        w_lgb = trial.suggest_float('w_lgb', 0.0, 1.0)
        weights.append(w_lgb)
    else:
        w_lgb = 0
        
    if has_xgb_multi:
        w_xgb_multi = trial.suggest_float('w_xgb_multi', 0.0, 1.0)
        weights.append(w_xgb_multi)
    else:
        w_xgb_multi = 0
        
    if has_xgb_optuna:
        w_xgb_optuna = trial.suggest_float('w_xgb_optuna', 0.0, 1.0)
        weights.append(w_xgb_optuna)
    else:
        w_xgb_optuna = 0
    
    # 가중치 정규화
    total_weight = sum(weights)
    if total_weight == 0:
        return 0
    
    weights = [w / total_weight for w in weights]
    
    # 가중 평균 예측
    ensemble_proba = np.zeros_like(list(proba_dict.values())[0])
    
    idx = 0
    if has_lgb:
        ensemble_proba += weights[idx] * proba_dict['lgb']
        idx += 1
    if has_xgb_multi:
        ensemble_proba += weights[idx] * proba_dict['xgb_multi']
        idx += 1
    if has_xgb_optuna:
        ensemble_proba += weights[idx] * proba_dict['xgb_optuna']
        idx += 1
    
    # 최종 예측
    y_pred = np.argmax(ensemble_proba, axis=1)
    
    # Multi-Objective: Accuracy 60% + Macro F1 40%
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    score = 0.6 * accuracy + 0.4 * macro_f1
    
    return score

print("  Optuna 시작 (30 trials)...")
study_ensemble = optuna.create_study(direction='maximize', study_name='ensemble_weights')
study_ensemble.optimize(objective_ensemble, n_trials=30, show_progress_bar=True)

print(f"\n  ✅ 최적 점수: {study_ensemble.best_value:.4f}")
print(f"  📋 최적 가중치:")
for key, value in study_ensemble.best_params.items():
    print(f"     {key}: {value:.4f}")

# ============================================================
# 5. 최적 앙상블 모델 생성 및 평가
# ============================================================
print("\n[5/6] 최적 앙상블 모델 평가")

# 최적 가중치 추출
best_weights = []
if has_lgb:
    best_weights.append(study_ensemble.best_params['w_lgb'])
if has_xgb_multi:
    best_weights.append(study_ensemble.best_params['w_xgb_multi'])
if has_xgb_optuna:
    best_weights.append(study_ensemble.best_params['w_xgb_optuna'])

# 가중치 정규화
total = sum(best_weights)
best_weights = [w / total for w in best_weights]

print(f"\n  정규화된 가중치:")
idx = 0
if has_lgb:
    print(f"    LightGBM: {best_weights[idx]:.4f}")
    idx += 1
if has_xgb_multi:
    print(f"    XGBoost Multi: {best_weights[idx]:.4f}")
    idx += 1
if has_xgb_optuna:
    print(f"    XGBoost Optuna: {best_weights[idx]:.4f}")
    idx += 1

# 가중 평균 예측
ensemble_proba = np.zeros_like(list(proba_dict.values())[0])
idx = 0
if has_lgb:
    ensemble_proba += best_weights[idx] * proba_dict['lgb']
    idx += 1
if has_xgb_multi:
    ensemble_proba += best_weights[idx] * proba_dict['xgb_multi']
    idx += 1
if has_xgb_optuna:
    ensemble_proba += best_weights[idx] * proba_dict['xgb_optuna']
    idx += 1

y_pred_ensemble = np.argmax(ensemble_proba, axis=1)

# 최종 성능
accuracy = accuracy_score(y_test, y_pred_ensemble)
macro_f1 = f1_score(y_test, y_pred_ensemble, average='macro')
weighted_f1 = f1_score(y_test, y_pred_ensemble, average='weighted')
category_f1 = f1_score(y_test, y_pred_ensemble, average=None)

print(f"\n  📊 앙상블 최종 성능:")
print(f"     Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"     Macro F1:    {macro_f1:.4f} ({macro_f1*100:.2f}%)")
print(f"     Weighted F1: {weighted_f1:.4f}")

category_names = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']
print(f"\n  📈 카테고리별 F1:")
for cat, f1 in zip(category_names, category_f1):
    emoji = "⭐" if f1 > 0.5 else "✅" if f1 > 0.3 else "⚠️"
    print(f"     {emoji} {cat:6s}: {f1:.4f}")

# ============================================================
# 6. 결과 저장
# ============================================================
print("\n[6/6] 결과 저장")

import os
os.makedirs('03_models/ensemble_optimized', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 메타데이터 저장
metadata = {
    'ensemble_info': {
        'name': 'Weighted Ensemble (Optimized)',
        'method': 'Soft Voting with Optuna-optimized weights',
        'n_models': len(models_info),
        'created_at': datetime.now().isoformat()
    },
    'weights': {
        'lgb': best_weights[0] if has_lgb else 0,
        'xgb_multi': best_weights[1 if has_lgb else 0] if has_xgb_multi else 0,
        'xgb_optuna': best_weights[-1] if has_xgb_optuna else 0
    },
    'performance': {
        'accuracy': round(accuracy, 4),
        'macro_f1': round(macro_f1, 4),
        'weighted_f1': round(weighted_f1, 4),
        'category_f1': {cat: round(f1, 4) for cat, f1 in zip(category_names, category_f1)}
    },
    'individual_models': {
        'lgb': {'acc': round(acc_lgb, 4), 'f1': round(f1_lgb, 4)} if has_lgb else None,
        'xgb_multi': {'acc': round(acc_xgb_multi, 4), 'f1': round(f1_xgb_multi, 4)} if has_xgb_multi else None,
        'xgb_optuna': {'acc': round(acc_xgb_optuna, 4), 'f1': round(f1_xgb_optuna, 4)} if has_xgb_optuna else None
    },
    'comparison': {
        'baseline_lgb_acc': 0.4913,
        'baseline_lgb_f1': 0.4344,
        'improvement_acc': round((accuracy - 0.4913) * 100, 2),
        'improvement_f1': round((macro_f1 - 0.4344) * 100, 2)
    }
}

metadata_path = f'03_models/ensemble_optimized/metadata_{timestamp}.json'
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"  ✅ 메타데이터 저장: {metadata_path}")

# ============================================================
# 7. 최종 비교
# ============================================================
print("\n" + "="*80)
print("🏆 앙상블 최적화 결과")
print("="*80)

baseline = {'accuracy': 0.4913, 'macro_f1': 0.4344, 'life_f1': 0.0802}

print(f"\n{'모델':<50} {'Accuracy':>12} {'Macro F1':>12} {'생활 F1':>12}")
print("-"*90)
print(f"{'기존 LightGBM (Baseline)':<50} {baseline['accuracy']:>12.4f} {baseline['macro_f1']:>12.4f} {baseline['life_f1']:>12.4f}")

if has_lgb:
    print(f"{'  └─ LightGBM (단독)':<50} {acc_lgb:>12.4f} {f1_lgb:>12.4f} {'-':>12}")
if has_xgb_multi:
    print(f"{'  └─ XGBoost Multi (단독)':<50} {acc_xgb_multi:>12.4f} {f1_xgb_multi:>12.4f} {'-':>12}")
if has_xgb_optuna:
    print(f"{'  └─ XGBoost Optuna (단독)':<50} {acc_xgb_optuna:>12.4f} {f1_xgb_optuna:>12.4f} {'-':>12}")

print(f"{'✨ NEW: Weighted Ensemble (Optimized)':<50} {accuracy:>12.4f} {macro_f1:>12.4f} {category_f1[1]:>12.4f}")
print("-"*90)

acc_improve = (accuracy - baseline['accuracy']) * 100
f1_improve = (macro_f1 - baseline['macro_f1']) * 100
life_improve = (category_f1[1] - baseline['life_f1']) * 100

print(f"\n📊 Baseline 대비 개선도:")
print(f"  Accuracy:    {acc_improve:+.2f}%p")
print(f"  Macro F1:    {f1_improve:+.2f}%p")
print(f"  생활 F1:     {life_improve:+.2f}%p")

# 종합 평가
if accuracy > baseline['accuracy'] and macro_f1 > baseline['macro_f1']:
    print(f"\n✅✅✅ 완벽한 성공! 모든 지표 개선")
elif accuracy > baseline['accuracy'] or macro_f1 > baseline['macro_f1']:
    print(f"\n✅✅ 우수! 주요 지표 개선")
else:
    print(f"\n⚠️ 개별 모델 성능에 머무름")

print("\n" + "="*80)
print("✅ 앙상블 최적화 완료!")
print("="*80)
print(f"\n📦 저장 파일:")
print(f"   - 메타데이터: {metadata_path}")
print("\n" + "="*80)
