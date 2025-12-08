"""
앙상블 모델 학습 (GPU 가속)
XGBoost(GPU) + CatBoost(GPU) 앙상블로 성능 개선
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
import joblib
import json
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import VotingClassifier
import time

print("="*80)
print("🚀 앙상블 모델 학습 (GPU 가속)")
print("="*80)

# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n[1/5] SMOTE 증강 데이터 로드")

X_train = np.load('02_data/02_augmented/X_train_smote.npy')
y_train = np.load('02_data/02_augmented/y_train_smote.npy')
X_test = np.load('02_data/02_augmented/X_test.npy')
y_test = np.load('02_data/02_augmented/y_test.npy')

print(f"  학습 데이터: {len(X_train):,}건")
print(f"  테스트 데이터: {len(X_test):,}건")

# ============================================================
# 2. 개별 모델 정의 (GPU)
# ============================================================
print("\n[2/5] GPU 모델 정의")

# XGBoost GPU
xgb_model = xgb.XGBClassifier(
    device='cuda',
    tree_method='hist',
    n_estimators=300,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
print("  ✅ XGBoost (GPU)")

# CatBoost GPU
cat_model = CatBoostClassifier(
    task_type='GPU',
    devices='0',
    iterations=300,
    depth=10,
    learning_rate=0.1,
    random_state=42,
    verbose=False
)
print("  ✅ CatBoost (GPU)")

# ============================================================
# 3. 개별 모델 학습
# ============================================================
print("\n[3/5] 개별 모델 학습")

# XGBoost 학습
print("\n  [XGBoost] 학습 시작...")
start = time.time()
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - start
print(f"  ✅ XGBoost 완료: {xgb_time:.2f}초")

# CatBoost 학습
print("\n  [CatBoost] 학습 시작...")
start = time.time()
cat_model.fit(X_train, y_train)
cat_time = time.time() - start
print(f"  ✅ CatBoost 완료: {cat_time:.2f}초")

# ============================================================
# 4. 개별 모델 평가
# ============================================================
print("\n[4/5] 개별 모델 평가")

results = {}

# XGBoost 평가
y_pred_xgb = xgb_model.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb, average='macro')
results['XGBoost (GPU)'] = {
    'accuracy': float(acc_xgb),
    'macro_f1': float(f1_xgb),
    'train_time': xgb_time
}
print(f"\n  XGBoost: Acc={acc_xgb:.4f}, F1={f1_xgb:.4f}")

# CatBoost 평가
y_pred_cat = cat_model.predict(X_test)
acc_cat = accuracy_score(y_test, y_pred_cat)
f1_cat = f1_score(y_test, y_pred_cat, average='macro')
results['CatBoost (GPU)'] = {
    'accuracy': float(acc_cat),
    'macro_f1': float(f1_cat),
    'train_time': cat_time
}
print(f"  CatBoost: Acc={acc_cat:.4f}, F1={f1_cat:.4f}")

# ============================================================
# 5. 앙상블 (Soft Voting)
# ============================================================
print("\n[5/5] 앙상블 예측 (Soft Voting)")

# 확률 예측
y_proba_xgb = xgb_model.predict_proba(X_test)
y_proba_cat = cat_model.predict_proba(X_test)

# Soft Voting (평균)
y_proba_ensemble = (y_proba_xgb + y_proba_cat) / 2
y_pred_ensemble = np.argmax(y_proba_ensemble, axis=1)

# 앙상블 평가
acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
f1_ensemble = f1_score(y_test, y_pred_ensemble, average='macro')
weighted_f1 = f1_score(y_test, y_pred_ensemble, average='weighted')
category_f1 = f1_score(y_test, y_pred_ensemble, average=None)

results['Ensemble (XGB+CAT)'] = {
    'accuracy': float(acc_ensemble),
    'macro_f1': float(f1_ensemble),
    'weighted_f1': float(weighted_f1),
    'train_time': xgb_time + cat_time
}

print(f"\n  📊 앙상블 성능:")
print(f"     Accuracy:    {acc_ensemble:.4f} ({acc_ensemble*100:.2f}%)")
print(f"     Macro F1:    {f1_ensemble:.4f} ({f1_ensemble*100:.2f}%)")
print(f"     Weighted F1: {weighted_f1:.4f}")

# 카테고리별 F1
category_names = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']
print(f"\n  📈 카테고리별 F1 Score:")
for cat_name, f1 in zip(category_names, category_f1):
    print(f"     {cat_name:6s}: {f1:.4f}")

# ============================================================
# 6. 결과 저장
# ============================================================
print("\n[6/6] 모델 및 결과 저장")

output_dir = '03_models/ensemble'
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 개별 모델 저장
xgb_path = f'{output_dir}/xgboost_gpu_{timestamp}.joblib'
cat_path = f'{output_dir}/catboost_gpu_{timestamp}.joblib'

joblib.dump(xgb_model, xgb_path)
joblib.dump(cat_model, cat_path)

print(f"  ✅ XGBoost 저장: {xgb_path}")
print(f"  ✅ CatBoost 저장: {cat_path}")

# 결과 저장
results_path = f'{output_dir}/ensemble_results_{timestamp}.json'
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"  ✅ 결과 저장: {results_path}")

# 메타데이터 저장
metadata = {
    'ensemble_info': {
        'name': 'XGBoost + CatBoost Ensemble',
        'method': 'Soft Voting (Average Probabilities)',
        'models': ['XGBoost (GPU)', 'CatBoost (GPU)'],
        'created_at': datetime.now().isoformat()
    },
    'performance': {
        'accuracy': round(acc_ensemble, 4),
        'macro_f1': round(f1_ensemble, 4),
        'weighted_f1': round(weighted_f1, 4),
        'category_f1': {cat: round(f1, 4) for cat, f1 in zip(category_names, category_f1)}
    },
    'individual_models': results,
    'model_files': {
        'xgboost': xgb_path,
        'catboost': cat_path
    }
}

metadata_path = f'{output_dir}/ensemble_metadata_{timestamp}.json'
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"  ✅ 메타데이터 저장: {metadata_path}")

# ============================================================
# 7. 성능 비교
# ============================================================
print("\n" + "="*80)
print("🏆 성능 비교 결과")
print("="*80)

print(f"\n{'모델':<25} {'Accuracy':>12} {'Macro F1':>12} {'개선'}")
print("-"*60)
print(f"{'XGBoost (GPU)':<25} {acc_xgb:>12.4f} {f1_xgb:>12.4f} {''}")
print(f"{'CatBoost (GPU)':<25} {acc_cat:>12.4f} {f1_cat:>12.4f} {''}")
print("-"*60)
print(f"{'🎯 Ensemble (XGB+CAT)':<25} {acc_ensemble:>12.4f} {f1_ensemble:>12.4f} {'✅ +{:.2f}%'.format((acc_ensemble - max(acc_xgb, acc_cat))*100)}")

# LightGBM 비교 (기존 프로덕션 모델)
lgb_acc = 0.4913
lgb_f1 = 0.4344
print(f"\n{'기존 LightGBM':<25} {lgb_acc:>12.4f} {lgb_f1:>12.4f}")
print(f"{'앙상블 vs LightGBM':<25} {''} {''} {'차이: {:.2f}%p'.format((acc_ensemble - lgb_acc)*100)}")

print("\n" + "="*80)
print("✅ 앙상블 모델 학습 완료!")
print("="*80)
print(f"\n📦 저장된 파일:")
print(f"   1. XGBoost: {xgb_path}")
print(f"   2. CatBoost: {cat_path}")
print(f"   3. 결과: {results_path}")
print(f"   4. 메타데이터: {metadata_path}")
print("\n" + "="*80)
