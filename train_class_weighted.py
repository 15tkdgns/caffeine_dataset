"""
Class Weight 조정으로 생활 카테고리 성능 개선
불균형 클래스에 집중하여 Macro F1 최적화
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.utils.class_weight import compute_class_weight
import joblib
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time

print("="*80)
print("⚖️ Class Weight 조정으로 불균형 개선 (GPU)")
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

# 클래스 분포 확인
unique, counts = np.unique(y_test, return_counts=True)
category_names = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']
print(f"\n  테스트 데이터 클래스 분포:")
for cat_id, cat_name, count in zip(unique, category_names, counts):
    print(f"     {cat_name:6s}: {count:,}건 ({count/len(y_test)*100:.1f}%)")

# ============================================================
# 2. Class Weight 계산
# ============================================================
print("\n[2/5] Class Weight 계산")

# sklearn으로 자동 계산
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

# 딕셔너리로 변환
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

print(f"  Class Weights (자동 계산):")
for cat_id, cat_name, weight in zip(unique, category_names, class_weights):
    print(f"     {cat_name:6s}: {weight:.4f}")

# 생활 카테고리 가중치 추가 부스팅
class_weight_dict[1] *= 5.0  # 생활 카테고리 (F1 8%)를 5배 강조

print(f"\n  생활 카테고리 부스팅 후:")
for cat_id, cat_name in enumerate(category_names):
    print(f"     {cat_name:6s}: {class_weight_dict[cat_id]:.4f}")

# Sample weight 생성
sample_weights = np.array([class_weight_dict[y] for y in y_train])

# ============================================================
# 3. XGBoost with Class Weight (GPU)
# ============================================================
print("\n[3/5] XGBoost with Class Weight")

print("  학습 시작...")
start = time.time()

xgb_weighted = xgb.XGBClassifier(
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

xgb_weighted.fit(X_train, y_train, sample_weight=sample_weights)
train_time_xgb = time.time() - start

# 평가
y_pred_xgb = xgb_weighted.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb, average='macro')
weighted_f1_xgb = f1_score(y_test, y_pred_xgb, average='weighted')
category_f1_xgb = f1_score(y_test, y_pred_xgb, average=None)

print(f"  ✅ 완료: {train_time_xgb:.2f}초")
print(f"\n  📊 성능:")
print(f"     Accuracy:    {acc_xgb:.4f} ({acc_xgb*100:.2f}%)")
print(f"     Macro F1:    {f1_xgb:.4f} ({f1_xgb*100:.2f}%)")
print(f"     Weighted F1: {weighted_f1_xgb:.4f}")
print(f"\n  카테고리별 F1:")
for cat_name, f1 in zip(category_names, category_f1_xgb):
    print(f"     {cat_name:6s}: {f1:.4f}")

# ============================================================
# 4. LightGBM with Class Weight
# ============================================================
print("\n[4/5] LightGBM with Class Weight")

print("  학습 시작...")
start = time.time()

lgb_weighted = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.1,
    num_leaves=128,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_weighted.fit(X_train, y_train, sample_weight=sample_weights)
train_time_lgb = time.time() - start

# 평가
y_pred_lgb = lgb_weighted.predict(X_test)
acc_lgb = accuracy_score(y_test, y_pred_lgb)
f1_lgb = f1_score(y_test, y_pred_lgb, average='macro')
weighted_f1_lgb = f1_score(y_test, y_pred_lgb, average='weighted')
category_f1_lgb = f1_score(y_test, y_pred_lgb, average=None)

print(f"  ✅ 완료: {train_time_lgb:.2f}초")
print(f"\n  📊 성능:")
print(f"     Accuracy:    {acc_lgb:.4f} ({acc_lgb*100:.2f}%)")
print(f"     Macro F1:    {f1_lgb:.4f} ({f1_lgb*100:.2f}%)")
print(f"     Weighted F1: {weighted_f1_lgb:.4f}")
print(f"\n  카테고리별 F1:")
for cat_name, f1 in zip(category_names, category_f1_lgb):
    print(f"     {cat_name:6s}: {f1:.4f}")

# ============================================================
# 5. 결과 저장 및 비교
# ============================================================
print("\n[5/5] 결과 저장")

import os
os.makedirs('03_models/class_weighted', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 최고 성능 모델 선택
if f1_xgb > f1_lgb:
    best_model = xgb_weighted
    best_name = 'XGBoost (Class Weighted)'
    best_acc = acc_xgb
    best_f1 = f1_xgb
    best_category_f1 = category_f1_xgb
    model_type = 'xgboost'
else:
    best_model = lgb_weighted
    best_name = 'LightGBM (Class Weighted)'
    best_acc = acc_lgb
    best_f1 = f1_lgb
    best_category_f1 = category_f1_lgb
    model_type = 'lightgbm'

# 모델 저장
model_path = f'03_models/class_weighted/{model_type}_weighted_{timestamp}.joblib'
joblib.dump(best_model, model_path)
print(f"  ✅ 모델 저장: {model_path}")

# 메타데이터
metadata = {
    'model_info': {
        'name': best_name,
        'method': 'Class Weight (Balanced + Life x5)',
        'created_at': datetime.now().isoformat()
    },
    'class_weights': {cat: float(class_weight_dict[i]) for i, cat in enumerate(category_names)},
    'performance': {
        'accuracy': round(best_acc, 4),
        'macro_f1': round(best_f1, 4),
        'category_f1': {cat: round(f1, 4) for cat, f1 in zip(category_names, best_category_f1)}
    },
    'comparison': {
        'xgboost': {
            'accuracy': round(acc_xgb, 4),
            'macro_f1': round(f1_xgb, 4),
            'life_f1': round(category_f1_xgb[1], 4)
        },
        'lightgbm': {
            'accuracy': round(acc_lgb, 4),
            'macro_f1': round(f1_lgb, 4),
            'life_f1': round(category_f1_lgb[1], 4)
        }
    }
}

metadata_path = f'03_models/class_weighted/metadata_{timestamp}.json'
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"  ✅ 메타데이터: {metadata_path}")

# ============================================================
# 비교 결과
# ============================================================
print("\n" + "="*80)
print("🏆 Class Weight 조정 결과")
print("="*80)

baseline = {'acc': 0.4913, 'f1': 0.4344, 'life_f1': 0.0802}

print(f"\n{'모델':<35} {'Accuracy':>12} {'Macro F1':>12} {'생활 F1':>12}")
print("-"*75)
print(f"{'Baseline (LightGBM)':<35} {baseline['acc']:>12.4f} {baseline['f1']:>12.4f} {baseline['life_f1']:>12.4f}")
print(f"{'XGBoost (Class Weighted)':<35} {acc_xgb:>12.4f} {f1_xgb:>12.4f} {category_f1_xgb[1]:>12.4f}")
print(f"{'LightGBM (Class Weighted)':<35} {acc_lgb:>12.4f} {f1_lgb:>12.4f} {category_f1_lgb[1]:>12.4f}")
print("-"*75)

print(f"\n생활 카테고리 개선:")
print(f"  XGBoost:  {baseline['life_f1']:.4f} → {category_f1_xgb[1]:.4f} ({(category_f1_xgb[1] - baseline['life_f1'])*100:+.2f}%p)")
print(f"  LightGBM: {baseline['life_f1']:.4f} → {category_f1_lgb[1]:.4f} ({(category_f1_lgb[1] - baseline['life_f1'])*100:+.2f}%p)")

if category_f1_xgb[1] > baseline['life_f1'] or category_f1_lgb[1] > baseline['life_f1']:
    print(f"\n✅ 생활 카테고리 성능 개선 성공!")
else:
    print(f"\n⚠️ 추가 조정 필요")

print("\n" + "="*80)
print("✅ Class Weight 실험 완료!")
print("="*80)
