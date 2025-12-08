"""
STEP 3: ADASYN - 생활 카테고리 집중 증강
- 생활 카테고리만 2.5배 증강
- Class Weight 유지
- 목표: Accuracy 회복 + 생활 F1 유지
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.utils.class_weight import compute_class_weight
import joblib
import json
from datetime import datetime
import time

print("="*80)
print("🔬 STEP 3: ADASYN - 생활 카테고리 집중 증강")
print("="*80)

# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n[1/7] 데이터 로드")

X_train = np.load('02_data/02_augmented/X_train_smote.npy')
y_train = np.load('02_data/02_augmented/y_train_smote.npy')
X_test = np.load('02_data/02_augmented/X_test.npy')
y_test = np.load('02_data/02_augmented/y_test.npy')

print(f"  원본 학습 데이터: {len(X_train):,}건")

category_names = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']

# ============================================================
# 2. 전략적 언더샘플링 (STEP 2와 동일)
# ============================================================
print("\n[2/7] 전략적 언더샘플링")

sampling_ratios = {
    0: 1.0,   # 교통
    1: 1.0,   # 생활 (증강 전)
    2: 0.7,   # 쇼핑
    3: 0.7,   # 식료품
    4: 0.7,   # 외식
    5: 1.0    # 주유
}

indices_to_keep = []
for class_id in range(6):
    class_mask = (y_train == class_id)
    class_indices = np.where(class_mask)[0]
    n_samples = len(class_indices)
    n_keep = int(n_samples * sampling_ratios[class_id])
    np.random.seed(42)
    kept_indices = np.random.choice(class_indices, n_keep, replace=False)
    indices_to_keep.extend(kept_indices)
    print(f"  {category_names[class_id]:6s}: {n_samples:,}건 → {n_keep:,}건")

X_train_sampled = X_train[np.array(indices_to_keep)]
y_train_sampled = y_train[np.array(indices_to_keep)]

print(f"\n  샘플링 후: {len(X_train_sampled):,}건")

# ============================================================
# 3. ADASYN - 생활 카테고리만 증강
# ============================================================
print("\n[3/7] ADASYN - 생활 카테고리 집중 증강")

# 현재 클래스 분포
unique, counts = np.unique(y_train_sampled, return_counts=True)
print(f"\n  증강 전 클래스 분포:")
for cat_id, count in zip(unique, counts):
    print(f"     {category_names[cat_id]:6s}: {count:,}건")

# 생활 카테고리를 다른 클래스 평균 수준으로 증강
생활_count = counts[1]
other_avg = counts[[0, 5]].mean()  # 교통, 주유 평균
target_생활_count = int(other_avg * 1.0)  # 평균 수준

print(f"\n  생활 카테고리 증강:")
print(f"     현재: {생활_count:,}건")
print(f"     목표: {target_생활_count:,}건")
print(f"     증강: {target_생활_count - 생활_count:,}건 추가")

# 생활 카테고리 인덱스
생활_indices = np.where(y_train_sampled == 1)[0]
생활_X = X_train_sampled[생활_indices]
생활_y = y_train_sampled[생활_indices]

# ADASYN 적용 (생활 카테고리만)
try:
    from imblearn.over_sampling import ADASYN
    
    # 생활(1)과 다른 클래스(0) 두 클래스로 변환
    other_indices = np.where(y_train_sampled != 1)[0]
    np.random.seed(42)
    other_sample_indices = np.random.choice(other_indices, len(생활_indices)*2, replace=False)
    
    temp_X = np.vstack([생활_X, X_train_sampled[other_sample_indices]])
    temp_y = np.hstack([np.ones(len(생활_X)), np.zeros(len(other_sample_indices))])
    
    # ADASYN으로 생활 증강
    adasyn = ADASYN(sampling_strategy='minority', random_state=42, n_neighbors=5)
    X_resampled, y_resampled = adasyn.fit_resample(temp_X, temp_y)
    
    # 증강된 생활 샘플만 추출
    생활_mask = y_resampled == 1
    생활_augmented_X = X_resampled[생활_mask]
    
    # 원본 생활 제외하고 증강된 것만
    n_original_생활 = len(생활_X)
    생활_new_X = 생활_augmented_X[n_original_생활:]
    
    # 목표 개수만큼만 추가
    n_to_add = min(len(생활_new_X), target_생활_count - 생활_count)
    생활_new_X = 생활_new_X[:n_to_add]
    생활_new_y = np.ones(n_to_add, dtype=int)
    
    # 원본 데이터에 증강 데이터 추가
    X_train_final = np.vstack([X_train_sampled, 생활_new_X])
    y_train_final = np.hstack([y_train_sampled, 생활_new_y])
    
    print(f"  ✅ ADASYN 증강 완료: {n_to_add:,}건 추가")
    
except Exception as e:
    print(f"  ⚠️ ADASYN 실패 ({e}), SMOTE 사용")
    
    # SMOTE 대체
    smote = SMOTE(sampling_strategy={1: target_생활_count}, random_state=42, k_neighbors=5)
    
    # 임시로 생활만 증강
    temp_sampling_strategy = {
        0: counts[0],
        1: target_생활_count,
        2: counts[2],
        3: counts[3],
        4: counts[4],
        5: counts[5]
    }
    
    X_train_final, y_train_final = smote.fit_resample(X_train_sampled, y_train_sampled)
    print(f"  ✅ SMOTE 증강 완료")

# 최종 분포
unique_final, counts_final = np.unique(y_train_final, return_counts=True)
print(f"\n  증강 후 클래스 분포:")
for cat_id, count in zip(unique_final, counts_final):
    print(f"     {category_names[cat_id]:6s}: {count:,}건")

print(f"\n  총 학습 데이터: {len(X_train_sampled):,}건 → {len(X_train_final):,}건")

# ============================================================
# 4. Class Weight 계산
# ============================================================
print("\n[4/7] Class Weight 계산")

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_final),
    y=y_train_final
)

# 생활 2배 부스팅 (STEP 2는 3배였지만, 데이터 증강했으므로 조정)
class_weights[1] *= 2.0

print(f"\n  Class Weights (생활 2배):")
for cat, weight in zip(category_names, class_weights):
    print(f"     {cat:6s}: {weight:.4f}")

sample_weights = np.array([class_weights[y] for y in y_train_final])

# ============================================================
# 5. LightGBM with ADASYN + Weight
# ============================================================
print("\n[5/7] LightGBM with ADASYN + Weight")

start = time.time()
lgb_model = lgb.LGBMClassifier(
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

lgb_model.fit(X_train_final, y_train_final, sample_weight=sample_weights)
train_time_lgb = time.time() - start

y_pred_lgb = lgb_model.predict(X_test)
acc_lgb = accuracy_score(y_test, y_pred_lgb)
macro_f1_lgb = f1_score(y_test, y_pred_lgb, average='macro')
category_f1_lgb = f1_score(y_test, y_pred_lgb, average=None)

print(f"  ✅ 학습 완료: {train_time_lgb:.2f}초")
print(f"\n  📊 성능:")
print(f"     Accuracy:    {acc_lgb:.4f} ({acc_lgb*100:.2f}%)")
print(f"     Macro F1:    {macro_f1_lgb:.4f}")

print(f"\n  카테고리별 F1:")
for cat, f1 in zip(category_names, category_f1_lgb):
    emoji = "⭐" if f1 > 0.5 else "✅" if f1 > 0.3 else "⚠️"
    if cat == "생활":
        print(f"     {emoji} {cat:6s}: {f1:.4f} (Baseline: 0.0802, STEP2: 0.2541)")
    else:
        print(f"     {emoji} {cat:6s}: {f1:.4f}")

# ============================================================
# 6. XGBoost with ADASYN + Weight  
# ============================================================
print("\n[6/7] XGBoost with ADASYN + Weight")

start = time.time()
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

xgb_model.fit(X_train_final, y_train_final, sample_weight=sample_weights)
train_time_xgb = time.time() - start

y_pred_xgb = xgb_model.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
macro_f1_xgb = f1_score(y_test, y_pred_xgb, average='macro')
category_f1_xgb = f1_score(y_test, y_pred_xgb, average=None)

print(f"  ✅ 학습 완료: {train_time_xgb:.2f}초")
print(f"\n  📊 성능:")
print(f"     Accuracy:    {acc_xgb:.4f} ({acc_xgb*100:.2f}%)")
print(f"     Macro F1:    {macro_f1_xgb:.4f}")

print(f"\n  카테고리별 F1:")
for cat, f1 in zip(category_names, category_f1_xgb):
    emoji = "⭐" if f1 > 0.5 else "✅" if f1 > 0.3 else "⚠️"
    if cat == "생활":
        print(f"     {emoji} {cat:6s}: {f1:.4f} (Baseline: 0.0802, STEP2: 0.2522)")
    else:
        print(f"     {emoji} {cat:6s}: {f1:.4f}")

# ============================================================
# 7. 결과 저장
# ============================================================
print("\n[7/7] 결과 저장")

import os
os.makedirs('04_logs/step3_adasyn', exist_ok=True)

best_model = lgb_model if acc_lgb >= acc_xgb else xgb_model
best_name = 'LightGBM' if acc_lgb >= acc_xgb else 'XGBoost'

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f'04_logs/step3_adasyn/{best_name.lower()}_adasyn_{timestamp}.joblib'
joblib.dump(best_model, model_path)

metadata = {
    'experiment': 'STEP 3: ADASYN + Class Weight',
    'strategy': {
        'method': 'ADASYN on Living category only',
        'augmentation': f'Living: {생활_count} → {counts_final[1]} (+{counts_final[1]-생활_count})',
        'class_weights': {cat: float(w) for cat, w in zip(category_names, class_weights)}
    },
    'results': {
        'lightgbm': {
            'accuracy': float(acc_lgb),
            'macro_f1': float(macro_f1_lgb),
            'category_f1': {cat: float(f1) for cat, f1 in zip(category_names, category_f1_lgb)}
        },
        'xgboost': {
            'accuracy': float(acc_xgb),
            'macro_f1': float(macro_f1_xgb),
            'category_f1': {cat: float(f1) for cat, f1 in zip(category_names, category_f1_xgb)}
        }
    }
}

with open(f'04_logs/step3_adasyn/metadata_{timestamp}.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"  ✅ 저장 완료")

# ============================================================
# 최종 비교
# ============================================================
print("\n" + "="*80)
print("🏆 전체 STEP 비교")
print("="*80)

print(f"\n{'단계':<25} {'Accuracy':>12} {'Macro F1':>12} {'생활 F1':>12}")
print("-"*65)
print(f"{'Baseline':<25} {0.4913:>12.4f} {0.4344:>12.4f} {0.0802:>12.4f}")
print(f"{'STEP 2 (Weight+Under)':<25} {0.4485:>12.4f} {0.4287:>12.4f} {0.2541:>12.4f}")
print(f"{'STEP 3 LGB (ADASYN)':<25} {acc_lgb:>12.4f} {macro_f1_lgb:>12.4f} {category_f1_lgb[1]:>12.4f}")
print(f"{'STEP 3 XGB (ADASYN)':<25} {acc_xgb:>12.4f} {macro_f1_xgb:>12.4f} {category_f1_xgb[1]:>12.4f}")
print("-"*65)

print(f"\n🎯 목표 달성 여부:")
acc_improved = max(acc_lgb, acc_xgb) > 0.4485
생활_maintained = min(category_f1_lgb[1], category_f1_xgb[1]) >= 0.20

if acc_improved and 생활_maintained:
    print(f"  ✅✅ 성공! Accuracy 회복 + 생활 F1 유지")
elif 생활_maintained:
    print(f"  ✅ 생활 F1 유지, Accuracy는 비슷")
else:
    print(f"  ⚠️ 추가 조정 필요")

print("\n=" *80)
print("✅ STEP 3 완료!")
print("="*80)
print(f"\n📦 다음 단계: STEP 4 - Focal Loss")
print("="*80)
