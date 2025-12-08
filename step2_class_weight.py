"""
STEP 2: Class Weight 적용 + 전략적 언더샘플링
- 생활 카테고리 가중치 증가
- 식료품, 쇼핑, 외식 언더샘플링 (0.7배)
- LightGBM, XGBoost 비교
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib
import json
from datetime import datetime
import time

print("="*80)
print("⚖️ STEP 2: Class Weight + 전략적 언더샘플링")
print("="*80)

# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n[1/6] 데이터 로드")

X_train = np.load('02_data/02_augmented/X_train_smote.npy')
y_train = np.load('02_data/02_augmented/y_train_smote.npy')
X_test = np.load('02_data/02_augmented/X_test.npy')
y_test = np.load('02_data/02_augmented/y_test.npy')

print(f"  원본 학습 데이터: {len(X_train):,}건")
print(f"  테스트 데이터: {len(X_test):,}건")

category_names = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']

# ============================================================
# 2. 전략적 언더샘플링
# ============================================================
print("\n[2/6] 전략적 언더샘플링")

# 분석 결과: 생활과 가장 헷갈리는 클래스
# 1. 식료품 (43,581건 혼동)
# 2. 쇼핑 (29,883건 혼동)  
# 3. 외식 (14,652건 혼동)

# 클래스별 샘플링 비율
sampling_ratios = {
    0: 1.0,   # 교통 - 유지
    1: 1.0,   # 생활 - 유지 (증강하고 싶지만 SMOTE 이미 적용됨)
    2: 0.7,   # 쇼핑 - 70%로 감소 (생활과 헷갈림)
    3: 0.7,   # 식료품 - 70%로 감소 (생활과 가장 헷갈림)
    4: 0.7,   # 외식 - 70%로 감소 (생활과 헷갈림)
    5: 1.0    # 주유 - 유지
}

# 샘플링 수행
indices_to_keep = []
for class_id in range(6):
    class_mask = (y_train == class_id)
    class_indices = np.where(class_mask)[0]
    
    n_samples = len(class_indices)
    n_keep = int(n_samples * sampling_ratios[class_id])
    
    # 랜덤 샘플링
    np.random.seed(42)
    kept_indices = np.random.choice(class_indices, n_keep, replace=False)
    indices_to_keep.extend(kept_indices)
    
    print(f"  {category_names[class_id]:6s}: {n_samples:,}건 → {n_keep:,}건 ({sampling_ratios[class_id]*100:.0f}%)")

indices_to_keep = np.array(indices_to_keep)
X_train_sampled = X_train[indices_to_keep]
y_train_sampled = y_train[indices_to_keep]

print(f"\n  총 학습 데이터: {len(X_train):,}건 → {len(X_train_sampled):,}건")

# ============================================================
# 3. Class Weight 계산
# ============================================================
print("\n[3/6] Class Weight 계산")

# 자동 계산 (balanced)
class_weights_auto = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_sampled),
    y=y_train_sampled
)

print(f"\n  Balanced Class Weights:")
for i, (cat, weight) in enumerate(zip(category_names, class_weights_auto)):
    print(f"     {cat:6s}: {weight:.4f}")

# 수동 조정: 생활 카테고리 3배 부스팅
class_weights_manual = class_weights_auto.copy()
class_weights_manual[1] *= 3.0  # 생활 카테고리

print(f"\n  Manual (생활 3배 부스팅):")
for i, (cat, weight) in enumerate(zip(category_names, class_weights_manual)):
    print(f"     {cat:6s}: {weight:.4f}")

# Sample weight 생성
sample_weights = np.array([class_weights_manual[y] for y in y_train_sampled])

# ============================================================
# 4. LightGBM with Class Weight
# ============================================================
print("\n[4/6] LightGBM with Class Weight")

print("  학습 시작...")
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

lgb_model.fit(X_train_sampled, y_train_sampled, sample_weight=sample_weights)
train_time_lgb = time.time() - start

# 예측
y_pred_lgb = lgb_model.predict(X_test)
y_proba_lgb = lgb_model.predict_proba(X_test)

# 평가
acc_lgb = accuracy_score(y_test, y_pred_lgb)
macro_f1_lgb = f1_score(y_test, y_pred_lgb, average='macro')
weighted_f1_lgb = f1_score(y_test, y_pred_lgb, average='weighted')
category_f1_lgb = f1_score(y_test, y_pred_lgb, average=None)

print(f"  ✅ 학습 완료: {train_time_lgb:.2f}초")
print(f"\n  📊 성능:")
print(f"     Accuracy:    {acc_lgb:.4f} ({acc_lgb*100:.2f}%)")
print(f"     Macro F1:    {macro_f1_lgb:.4f} ({macro_f1_lgb*100:.2f}%)")
print(f"     Weighted F1: {weighted_f1_lgb:.4f}")

print(f"\n  카테고리별 F1:")
for cat, f1 in zip(category_names, category_f1_lgb):
    emoji = "⭐" if f1 > 0.5 else "✅" if f1 > 0.3 else "⚠️"
    improvement = ""
    if cat == "생활":
        baseline_f1 = 0.0802
        imp = (f1 - baseline_f1) * 100
        improvement = f" ({imp:+.2f}%p from baseline)"
    print(f"     {emoji} {cat:6s}: {f1:.4f}{improvement}")

# ============================================================
# 5. XGBoost with Class Weight
# ============================================================
print("\n[5/6] XGBoost with Class Weight")

print("  학습 시작...")
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

xgb_model.fit(X_train_sampled, y_train_sampled, sample_weight=sample_weights)
train_time_xgb = time.time() - start

# 예측
y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)

# 평가
acc_xgb = accuracy_score(y_test, y_pred_xgb)
macro_f1_xgb = f1_score(y_test, y_pred_xgb, average='macro')
weighted_f1_xgb = f1_score(y_test, y_pred_xgb, average='weighted')
category_f1_xgb = f1_score(y_test, y_pred_xgb, average=None)

print(f"  ✅ 학습 완료: {train_time_xgb:.2f}초")
print(f"\n  📊 성능:")
print(f"     Accuracy:    {acc_xgb:.4f} ({acc_xgb*100:.2f}%)")
print(f"     Macro F1:    {macro_f1_xgb:.4f} ({macro_f1_xgb*100:.2f}%)")
print(f"     Weighted F1: {weighted_f1_xgb:.4f}")

print(f"\n  카테고리별 F1:")
for cat, f1 in zip(category_names, category_f1_xgb):
    emoji = "⭐" if f1 > 0.5 else "✅" if f1 > 0.3 else "⚠️"
    improvement = ""
    if cat == "생활":
        baseline_f1 = 0.0802
        imp = (f1 - baseline_f1) * 100
        improvement = f" ({imp:+.2f}%p from baseline)"
    print(f"     {emoji} {cat:6s}: {f1:.4f}{improvement}")

# ============================================================
# 6. 결과 저장 및 비교
# ============================================================
print("\n[6/6] 결과 저장")

import os
os.makedirs('04_logs/step2_class_weight', exist_ok=True)

# 최고 성능 모델 선택
if category_f1_lgb[1] > category_f1_xgb[1]:  # 생활 F1 기준
    best_model = lgb_model
    best_name = 'LightGBM'
    best_metrics = {
        'accuracy': acc_lgb,
        'macro_f1': macro_f1_lgb,
        'category_f1': category_f1_lgb
    }
else:
    best_model = xgb_model
    best_name = 'XGBoost'
    best_metrics = {
        'accuracy': acc_xgb,
        'macro_f1': macro_f1_xgb,
        'category_f1': category_f1_xgb
    }

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 모델 저장
model_path = f'04_logs/step2_class_weight/{best_name.lower()}_weighted_{timestamp}.joblib'
joblib.dump(best_model, model_path)
print(f"  ✅ 최고 모델 저장: {model_path}")

# 메타데이터
metadata = {
    'experiment': 'STEP 2: Class Weight + Undersampling',
    'strategy': {
        'undersampling': sampling_ratios,
        'class_weights': {cat: float(w) for cat, w in zip(category_names, class_weights_manual)},
        'living_boost_factor': 3.0
    },
    'results': {
        'lightgbm': {
            'accuracy': float(acc_lgb),
            'macro_f1': float(macro_f1_lgb),
            'category_f1': {cat: float(f1) for cat, f1 in zip(category_names, category_f1_lgb)},
            'train_time': float(train_time_lgb)
        },
        'xgboost': {
            'accuracy': float(acc_xgb),
            'macro_f1': float(macro_f1_xgb),
            'category_f1': {cat: float(f1) for cat, f1 in zip(category_names, category_f1_xgb)},
            'train_time': float(train_time_xgb)
        },
        'best_model': best_name
    },
    'comparison': {
        'baseline_acc': 0.4913,
        'baseline_macro_f1': 0.4344,
        'baseline_living_f1': 0.0802
    }
}

metadata_path = f'04_logs/step2_class_weight/metadata_{timestamp}.json'
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"  ✅ 메타데이터 저장: {metadata_path}")

# ============================================================
# 7. 최종 비교
# ============================================================
print("\n" + "="*80)
print("🏆 STEP 2 결과 비교")
print("="*80)

baseline = {'acc': 0.4913, 'f1': 0.4344, 'living_f1': 0.0802}

print(f"\n{'모델':<40} {'Accuracy':>12} {'Macro F1':>12} {'생활 F1':>12}")
print("-"*80)
print(f"{'Baseline (원본)':<40} {baseline['acc']:>12.4f} {baseline['f1']:>12.4f} {baseline['living_f1']:>12.4f}")
print(f"{'LightGBM (Weight+Undersample)':<40} {acc_lgb:>12.4f} {macro_f1_lgb:>12.4f} {category_f1_lgb[1]:>12.4f}")
print(f"{'XGBoost (Weight+Undersample)':<40} {acc_xgb:>12.4f} {macro_f1_xgb:>12.4f} {category_f1_xgb[1]:>12.4f}")
print("-"*80)

print(f"\n📊 변화량:")
print(f"  LightGBM:")
print(f"    Accuracy:  {(acc_lgb - baseline['acc'])*100:+.2f}%p")
print(f"    Macro F1:  {(macro_f1_lgb - baseline['f1'])*100:+.2f}%p")
print(f"    생활 F1:   {(category_f1_lgb[1] - baseline['living_f1'])*100:+.2f}%p ⭐")

print(f"\n  XGBoost:")
print(f"    Accuracy:  {(acc_xgb - baseline['acc'])*100:+.2f}%p")
print(f"    Macro F1:  {(macro_f1_xgb - baseline['f1'])*100:+.2f}%p")
print(f"    생활 F1:   {(category_f1_xgb[1] - baseline['living_f1'])*100:+.2f}%p ⭐")

# 평가
living_improved = category_f1_lgb[1] > baseline['living_f1'] or category_f1_xgb[1] > baseline['living_f1']
acc_maintained = acc_lgb > 0.45 or acc_xgb > 0.45

print(f"\n🎯 평가:")
if living_improved and acc_maintained:
    print(f"  ✅✅ 성공! 생활 F1 개선 + Accuracy 유지")
elif living_improved:
    print(f"  ✅ 생활 F1 개선됨, 하지만 Accuracy 하락")
else:
    print(f"  ⚠️ 추가 조정 필요")

print("\n" + "="*80)
print("✅ STEP 2 완료!")
print("="*80)
print(f"\n📦 다음 단계:")
print(f"   STEP 3: SMOTE/ADASYN (생활 카테고리만 증강)")
print(f"   STEP 4: Focal Loss 도입")
print("="*80)
